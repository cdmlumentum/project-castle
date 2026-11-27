"""Isolated, budgeted test of sharing the cross-state lower mode with bar.

This diagnostic never changes production coefficients, caches or block handoffs.
All optical samples come from the native MODE core. Synthetic inputs exist only
in tests and cannot produce an eligible production recommendation.
"""
from dataclasses import asdict, dataclass, replace
from importlib.resources import files
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
from scipy.interpolate import PchipInterpolator

from .block0 import base_profiles, certified_profile, installation_identity, margins
from .block_io import digest, load_block, source_record, write_json
from .geometry import ClampedCubicProfile, make_geometry
from .materials import build_material_bank
from .modes import (ModeReference, ModeSnapshot, InvalidModeError, field_overlap,
                    select_mode, select_supermode_pair, extract_coupling)
from .profiles import minimax_objective
from .propagate import propagate_converged, port_metrics
from .runtime import BudgetExceeded, RunBudget

METHOD = 'cross_reference_lower_mode_check_v1'


@dataclass(frozen=True)
class Limits:
    minimum_squared_field_overlap: float = .9999
    phase_rad: float = .01
    attenuation_db: float = .01
    coefficient_fraction: float = .001
    il_db: float = .01
    xt_db: float = .1
    absolute_port_power: float = 1e-6
    midpoint_fraction: float = .01
    uncertainty_fraction: float = 1/3


class NativeProbe:
    """Fresh raw candidates with the production native geometry and full domain."""
    synthetic = False

    def __init__(self, config):
        self.config, self.backend = config, None

    def __call__(self, geometry, wavelength, kind, profile):
        from .native_backend import NativeLumapiBackend, scalar, snapshot_from_native
        if self.backend is None:
            class CheckBackend(NativeLumapiBackend):
                def native_config(self, mesh_accuracy=None):
                    value = super().native_config(mesh_accuracy)
                    value['full_profile'] = self.check_profile.native()
                    # Same physical comparison coordinates for every numerical variant.
                    value['comparison_enabled'] = 1.
                    value['comparison_x_m'], value['comparison_y_m'] = (
                        v.reshape(-1, 1) for v in self.check_profile.grid())
                    return value
            self.backend = CheckBackend(self.config)
        self.backend.check_profile = profile
        value = self.backend._call_native('vlac_raw', [asdict(geometry), float(wavelength),
            float(kind != 'lower'), float(kind != 'upper'), 0.])
        self.actual_mesh = [v.get('mesh') for v in value['items'][:int(scalar(value['n']))]]
        return ([snapshot_from_native(v) for v in value['items'][:int(scalar(value['n']))]],
                int(scalar(value['attempts'])))

    def close(self):
        if self.backend is not None:
            self.backend.close()


def mode_difference(a, b, wavelength, length):
    """Keep complex beta; field metric is similarity, not physical power flux."""
    delta = 2*np.pi*(b.neff-a.neff)/wavelength
    return {'squared_field_overlap': field_overlap(a, b)**2,
            'delta_neff': b.neff-a.neff, 'delta_beta_rad_per_m': delta,
            'phase_error_rad': abs(delta.real)*length,
            'attenuation_error_db': 20/np.log(10)*abs(delta.imag)*length}


def mode_pass(value, limits, fraction=1.):
    return bool(value['phase_error_rad'] <= limits.phase_rad*fraction
        and value['attenuation_error_db'] <= limits.attenuation_db*fraction
        and 1-value['squared_field_overlap'] <= (1-limits.minimum_squared_field_overlap)*fraction)


def select_isolated(candidates, config, geometry, wavelength, guide, references=()):
    selected, overlap = select_mode(candidates, previous=tuple(references), guide=guide,
                                   minimum_te_fraction=config.solver.minimum_te_fraction)
    values = [selected.neff, selected.te_fraction, selected.core_fraction, overlap]
    if not np.isfinite(values).all() or not np.isfinite([selected.ex, selected.ey]).all():
        raise InvalidModeError('Non-finite isolated mode data')
    if (selected.te_fraction < config.solver.minimum_te_fraction or selected.core_fraction <= 0
            or (references and overlap < config.solver.minimum_mode_overlap)):
        raise InvalidModeError('Isolated mode failed polarization, confinement or tracking checks')
    bank = build_material_bank(config)
    clad = float(np.real(bank.silica.index(wavelength)))
    if guide == 'upper':
        clad = max(clad, geometry.liquid_index)
    if selected.neff.real <= clad+config.endpoint_search.pso_guided_neff_margin:
        raise InvalidModeError('Isolated mode failed the configured guiding-index check')
    if references:
        target = getattr(references[0], guide)
        # A near-tied field match cannot establish branch identity.
        best = field_overlap(selected, target)
        if any(m is not selected and m.te_fraction >= config.solver.minimum_te_fraction
               and field_overlap(m, target) >= best-1e-4 for m in candidates):
            raise InvalidModeError('Ambiguous isolated-mode field match')
    return selected, overlap


def coefficient_pair(upper, lower, candidates, references, previous_kappa, config, wavelength):
    plus, minus, overlap = select_supermode_pair(candidates, upper, lower,
        previous=tuple(references), minimum_te_fraction=config.solver.minimum_te_fraction)
    scale = 2*np.pi/wavelength
    ul = field_overlap(upper, lower)
    u, l = field_overlap(plus, upper)**2, field_overlap(plus, lower)**2
    check = float('nan')
    if u+l > 0 and ul <= config.solver.field_overlap_crosscheck_max_basis_overlap:
        check = abs(scale*(plus.neff-minus.neff)/2)*2*np.sqrt(u*l)/(u+l)
    coefficient = extract_coupling(*(scale*m.neff for m in (upper, lower, plus, minus)),
        previous_kappa=previous_kappa, kappa_profile_check=check,
        overlap_upper_lower=ul, min_mode_overlap=overlap,
        identities=(upper.identity, lower.identity, plus.identity, minus.identity))
    if overlap < config.solver.minimum_mode_overlap:
        coefficient.extraction_flag = 'low_mode_overlap'
    if min(plus.te_fraction, minus.te_fraction) < config.solver.minimum_te_fraction:
        coefficient.extraction_flag = 'rejected_supermode_polarization'
    return coefficient, ModeReference(upper, lower, (plus, minus))


def profile_positions(config, solution):
    """Three base points, their independent midpoints, and zero-offset roots."""
    length = config.device.length_m
    coarse = np.linspace(0, length, 3)
    offset = ClampedCubicProfile(length, solution.d_c2c_control_m)
    from scipy.interpolate import PPoly
    roots = PPoly.from_spline(offset._spline).roots(extrapolate=False)*length
    roots = roots[np.isfinite(roots) & (roots > 0) & (roots < length)]
    fine = np.unique(np.round(np.r_[coarse, (coarse[:-1]+coarse[1:])/2, roots]/length, 12))*length
    return coarse, fine


def prepare(config, stage, profile_dir=None, limits=Limits()):
    config.validate()
    if stage not in ('screen', 'full'):
        raise ValueError('Stage must be screen or full')
    endpoint = solution = source = None
    if profile_dir:
        endpoint, solution, _, artifact = load_block(profile_dir, 2, config)
        source = source_record(artifact)
    elif stage == 'full':
        raise ValueError('--stage full requires --profile-dir with a validated Block 2 handoff')
    elif any(a != b for a, b in (config.geometry.h_gap_bounds_m, config.geometry.h_rib_bounds_m)):
        raise ValueError('Screen without a saved profile requires fixed gap and slab thickness')
    base = certified_profile(config, 'coupled') if config.solver.calibration else base_profiles(config)['coupled']
    # Preserve certified local overrides when refining mesh and expanding domain.
    mesh = replace(base, dx_m=base.dx_m/1.5, dy_m=base.dy_m/1.5,
                   local_dx_m=base.local_dx_m/1.5, local_dy_m=base.local_dy_m/1.5)
    domain = margins(config, base, 'coupled', 1.5)
    grid = dict(tracking_x_span_m=base.x_span_m, tracking_y_min_m=base.y_min_m,
                tracking_y_max_m=base.y_max_m, tracking_nx=config.solver.tracking_grid_nx,
                tracking_ny=config.solver.tracking_grid_ny)
    if config.solver.calibration:
        grid = {k: getattr(base, k) for k in grid}
    profiles = {k: replace(p, **grid) for k, p in [('production', base), ('mesh', mesh), ('domain', domain)]}
    wavelengths = np.linspace(config.device.wavelength_min_m, config.device.wavelength_max_m, 3)
    coarse, fine = profile_positions(config, solution) if stage == 'full' else (np.array([]), np.array([]))
    plan = {'method': METHOD, 'stage': stage, 'wavelengths_m': wavelengths.tolist(),
        'coarse_z_m': coarse.tolist(), 'z_m': fine.tolist(), 'source': source,
        'profiles': {k: asdict(v) for k, v in profiles.items()}, 'limits': asdict(limits),
        'nominal_fde_calls': 20+4*len(fine)*len(wavelengths),
        'coverage': 'Three wavelengths (edges and center); five z points plus zero-offset roots. No continuous-band or optimizer-envelope claim.',
        'scope': 'lower-mode approximation on this physical stack and fixed sampled profile; no optimizer or physical-device validation',
        'field_metric': 'Squared normalized transverse E overlap; not an E/H power integral',
        'wall_time_policy': 'Soft boundary between FDE calls and propagation cases; active calls may overrun',
        'production_method_changed': False}
    return plan, endpoint, solution, profiles


class CheckRun:
    def __init__(self, config, output, plan, probe, max_solves, max_seconds, resume, run_id):
        if not isinstance(max_solves, int) or max_solves < 1 or not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('Run budgets must be positive and finite')
        self.config, self.out, self.plan, self.probe = config, Path(output).resolve(), plan, probe
        self.budget = RunBudget(type('Settings', (), {'max_solves': max_solves, 'max_seconds': max_seconds})())
        self.started = time.monotonic()
        self.confirmed_solves = self.uncertain_attempts = 0
        self.hits = 0
        self.rows = []
        self.identity = digest({'plan': plan, 'config': config.as_dict(),
            'implementation': {name: hashlib.sha256(files('vlac').joinpath(name).read_bytes()).hexdigest()
                for name in ('lower_state_check.py', 'modes.py', 'geometry.py', 'materials.py',
                             'native_backend.py', 'native/vlac_core.lsf', 'propagate.py')},
            'installation': installation_identity(config), 'synthetic': bool(probe.synthetic)})
        previous = self.out/'plan.json'
        if previous.exists():
            if not resume:
                raise FileExistsError('Check output exists; use --resume or a new directory')
            if json.loads(previous.read_text())['identity'] != self.identity:
                raise ValueError('Cannot resume changed inputs, source, implementation or MODE installation')
        elif resume:
            raise FileNotFoundError('No check plan exists to resume')
        elif self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError('Check output must be empty; production outputs are never overwritten')
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out/'samples').mkdir(exist_ok=True)
        write_json(previous, dict(plan, identity=self.identity,
            budgets={'max_solves': max_solves, 'max_seconds': max_seconds, 'scope': 'per invocation'}))
        self.report = dict(method=METHOD, stage=plan['stage'], identity=self.identity, run_id=run_id,
            status='running', execution_completed=False, synthetic=bool(probe.synthetic),
            eligible_for_shared_lower_mode=False, scope=plan['scope'], screen=[], numerical_checks=[],
            coefficient_comparisons=[], propagation=[], midpoint_checks=[], production_method_changed=False)
        self.save()

    def save(self):
        self.report.update(fresh_fde_calls=self.confirmed_solves,
                           accounted_fde_attempts_upper_bound=self.budget.solves,
                           unconfirmed_attempts_reserved=self.uncertain_attempts, reused_sample_packets=self.hits,
                           elapsed_seconds=time.monotonic()-self.started)
        write_json(self.out/'report.json', self.report)

    def raw(self, geometry, wavelength, kind, profile, repeat=0):
        self.budget.check()
        key = digest({'geometry': asdict(geometry), 'wavelength_m': wavelength,
                      'kind': kind, 'profile': asdict(profile), 'repeat': repeat})
        path = self.out/'samples'/key
        meta, arrays = path.with_suffix('.json'), path.with_suffix('.npz')
        if meta.exists() and arrays.exists():
            value = json.loads(meta.read_text())
            signature = value.pop('metadata_sha256', None)
            if signature != digest(value):
                raise ValueError('Saved sample metadata checksum mismatch')
            if value['sha256'] != hashlib.sha256(arrays.read_bytes()).hexdigest():
                raise ValueError('Saved field sample checksum mismatch')
            with np.load(arrays, allow_pickle=False) as a:
                modes = [ModeSnapshot(complex(**m['neff']) if isinstance(m['neff'], dict) else complex(m['neff']),
                    a[f'ex_{i}'].copy(), a[f'ey_{i}'].copy(), m['te_fraction'], m['core_fraction'], m['identity'])
                    for i, m in enumerate(value['modes'])]
            self.hits += 1
            return modes
        reserve = 1+self.config.solver.zero_mode_retries
        self.budget.check(reserve)
        print(f'Lower-state check: {kind} {geometry.state}, {wavelength*1e9:.0f} nm; '
              f'{profile.nx}x{profile.ny}; completed FDE calls={self.budget.solves}', flush=True)
        try:
            modes, count = self.probe(geometry, wavelength, kind, profile)
        except BaseException:
            self.budget.consume(reserve)  # Conservative after an interrupted native call.
            self.uncertain_attempts += reserve
            self.save()
            raise
        self.budget.consume(count)
        self.confirmed_solves += count
        if not modes or not 1 <= count <= reserve:
            raise InvalidModeError('Empty mode list or invalid native solve count')
        for m in modes:
            if not np.isfinite([m.neff, m.te_fraction, m.core_fraction]).all() or not np.isfinite([m.ex, m.ey]).all():
                raise InvalidModeError('Non-finite raw mode sample')
            if m.ex.shape != (profile.tracking_nx, profile.tracking_ny) or m.ey.shape != m.ex.shape:
                raise InvalidModeError('Raw mode fields disagree with the common physical comparison grid')
        np.savez_compressed(arrays, **{f'{axis}_{i}': getattr(m, axis)
            for i, m in enumerate(modes) for axis in ('ex', 'ey')},
            x_m=profile.grid()[0], y_m=profile.grid()[1])
        metadata = {'modes': [{k: v for k, v in asdict(m).items() if k not in ('ex', 'ey')} for m in modes],
            'geometry': asdict(geometry), 'wavelength_m': wavelength, 'kind': kind,
            'profile': asdict(profile), 'repeat': repeat, 'solves': count,
            'actual_mesh': getattr(self.probe, 'actual_mesh', None),
            'sha256': hashlib.sha256(arrays.read_bytes()).hexdigest()}
        write_json(meta, dict(metadata, metadata_sha256=digest(metadata)))
        self.save()
        return modes


def run_check(config, output, *, stage='screen', profile_dir=None, max_solves=120,
              max_seconds=3300., resume=False, limits=Limits(), probe_factory=NativeProbe, run_id=None):
    plan, endpoint, solution, profiles = prepare(config, stage, profile_dir, limits)
    # A dedicated backend and dedicated raw sample store; never open production caches.
    probe_config = replace(config, execution=replace(config.execution, max_solves=max_solves, max_seconds=max_seconds))
    probe = probe_factory(probe_config)
    run = None
    try:
        run = CheckRun(config, output, plan, probe, max_solves, max_seconds, resume, run_id or uuid.uuid4().hex)
        bank, g, length = build_material_bank(config), config.geometry, config.device.length_m
        def geometry(state, wavelength, width=None, offset=0.):
            liquid = bank.liquid_cross if state == 'cross' else bank.liquid_bar
            return make_geometry(g, w1_m=float(width if width is not None else g.w1_min_m),
                h_gap_m=endpoint.h_gap_m if endpoint else g.h_gap_bounds_m[0],
                h_rib_m=endpoint.h_rib_m if endpoint else g.h_rib_bounds_m[0], d_c2c_m=float(offset),
                state=state, liquid_index=float(liquid.index(wavelength)))
        def ref(mode):
            return ModeReference(mode, mode, (mode, mode))
        lower = {}
        for variant, profile in profiles.items():
            previous = {}
            for wavelength in plan['wavelengths_m']:
                for state in ('cross', 'bar'):
                    geom = geometry(state, wavelength)
                    anchor = (lower['production', state, wavelength] if variant != 'production'
                              else lower.get((variant, 'cross', wavelength)) if state == 'bar'
                              else previous.get(state))
                    mode, _ = select_isolated(run.raw(geom, wavelength, 'lower', profile), config,
                        geom, wavelength, 'lower', () if anchor is None else (ref(anchor),))
                    lower[variant, state, wavelength] = previous[state] = mode
                difference = mode_difference(lower[variant, 'cross', wavelength], lower[variant, 'bar', wavelength], wavelength, length)
                run.report['screen'].append(dict(variant=variant, wavelength_m=wavelength,
                    passed=mode_pass(difference, limits), **difference))
                if variant != 'production':
                    for state in ('cross', 'bar'):
                        numerical = mode_difference(lower['production', state, wavelength], lower[variant, state, wavelength], wavelength, length)
                        run.report['numerical_checks'].append(dict(variant=variant, state=state, wavelength_m=wavelength,
                            passed=mode_pass(numerical, limits, limits.uncertainty_fraction), **numerical))
                run.save()
        worst = max((r for r in run.report['screen'] if r['variant'] == 'production'),
                    key=lambda r: max(r['phase_error_rad']/limits.phase_rad, r['attenuation_error_db']/limits.attenuation_db))['wavelength_m']
        for state in ('cross', 'bar'):
            geom = geometry(state, worst)
            original = lower['production', state, worst]
            repeated, _ = select_isolated(run.raw(geom, worst, 'lower', profiles['production'], repeat=1),
                config, geom, worst, 'lower', (ref(original),))
            difference = mode_difference(original, repeated, worst, length)
            run.report['numerical_checks'].append(dict(variant='fresh_repeat', state=state, wavelength_m=worst,
                passed=mode_pass(difference, limits, limits.uncertainty_fraction), **difference))
        if not all(r['passed'] for r in run.report['numerical_checks']):
            run.report.update(status='inconclusive', reason='Mesh/domain/repeat uncertainty exceeds one third of the approximation budget')
        elif not all(r['passed'] for r in run.report['screen']):
            run.report.update(status='failed', reason='Lower-mode state difference exceeds a proposed acceptance limit')
        elif stage == 'screen':
            run.report.update(status='screen_passed', reason='Coefficient and propagation substitution have not been checked')
        else:
            _check_profile(run, config, endpoint, solution, geometry, lower, profiles['production'], limits)
        run.report['execution_completed'] = True
        if probe.synthetic and run.report['status'] in ('passed', 'screen_passed'):
            run.report.update(status='inconclusive', reason='Synthetic test fixtures cannot qualify physical optical reuse')
        run.report['eligible_for_shared_lower_mode'] = run.report['status'] == 'passed' and not probe.synthetic
    except BudgetExceeded as exc:
        if run is None:
            raise
        run.report.update(status='incomplete', reason=str(exc))
    except Exception as exc:
        if run is None:
            raise
        run.report.update(status='failed', reason=str(exc), error_type=type(exc).__name__)
    finally:
        try:
            probe.close()
        finally:
            if run is not None:
                run.save()
    return run.report


def _check_profile(run, config, endpoint, solution, geometry, lower, profile, limits):
    z = np.asarray(run.plan['z_m'])
    coarse = np.asarray(run.plan['coarse_z_m'])
    w = ClampedCubicProfile(config.device.length_m, solution.w1_control_m)
    d = ClampedCubicProfile(config.device.length_m, solution.d_c2c_control_m)
    scale = endpoint.delta_beta_max_rad_per_m
    healthy, approximation_ok, midpoints_ok, propagation_ok = True, True, True, True
    for state in ('cross', 'bar'):
        previous_lambda = [None]*len(z)
        approximate_previous_lambda = [None]*len(z)
        for wavelength in run.plan['wavelengths_m']:
            baseline, substituted, current, approximate_current = [], [], [], []
            previous = previous_approximate = None
            previous_kappa = previous_approximate_kappa = None
            for iz, position in enumerate(z):
                geom = geometry(state, wavelength, w(position), d(position))
                references = [v for v in (previous, previous_lambda[iz]) if v is not None]
                upper, _ = select_isolated(run.raw(geom, wavelength, 'upper', profile), config,
                                           geom, wavelength, 'upper', references)
                raw = run.raw(geom, wavelength, 'coupled', profile)
                own_lower = lower['production', state, wavelength]
                shared_lower = lower['production', 'cross', wavelength]
                base, previous = coefficient_pair(upper, own_lower, raw, references, previous_kappa, config, wavelength)
                # Keep actual coupled eigenvalues fixed to isolate this substitution.
                approx = extract_coupling(base.beta_upper, (2*np.pi/wavelength)*shared_lower.neff,
                    base.beta_plus, base.beta_minus, previous_kappa=previous_approximate_kappa,
                    kappa_profile_check=base.kappa_profile_check,
                    overlap_upper_lower=field_overlap(upper, shared_lower),
                    min_mode_overlap=base.min_mode_overlap,
                    identities=(upper.identity, shared_lower.identity, *base.supermode_identities))
                # Also test whether using the shared field would select a different pair.
                approximate_references = [v for v in (previous_approximate, approximate_previous_lambda[iz]) if v is not None]
                chosen, previous_approximate = coefficient_pair(upper, shared_lower, raw,
                    approximate_references,
                    previous_approximate_kappa, config, wavelength)
                approx.kappa_profile_check = chosen.kappa_profile_check
                same_pair = ({m.fingerprint() for m in previous.supermodes}
                             == {m.fingerprint() for m in previous_approximate.supermodes})
                delta_error = abs(approx.delta_beta-base.delta_beta)/scale
                kappa_error = abs(approx.kappa-base.kappa)/(scale/2)
                valid = base.extraction_flag == approx.extraction_flag == chosen.extraction_flag == 'ok' and same_pair
                healthy &= valid
                passed = valid and delta_error <= limits.coefficient_fraction and kappa_error <= limits.coefficient_fraction
                approximation_ok &= passed
                run.report['coefficient_comparisons'].append(dict(state=state, wavelength_m=wavelength, z_m=float(position),
                    baseline=asdict(base), shared_lower=asdict(approx), passed=bool(passed),
                    same_coupled_pair=bool(same_pair), baseline_valid=bool(valid),
                    normalized_detuning_error=delta_error, normalized_complex_kappa_error=kappa_error))
                if not valid:
                    run.report.update(status='inconclusive',
                        reason='Invalid reference extraction or changed coupled pair; stopped before further FDE work')
                    return
                baseline.append(base); substituted.append(approx); current.append(previous)
                approximate_current.append(previous_approximate)
                previous_kappa, previous_approximate_kappa = base.kappa, approx.kappa
                run.save()
            previous_lambda = current
            approximate_previous_lambda = approximate_current
            indexes = np.array([np.argmin(abs(z-v)) for v in coarse])
            held = np.array([i for i in range(len(z)) if i not in indexes])
            for label, coefficients in [('baseline', baseline), ('shared_lower', substituted)]:
                for name, target in [('delta_beta', scale), ('kappa', scale/2)]:
                    values = np.array([getattr(c, name) for c in coefficients])
                    estimate = PchipInterpolator(coarse, values[indexes].real)(z[held])+1j*PchipInterpolator(coarse, values[indexes].imag)(z[held])
                    error = float(np.max(abs(estimate-values[held]))/target)
                    ok = np.isfinite(error) and error <= limits.midpoint_fraction
                    midpoints_ok &= ok
                    run.report['midpoint_checks'].append(dict(state=state, wavelength_m=wavelength, method=label,
                        quantity=name, normalized_error=error, passed=bool(ok)))
            if state == 'cross':
                for label, values in [('baseline', baseline), ('shared_lower', substituted)]:
                    objective = minimax_objective([values], z, config.device.length_m, scale)[0]
                    run.report.setdefault('cross_objectives', []).append(dict(wavelength_m=wavelength, method=label, value=objective))
            for launch in (0, 1):
                metrics, powers = [], []
                for coefficients in (baseline, substituted):
                    run.budget.check()
                    a, b, k = (np.array([getattr(c, name) for c in coefficients]) for name in ('beta_upper', 'beta_lower', 'kappa'))
                    result = propagate_converged(z, a, b, k, launch_layer=launch, settings=config.propagation)
                    metrics.append(port_metrics(result, state=state, launch_layer=launch, xt_power_floor=config.propagation.xt_power_floor))
                    powers.append(result.output_powers)
                a, b = metrics
                power_error = float(np.max(abs(powers[0]-powers[1])))
                il_error, xt_error = abs(a['il_db']-b['il_db']), abs(a['xt_db']-b['xt_db'])
                above_floor = min(a['undesired_power'], b['undesired_power'], a['desired_power'], b['desired_power']) > limits.absolute_port_power
                ok = (all(m['integration_converged'] for m in metrics) and np.isfinite(il_error)
                      and il_error <= limits.il_db and power_error <= limits.absolute_port_power
                      and (not above_floor or (np.isfinite(xt_error) and xt_error <= limits.xt_db)))
                propagation_ok &= ok
                run.report['propagation'].append(dict(state=state, wavelength_m=wavelength, launch_layer=launch,
                    baseline=a, shared_lower=b, il_error_db=il_error, xt_error_db=xt_error,
                    xt_db_gate_applied=bool(above_floor), maximum_port_power_error=power_error, passed=bool(ok)))
                run.save()
    if not healthy or not midpoints_ok:
        run.report.update(status='inconclusive', reason='Invalid baseline/pair selection or insufficient longitudinal coefficient resolution')
    elif not approximation_ok or not propagation_ok:
        run.report.update(status='failed', reason='Substitution exceeds coefficient or propagation error limits')
    else:
        run.report.update(status='passed', reason='All sampled approximation, numerical-resolution and propagation checks passed')
    objectives = run.report.get('cross_objectives', [])
    if objectives:
        values = {label: max(r['value'] for r in objectives if r['method'] == label)
                  for label in ('baseline', 'shared_lower')}
        run.report['cross_objective_comparison'] = dict(values, absolute_change=abs(values['baseline']-values['shared_lower']))


def _supervise(arguments, output, seconds):
    """Bound even a blocking lumapi call in a dedicated worker process tree."""
    run_id = uuid.uuid4().hex
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])+os.pathsep+env.get('PYTHONPATH', '')
    command = [sys.executable, '-u', '-m', 'vlac.lower_state_check', *arguments, '--worker-id', run_id]
    settings = {'env': env, 'stdout': subprocess.PIPE, 'stderr': subprocess.STDOUT}
    if os.name == 'nt':
        settings['creationflags'] = subprocess.CREATE_NO_WINDOW
    else:
        settings['start_new_session'] = True
    worker = subprocess.Popen(command, **settings)
    try:
        stdout, _ = worker.communicate(timeout=seconds)
        if stdout:
            print(stdout.decode('utf-8', errors='replace'), end='')
        return worker.returncode
    except subprocess.TimeoutExpired:
        # Only the dedicated diagnostic worker and its descendants are terminated.
        cleanup_error = None
        try:
            if os.name == 'nt':
                result = subprocess.run(['taskkill', '/PID', str(worker.pid), '/T', '/F'],
                                        check=False, capture_output=True, timeout=20)
                if result.returncode and worker.poll() is None:
                    cleanup_error = 'Could not confirm termination of all worker descendants'
            else:
                os.killpg(worker.pid, signal.SIGKILL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            cleanup_error = str(exc)
        finally:
            worker.kill()
            stdout, _ = worker.communicate(timeout=20)
            if stdout:
                print(stdout.decode('utf-8', errors='replace'), end='')
        report_path = Path(output)/'report.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if report.get('run_id') == run_id:
            report.update(status='incomplete', execution_completed=False,
                eligible_for_shared_lower_mode=False, reason='Hard wall-time cutoff stopped the isolated MODE worker',
                hard_timeout_seconds=seconds, cleanup_error=cleanup_error)
            write_json(report_path, report)
        write_json(Path(output)/'timeout.json', dict(run_id=run_id, status='incomplete',
            eligible_for_shared_lower_mode=False, hard_timeout_seconds=seconds, cleanup_error=cleanup_error))
        print(f'Lower-state check incomplete: hard cutoff at {seconds:g} seconds; completed sample packets retained')
        return 3


def main(argv=None):
    from .config import load_config
    parser = argparse.ArgumentParser(prog='vlac check-lower-state', description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--stage', choices=('screen', 'full'), default='screen')
    parser.add_argument('--profile-dir')
    parser.add_argument('--max-solves', type=int, default=120)
    parser.add_argument('--max-seconds', type=float, default=3300., help='Hard CLI worker cutoff, at most 3300 s; leaves teardown time within one hour')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--worker-id', help=argparse.SUPPRESS)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        config = load_config(args.config)
        if args.max_solves < 1 or not np.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 3300:
            raise ValueError('Positive solve budget and 0 < max-seconds <= 3300 are required')
        if args.plan_only:
            print(json.dumps(prepare(config, args.stage, args.profile_dir)[0], indent=2))
            return 0
        if args.worker_id is None:
            # Validate the saved handoff and scope before launching the worker.
            prepare(config, args.stage, args.profile_dir)
            return _supervise(arguments, args.out, args.max_seconds)
        report = run_check(config, args.out, stage=args.stage, profile_dir=args.profile_dir,
            max_solves=args.max_solves, max_seconds=args.max_seconds, resume=args.resume, run_id=args.worker_id)
        print(f'Lower-state check {report["status"]}: {Path(args.out).resolve()/"report.json"}')
        return 0 if report['status'] in ('passed', 'screen_passed') else 2 if report['status'] == 'failed' else 3
    except Exception as exc:
        print(f'Lower-state check failed: {type(exc).__name__}: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
