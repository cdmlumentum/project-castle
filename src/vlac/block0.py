"""Bounded MODE calibration, restartable evidence, and certified solver profiles.

The search is Python orchestration; every optical sample is calculated by the
native MODE core. Trial or synthetic runs can never issue production settings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
import hashlib
import html
from importlib.resources import files
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np
from scipy.io import savemat

from .block_io import clean, digest, write_json
from .geometry import solution_profiles, make_geometry
from .materials import build_material_bank

SCHEMA = 1
METHOD = 'mode_domain_mesh_calibration_v2_fixed_lattice'
KINDS = ('isolated', 'coupled')


def _ceil_cells(value):
    """Ceil a cell count without adding a cell for floating-point roundoff."""
    return max(2, math.ceil(value-1e-12*max(1., abs(value))))


from .runtime import BudgetExceeded


@lru_cache(maxsize=64)
def _hash_file(path, modified, size):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_hash(path):
    path = Path(path)
    info = path.stat()
    return _hash_file(str(path.resolve()), info.st_mtime_ns, info.st_size)


def implementation_identity():
    names = ('block0.py', 'config.py', 'geometry.py', 'materials.py', 'modes.py',
             'solver.py', 'native_backend.py', 'native/vlac_core.lsf', 'native/vlac_convergence.lsf')
    return digest({name: hashlib.sha256(files('vlac').joinpath(name).read_text(encoding='utf-8').replace('\r\n', '\n').encode()).hexdigest()
                   for name in names})


def installation_identity(config):
    api = Path(config.solver.api_path or '')
    if api.is_file():
        api = api.parent
    executable = api.parent.parent / 'bin' / 'mode-solutions.exe'
    def signature(path):
        return file_hash(path) if path.is_file() else None
    return {'api_sha256': signature(api / 'lumapi.py'), 'executable_sha256': signature(executable),
            'installation': api.parent.parent.name}


def input_identity(config):
    """Accuracy and physical envelope, excluding output paths and run budgets."""
    raw = config.as_dict()
    materials = dict(raw['materials'])
    for key in ('silicon_data_csv', 'sin_extinction_csv', 'liquid_dispersion_csv'):
        path = config.resolved_path(materials.pop(key))
        materials[key] = hashlib.sha256(path.read_bytes()).hexdigest() if path else None
    solver = {k: v for k, v in raw['solver'].items()
              if k not in {'calibration', 'engine', 'api_path', 'hide_gui', 'platform', 'threads',
                           'reuse_mode_project', 'batch_mode_data', 'raw_cache_memory_entries'}}
    limits = {k: v for k, v in raw['convergence'].items()
              if k not in {'max_solves', 'max_seconds', 'timing_repeats'}}
    return digest({'method': METHOD, 'geometry': raw['geometry'], 'materials': materials,
                   'wavelength_bounds_m': [config.device.wavelength_min_m, config.device.wavelength_max_m],
                   'length_m': config.device.length_m, 'solver': solver, 'accuracy': limits,
                   'endpoint_assumptions': {'state_independent_lower': config.endpoint_search.pso_lower_mode_state_independent,
                                            'method': config.endpoint_search.method,
                                            'guided_neff_margin': config.endpoint_search.pso_guided_neff_margin}})


@dataclass(frozen=True)
class SolverProfile:
    x_span_m: float
    y_min_m: float
    y_max_m: float
    dx_m: float
    dy_m: float
    pml_layers: int = 12
    local_dx_m: float = 0.
    local_dy_m: float = 0.
    local_padding_m: float = .4e-6
    tracking_x_span_m: float = 0.
    tracking_y_min_m: float = 0.
    tracking_y_max_m: float = 0.
    tracking_nx: int = 0
    tracking_ny: int = 0

    @property
    def nx(self):
        return _ceil_cells(self.x_span_m / self.dx_m)

    @property
    def ny(self):
        return _ceil_cells((self.y_max_m-self.y_min_m) / self.dy_m)

    def native(self):
        return {**asdict(self), 'mesh_cells_x': float(self.nx), 'mesh_cells_y': float(self.ny)}

    def grid(self):
        return (np.linspace(-self.tracking_x_span_m/2, self.tracking_x_span_m/2, self.tracking_nx),
                np.linspace(self.tracking_y_min_m, self.tracking_y_max_m, self.tracking_ny))


def base_profiles(config):
    g, s, c = config.geometry, config.solver, config.convergence
    half = max(abs(v) for v in g.d_c2c_bounds_m) + max(g.w1_max_m, g.w1_lower_m)/2
    full_span = 2*(half+g.domain_margin_x_m)
    ymin = -g.box_thickness_m-s.domain_margin_y_bottom_m
    ymax = 2*g.h_wg_m+g.h_gap_bounds_m[1]+g.domain_margin_top_m
    values = dict(y_min_m=ymin, y_max_m=ymax, dx_m=full_span/s.mesh_cells_x,
                  dy_m=(ymax-ymin)/s.mesh_cells_y, pml_layers=c.pml_layers,
                  local_padding_m=c.local_mesh_padding_m)
    return {'coupled': SolverProfile(x_span_m=full_span, **values),
            'isolated': SolverProfile(x_span_m=max(g.w1_max_m, g.w1_lower_m)+2*g.domain_margin_x_m, **values)}


def margins(config, profile, kind, factor=1.):
    g = config.geometry
    envelope = max(g.w1_max_m, g.w1_lower_m)
    if kind == 'coupled':
        envelope += 2*max(abs(v) for v in g.d_c2c_bounds_m)
    top = 2*g.h_wg_m+g.h_gap_bounds_m[1]
    return replace(profile, x_span_m=envelope+(profile.x_span_m-envelope)*factor,
                   y_min_m=-g.box_thickness_m-(-profile.y_min_m-g.box_thickness_m)*factor,
                   y_max_m=top+(profile.y_max_m-top)*factor)


def comparison_grid(config, profile, kind):
    largest = margins(config, profile, kind, max(config.convergence.reference_domain_factors)*1.25)
    c = config.convergence
    # Odd lengths permit a nested every-other-point aliasing check.
    nx = max(17, 2*math.ceil(largest.x_span_m/c.comparison_dx_m/2)+1)
    ny = max(17, 2*math.ceil((largest.y_max_m-largest.y_min_m)/c.comparison_dy_m/2)+1)
    return dict(tracking_x_span_m=largest.x_span_m, tracking_y_min_m=largest.y_min_m,
                tracking_y_max_m=largest.y_max_m, tracking_nx=nx, tracking_ny=ny)


def domain_reference_profile(config, anchor, kind, axis, factor):
    """Expand padding on the existing uniform lattice, keeping material sampling fixed."""
    target = margins(config, anchor, kind, factor)
    dx = anchor.x_span_m/anchor.nx
    dy = (anchor.y_max_m-anchor.y_min_m)/anchor.ny
    if axis == 'domain_x':
        cells = _ceil_cells(target.x_span_m/dx)
        # With a centered domain, equal parity preserves the x=0 registration.
        cells += (cells-anchor.nx) % 2
        return replace(anchor, x_span_m=cells*dx, dx_m=dx)
    if axis == 'domain_top':
        cells = max(0, math.ceil((target.y_max_m-anchor.y_max_m)/dy-1e-10))
        return replace(anchor, y_max_m=anchor.y_max_m+cells*dy, dy_m=dy)
    if axis == 'domain_bottom':
        cells = max(0, math.ceil((anchor.y_min_m-target.y_min_m)/dy-1e-10))
        return replace(anchor, y_min_m=anchor.y_min_m-cells*dy, dy_m=dy)
    raise ValueError('Unknown reference domain axis: '+axis)


def test_cases(config, *, trial=False):
    """Deterministic corners, interiors and two liquid states; no surrogate data."""
    g, d = config.geometry, config.device
    widths = [g.w1_min_m, (g.w1_min_m+g.w1_max_m)/2, g.w1_max_m]
    wavelengths = [d.wavelength_min_m, (d.wavelength_min_m+d.wavelength_max_m)/2, d.wavelength_max_m]
    offsets = sorted(set([*g.d_c2c_bounds_m, float(np.clip(0., *g.d_c2c_bounds_m))]))
    ribs = sorted(set(g.h_rib_bounds_m))
    gaps = sorted(set(g.h_gap_bounds_m))
    def case(width, offset, wavelength, rib, gap, state):
        return dict(w1_m=width, d_c2c_m=offset, wavelength_m=wavelength,
                    h_rib_m=rib, h_gap_m=gap, state=state)
    if trial:
        center = [case(widths[1], offsets[len(offsets)//2], wavelengths[1], ribs[0], gaps[0], state)
                  for state in ('cross', 'bar')]
        return {kind: center for kind in KINDS}
    result = {}
    for kind in KINDS:
        result[kind] = [case(*values) for values in itertools.product(
            widths, [float(np.clip(0., *g.d_c2c_bounds_m))] if kind == 'isolated' else offsets,
            wavelengths, ribs, gaps, ('cross', 'bar'))]
    return result


def sentinel_cases(cases, count):
    """Cover states and narrow/wide endpoints before the full envelope gate."""
    ordered = sorted(cases, key=lambda r: (r['w1_m'], -r['wavelength_m'], r['d_c2c_m'], r['state']))
    indexes = np.linspace(0, len(ordered)-1, min(count, len(ordered))).round().astype(int)
    return [ordered[i] for i in indexes]


def _power_overlap(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    denom = np.vdot(a, a).real*np.vdot(b, b).real
    return float(np.clip(abs(np.vdot(a, b))**2/denom, 0., 1.)) if denom > 0 else 0.


def _subspace_overlap(a, b):
    # Orthonormalize the selected pair to handle a rotation near degeneracy.
    a, b = a.reshape(2, -1).T, b.reshape(2, -1).T
    qa, ra = np.linalg.qr(a); qb, rb = np.linalg.qr(b)
    if min(abs(np.diag(ra)).min(), abs(np.diag(rb)).min()) < 1e-10:
        return 0.
    return float(np.clip(np.linalg.svd(qa.conj().T @ qb, compute_uv=False).min()**2, 0., 1.))


def compare_saved_coefficients(config, saved, reference):
    """Check the actual stored branch/coefficient values, not just its geometry."""
    c, length = config.convergence, config.device.length_m
    factor = 2*np.pi/reference['wavelength_m']
    names = ('beta_upper', 'beta_lower', 'beta_plus', 'beta_minus')[:len(reference['neff'])]
    errors = [complex(saved[key])-factor*reference['neff'][i] for i, key in enumerate(names) if key in saved]
    if not errors or not all(np.isfinite(value) for value in errors):
        return {'passed': False, 'scope': 'stored_scalar_coefficients', 'reason': 'Non-finite or absent stored propagation constants.'}
    neff_error = max(abs(v.real)/factor for v in errors)
    loss_error = max(abs(v.imag)*2*length*10/np.log(10) for v in errors)
    passed = neff_error <= c.neff_atol and loss_error <= c.loss_error_db and saved.get('flag', 'ok') == 'ok'
    metrics = {'maximum_real_neff_error': neff_error, 'maximum_loss_error_db': loss_error}
    for key in ('delta_beta', 'kappa'):
        if key in saved:
            a, b = complex(saved[key]), complex(reference[key])
            if key == 'kappa' and abs(-a-b) < abs(a-b):
                a = -a
            error = abs(a-b)
            passed &= bool(np.isfinite(error) and error <= c.coefficient_atol_rad_per_m+c.coefficient_rtol*abs(b)
                           and error*length <= c.phase_error_rad)
            metrics[key+'_error_rad_per_m'] = error
    return {'passed': bool(passed), 'scope': 'stored_scalar_coefficients', **metrics}


def compare_samples(config, actual, reference):
    """Compare complex outputs on identical, zero-padded physical field grids."""
    c, length = config.convergence, config.device.length_m
    an, rn = actual['neff'], reference['neff']
    af, rf = actual['fields'], reference['fields']
    if af.shape != rf.shape or an.shape != rn.shape:
        raise ValueError('Block 0 samples use incompatible mode/field grids.')
    finite = all(np.all(np.isfinite(v)) for v in (an, rn, af, rf, actual['te'], reference['te']))
    overlaps = [_power_overlap(a, b) for a, b in zip(af, rf)]
    coarse_overlaps = [_power_overlap(a[:, ::2, ::2], b[:, ::2, ::2]) for a, b in zip(af, rf)]
    subspace = 1.
    if len(an) == 4:
        subspace = _subspace_overlap(af[2:], rf[2:])
        # Beta ordering may exchange individual modes in a degenerate pair.
        factor = 2*np.pi/actual['wavelength_m']
        degenerate = abs(rn[2]-rn[3])*factor*length <= c.phase_error_rad
        field_min = min(overlaps[:2]+([subspace] if degenerate else overlaps[2:]))
    else:
        field_min = min(overlaps)
    neff_error = float(np.max(np.abs(an.real-rn.real)))
    loss_error = float(np.max(abs(an.imag-rn.imag))*2*(2*np.pi/actual['wavelength_m'])*length*10/np.log(10))
    polarization_error = float(np.max(abs(actual['te']-reference['te'])))
    aliasing_error = float(max(abs(a-b) for a, b in zip(overlaps, coarse_overlaps)))
    errors, coefficients_pass = {}, True
    for key in ('delta_beta', 'kappa'):
        if key not in actual:
            continue
        a, b = complex(actual[key]), complex(reference[key])
        if key == 'kappa' and abs(-a-b) < abs(a-b):
            a = -a  # Compare the same square-root gauge.
        error = abs(a-b)
        allowed = c.coefficient_atol_rad_per_m+c.coefficient_rtol*abs(b)
        phase = error*length
        errors[key+'_error_rad_per_m'] = float(error)
        errors[key+'_phase_error_rad'] = float(phase)
        coefficients_pass &= error <= allowed and phase <= c.phase_error_rad
    flags_ok = actual.get('flag', 'ok') == reference.get('flag', 'ok') == 'ok'
    polarization_valid = min(actual['te'].min(), reference['te'].min()) >= config.solver.minimum_te_fraction
    passed = (finite and flags_ok and polarization_valid and neff_error <= c.neff_atol and loss_error <= c.loss_error_db
              and polarization_error <= c.polarization_atol and field_min >= c.minimum_power_overlap
              and subspace >= c.minimum_power_overlap and coefficients_pass
              and aliasing_error <= (1-c.minimum_power_overlap)/5)
    return {'passed': bool(passed), 'finite': bool(finite), 'flags_ok': flags_ok,
            'polarization_valid': bool(polarization_valid),
            'minimum_te_fraction': float(min(actual['te'].min(), reference['te'].min())),
            'maximum_real_neff_error': neff_error, 'maximum_loss_error_db': loss_error,
            'minimum_transverse_e_power_overlap': field_min, 'pair_subspace_overlap': subspace,
            'maximum_te_fraction_error': polarization_error, 'overlap_grid_aliasing_error': aliasing_error, **errors}


class NativeEvaluator:
    """One session; selectors, eigensolves and field interpolation run in MODE."""
    synthetic = False

    def __init__(self, config, backend=None):
        self.config = config
        self.backend = backend
        self.owns_backend = backend is None
        self.functions_loaded = False
        self.bank = build_material_bank(config)

    def _open(self):
        from .native_backend import NativeLumapiBackend, load_native_functions
        if self.backend is None:
            self.backend = NativeLumapiBackend(self.config)
        try:
            load_native_functions(self.backend.session, 'vlac_core.lsf')
            load_native_functions(self.backend.session, 'vlac_convergence.lsf')
            self.functions_loaded = True
        except Exception:
            self.close()
            raise

    def close(self):
        if self.backend is not None and self.owns_backend:
            self.backend.close()
        elif self.backend is not None:
            # Native geometry operations invalidate a Python adapter's layout.
            self.backend._uploaded_config = None
            self.backend._project_ready = False
            self.backend._installed_material_names = set()

    def __call__(self, kind, case, profile):
        from .native_backend import scalar, column
        if not self.functions_loaded:
            self._open()
        g = self.config.geometry
        liquid = self.bank.liquid_cross if case['state'] == 'cross' else self.bank.liquid_bar
        geometry = make_geometry(g, **{k: case[k] for k in ('w1_m', 'd_c2c_m', 'h_rib_m', 'h_gap_m', 'state')},
                                 liquid_index=float(liquid.index(case['wavelength_m'])))
        from .native_mode import config_payload
        cfg = config_payload(self.config)
        cfg['full_profile'] = cfg['compact_profile'] = profile.native()
        x, y = profile.grid()
        cfg['comparison_enabled'] = 1.
        cfg['comparison_x_m'], cfg['comparison_y_m'] = column(x), column(y)
        session = self.backend.session
        signature = digest(cfg)
        if getattr(self, '_scene_signature', None) != signature:
            session.eval('newproject;')
            self._scene_signature = signature
        reserved = (2 if kind=='isolated' else 3)*(1+self.config.solver.zero_mode_retries)
        self.backend._budget().check(reserved)
        session.putv('vlac_cfg', cfg)
        session.putv('vlac_request', {'geometry': asdict(geometry), 'wavelength_m': case['wavelength_m'], 'kind': kind})
        started = time.perf_counter()
        try:
            session.eval('vlac_sample_error=""; try { vlac_sample=vlac_convergence_sample(vlac_cfg,vlac_request); } catch(vlac_sample_error);')
            error = session.getv('vlac_sample_error')
            if error:
                raise RuntimeError('Block 0 native script error: '+str(error))
            value = session.getv('vlac_sample')
        except BaseException:
            self.backend._budget().consume(reserved)
            raise
        elapsed = time.perf_counter()-started
        self.backend._budget().consume(int(scalar(value.get('solves', 0))))
        if not bool(scalar(value['ok'])):
            raise RuntimeError('Block 0 MODE sample failed: '+value.get('error', 'unknown'))
        modes = list(value['modes'])
        result = {'neff': np.array([complex(scalar(m['neff'])) for m in modes]),
                  'fields': np.array([[np.asarray(m['ex']), np.asarray(m['ey'])] for m in modes], dtype=complex),
                  'te': np.array([float(scalar(m['te_fraction'])) for m in modes]),
                  'wavelength_m': case['wavelength_m'], 'elapsed_seconds': elapsed,
                  'solves': int(scalar(value['solves'])), 'flag': value['flag'],
                  'actual_mesh': [clean(m['mesh']) for m in modes]}
        if kind == 'coupled':
            result.update(delta_beta=complex(scalar(value['delta_beta'])), kappa=complex(scalar(value['kappa'])))
        else:
            result['delta_beta'] = (result['neff'][0]-result['neff'][1])*2*np.pi/case['wavelength_m']
        return result


class CalibrationRun:
    def __init__(self, config, output, evaluator, *, trial=False, resume=False):
        self.config, self.out, self.evaluator, self.trial = config, Path(output).resolve(), evaluator, trial
        self.started = time.monotonic()
        self.solves, self.executed = 0, 0
        self.rows, self.references = [], {}
        self.out.mkdir(parents=True, exist_ok=True)
        self.samples = self.out/'samples'
        self.samples.mkdir(exist_ok=True)
        self.identity = {'schema': SCHEMA, 'method': METHOD, 'input_id': input_identity(config),
                         'implementation_id': implementation_identity(), 'installation_id': installation_identity(config),
                         'trial': trial, 'synthetic': bool(evaluator.synthetic),
                         'timing_environment': {'host': platform.node(), 'threads': config.solver.threads}}
        checkpoint = self.out/'checkpoint.json'
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding='utf-8'))
            if not resume:
                raise FileExistsError('Block 0 output exists; use --resume or choose a new directory.')
            if saved['identity'] != self.identity:
                raise ValueError('Block 0 resume inputs, implementation or MODE installation changed.')
        elif resume:
            raise FileNotFoundError('No Block 0 checkpoint exists to resume.')
        self.checkpoint()
        write_json(self.out/'plan.json', {'identity': self.identity,
            'initial_case_counts': {k: len(v) for k, v in test_cases(config, trial=trial).items()},
            'budgets': {'max_solves': config.convergence.max_solves, 'max_seconds': config.convergence.max_seconds},
            'budget_policy': 'Per invocation; reserve all configured retries before starting an indivisible sample.',
            'wall_time_policy': 'Soft limit checked between samples; an active eigensolve can exceed it.',
            'method': 'Reference domain and mesh sequences, PML guard, coordinate screening, combined candidates, full envelope and adaptive gates.',
            'field_metric': 'Normalized transverse electric-field overlap squared; not a full E/H power-flux integral.'})

    def checkpoint(self, **extra):
        write_json(self.out/'checkpoint.json', {'identity': self.identity, 'session_solves': self.solves,
                   'session_samples': self.executed, 'elapsed_seconds': time.monotonic()-self.started, **extra})

    def sample(self, kind, case, profile, *, repeat=0):
        key = digest({'identity': self.identity, 'kind': kind, 'case': case, 'profile': asdict(profile), 'repeat': repeat})
        metadata_path, fields_path = self.samples/(key+'.json'), self.samples/(key+'.npz')
        if metadata_path.exists() and fields_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if metadata['npz_sha256'] != hashlib.sha256(fields_path.read_bytes()).hexdigest():
                raise ValueError('Block 0 checkpoint sample checksum mismatch.')
            with np.load(fields_path, allow_pickle=False) as arrays:
                value = {name: arrays[name].copy() for name in ('neff', 'fields', 'te')}
            value.update(metadata['values'])
            for name in ('kappa', 'delta_beta'):
                if name in value:
                    value[name] = complex(**value[name])
            return value
        reserved = (2 if kind == 'isolated' else 3)*(1+self.config.solver.zero_mode_retries)
        c = self.config.convergence
        if self.solves+reserved > c.max_solves or time.monotonic()-self.started >= c.max_seconds:
            raise BudgetExceeded('Block 0 solve/time budget reached between samples; resume to continue.')
        print(f'Block 0 {kind}, {case["state"]}, width={case["w1_m"]*1e6:.3g} um, '
              f'offset={case["d_c2c_m"]*1e6:.3g} um, lambda={case["wavelength_m"]*1e9:.0f} nm; '
              f'{profile.nx} x {profile.ny} base cells', flush=True)
        try:
            value = self.evaluator(kind, case, profile)
        except Exception:
            # Reserve all possible attempts after an interrupted/failed native call.
            self.solves += reserved
            self.checkpoint(last_sample_failed=True)
            raise
        self.solves += value['solves']
        self.executed += 1
        if not all(np.all(np.isfinite(value[name])) for name in ('neff', 'fields', 'te')):
            raise ValueError('Block 0 sample contains non-finite modes or fields.')
        np.savez_compressed(fields_path, **{name: value[name] for name in ('neff', 'fields', 'te')})
        metadata = {'kind': kind, 'case': case, 'profile': asdict(profile),
                    'values': {k: v for k, v in value.items() if k not in {'neff', 'fields', 'te'}},
                    'npz_sha256': hashlib.sha256(fields_path.read_bytes()).hexdigest()}
        write_json(metadata_path, metadata)
        self.checkpoint(last_sample=key)
        if not (self.out/'pilot.json').exists():
            counts = {k: len(v) for k, v in test_cases(self.config, trial=self.trial).items()}
            write_json(self.out/'pilot.json', {'first_sample_kind': kind, 'seconds': value['elapsed_seconds'],
                'findmodes_calls': value['solves'], 'seconds_per_findmodes_including_setup_and_fields': value['elapsed_seconds']/value['solves'],
                'initial_case_counts': counts,
                'note': 'Timing includes native geometry, eigensolves, selectors and field transfer. References/refinements add work beyond one envelope pass.'})
        return value

    def compare(self, kind, cases, profile, reference_profile, stage):
        passed, times = True, []
        for case in cases:
            reference = self.sample(kind, case, reference_profile)
            actual = self.sample(kind, case, profile)
            metrics = compare_samples(self.config, actual, reference)
            self.rows.append({'stage': stage, 'kind': kind, 'case_id': digest(case), **case,
                              'profile_id': digest(asdict(profile)), 'elapsed_seconds': actual['elapsed_seconds'],
                              'actual_mesh': actual.get('actual_mesh', []), **metrics})
            passed &= metrics['passed']
            times.append(actual['elapsed_seconds'])
        self.checkpoint(stage=stage)
        return bool(passed), float(np.mean(times))

    def reference(self, kind, cases, base):
        c = self.config.convergence
        current = base
        # Domain tests preserve requested dx/dy; mesh tests preserve the domain.
        for axis, factors in [(name, c.reference_domain_factors) for name in ('domain_x', 'domain_top', 'domain_bottom')]+[
                (name, c.reference_mesh_factors) for name in ('mesh_x', 'mesh_y')]:
            previous, stable = None, 0
            anchor = current
            for factor in factors:
                enlarged = margins(self.config, anchor, kind, factor)
                changes = {'domain_x': {'x_span_m': enlarged.x_span_m},
                           'domain_top': {'y_max_m': enlarged.y_max_m},
                           'domain_bottom': {'y_min_m': enlarged.y_min_m},
                           'mesh_x': {'dx_m': anchor.dx_m*factor},
                           'mesh_y': {'dy_m': anchor.dy_m*factor}}
                profile = (domain_reference_profile(self.config, anchor, kind, axis, factor)
                           if axis.startswith('domain_') else replace(anchor, **changes[axis]))
                if previous is not None:
                    passed, _ = self.compare(kind, cases, previous, profile, 'reference_'+axis)
                    stable = stable+1 if passed else 0
                else:
                    for case in cases:
                        self.sample(kind, case, profile)
                previous = current = profile
                if stable >= 2:
                    break
            if stable < 2:
                return None, 'Reference '+axis+' did not stabilize over two consecutive refinements.'
        # Verify the interaction at the final mesh and a further domain expansion.
        joint = replace(margins(self.config, current, kind, 1.15), dx_m=current.dx_m*.8, dy_m=current.dy_m*.8)
        passed, _ = self.compare(kind, cases, current, joint, 'reference_joint')
        if not passed:
            return None, 'Combined reference domain/mesh check failed.'
        pml = replace(joint, pml_layers=joint.pml_layers+c.pml_test_extra_layers)
        passed, _ = self.compare(kind, cases, pml, joint, 'reference_pml')
        if not passed:
            return None, 'Reference PML sensitivity exceeds the error budget.'
        return joint, ''

    def candidates(self, kind, base):
        """Coordinate screening plus combined candidates; not a Cartesian sweep."""
        c, g = self.config.convergence, self.config.geometry
        candidates = [base]
        envelope = max(g.w1_max_m, g.w1_lower_m)+(2*max(abs(v) for v in g.d_c2c_bounds_m) if kind == 'coupled' else 0)
        top = 2*g.h_wg_m+g.h_gap_bounds_m[1]
        for factor in c.candidate_margin_factors:
            candidates.extend([replace(base, x_span_m=envelope+(base.x_span_m-envelope)*factor),
                               replace(base, y_max_m=top+(base.y_max_m-top)*factor),
                               replace(base, y_min_m=-g.box_thickness_m-(-base.y_min_m-g.box_thickness_m)*factor),
                               margins(self.config, base, kind, factor)])
        for factor in c.candidate_mesh_factors:
            candidates.extend([replace(base, dx_m=base.dx_m*factor), replace(base, dy_m=base.dy_m*factor),
                               replace(base, dx_m=base.dx_m*factor, dy_m=base.dy_m*factor)])
        # Local overrides cover both slabs vertically and each active ridge laterally.
        candidates.append(replace(base, dx_m=base.dx_m*2, dy_m=base.dy_m*2,
                                  local_dx_m=base.dx_m, local_dy_m=base.dy_m))
        unique = {digest(asdict(profile)): profile for profile in candidates}
        return list(unique.values())

    def adaptive_cases(self, kind, cases, reference):
        """Add near-phase-matched widths and poorly overlapping branch midpoints."""
        groups, result = {}, {digest(case): case for case in cases}
        for case in cases:
            key = digest({k: v for k, v in case.items() if k != 'w1_m'})
            groups.setdefault(key, []).append(case)
        for group in groups.values():
            group.sort(key=lambda case: case['w1_m'])
            for left, right in zip(group, group[1:]):
                a, b = self.sample(kind, left, reference), self.sample(kind, right, reference)
                da, db = a['delta_beta'].real, b['delta_beta'].real
                widths = []
                if da*db <= 0 and da != db:
                    widths += [(left['w1_m']+right['w1_m'])/2,
                               left['w1_m']+(right['w1_m']-left['w1_m'])*abs(da)/(abs(da)+abs(db))]
                if (_power_overlap(a['fields'][0], b['fields'][0]) < .95
                        or min(a['te'].min(), b['te'].min()) < self.config.solver.minimum_te_fraction+.05):
                    widths.append((left['w1_m']+right['w1_m'])/2)
                for width in widths:
                    case = dict(left, w1_m=float(width))
                    result[digest(case)] = case
        return list(result.values())

    def certify_kind(self, kind, cases, base):
        sentinels = sentinel_cases(cases, self.config.convergence.reference_sentinels)
        reference, error = self.reference(kind, sentinels, base)
        if reference is None:
            return None, {'status': 'failed', 'reason': error}
        self.references[kind] = asdict(reference)
        cases = self.adaptive_cases(kind, cases, reference)
        # Persist the actual expanded sample set, including phase-match probes.
        write_json(self.out/(kind+'_certification_cases.json'), cases)
        candidates = self.candidates(kind, base)+[reference]
        passing = []
        for profile in candidates:
            passed, elapsed = self.compare(kind, sentinels, profile, reference, 'candidate_screen')
            if passed:
                passing.append((elapsed, profile))
        # Combine independently successful reductions and retest interactions.
        if passing:
            values = [p for _, p in passing]
            combined = replace(base, x_span_m=min(p.x_span_m for p in values),
                               y_min_m=max(p.y_min_m for p in values), y_max_m=min(p.y_max_m for p in values),
                               dx_m=max(p.dx_m for p in values), dy_m=max(p.dy_m for p in values))
            passed, elapsed = self.compare(kind, sentinels, combined, reference, 'candidate_combined')
            if passed:
                passing.append((elapsed, combined))
        failures = []
        for _, profile in sorted(passing, key=lambda p: p[0]):
            # A candidate that passed only the sentinels cannot be certified.
            passed, _ = self.compare(kind, cases, profile, reference, 'full_envelope_gate')
            if not passed:
                failures.append(digest(asdict(profile)))
                continue
            # Reference convergence is verified on EVERY certification case,
            # including cases absent from the initial screening set.
            guard = replace(margins(self.config, reference, kind, 1.05),
                            dx_m=reference.dx_m*.9, dy_m=reference.dy_m*.9,
                            pml_layers=reference.pml_layers+self.config.convergence.pml_test_extra_layers)
            passed, _ = self.compare(kind, cases, reference, guard, 'full_reference_gate')
            if not passed:
                return None, {'status': 'failed', 'reason': 'An added case invalidated the converged reference; refine the reference plan.',
                              'failed_candidate_ids': failures}
            timings, baseline_timings = [], []
            for repeat in range(1, self.config.convergence.timing_repeats+1):
                for case in sentinels:
                    # No MODE-result cache: each timed replicate runs fresh FDE.
                    timings.append(self.sample(kind, case, profile, repeat=repeat)['elapsed_seconds'])
                    baseline_timings.append(self.sample(kind, case, base, repeat=repeat)['elapsed_seconds'])
            return profile, {'status': 'passed', 'cases': len(cases), 'reference': asdict(reference),
                             'mean_seconds': float(np.mean(timings)), 'baseline_mean_seconds': float(np.mean(baseline_timings)),
                             'measured_speedup': float(np.mean(baseline_timings)/np.mean(timings)),
                             'failed_candidate_ids': failures}
        return None, {'status': 'failed', 'reason': 'No candidate passed the complete envelope.'}

    def finish(self, certificate, assessments):
        certificate = {**certificate, **self.identity, 'created_utc': datetime.now(timezone.utc).isoformat(),
                       'session_solves': self.solves, 'session_samples': self.executed,
                       'elapsed_seconds': time.monotonic()-self.started, 'assessments': assessments,
                       'hardware': {'host': platform.node(), 'platform': platform.platform(), 'python': sys.version,
                                    'solver_threads': self.config.solver.threads},
                       'accuracy_limits': asdict(self.config.convergence),
                       'scope_note': 'Numerical MODE calibration within sampled bounds; not physical device validation.'}
        evidence = sorted(self.samples.glob('*.json'))
        certificate['evidence'] = {'comparison_results_digest': digest(self.rows), 'sample_metadata_count': len(evidence),
                                   'sample_index_digest': digest({p.name: file_hash(p) for p in evidence})}
        certificate['certificate_id'] = digest(certificate)
        # Results are evidence even if incomplete; only passed, real, full runs
        # may publish solver_profiles.{json,mat} and calibrated_configuration.json.
        write_json(self.out/'certificate.json', certificate)
        write_json(self.out/'test_cases.json', test_cases(self.config, trial=self.trial))
        write_json(self.out/'convergence_results.json', self.rows)
        columns = ['stage', 'kind', 'state', 'w1_m', 'd_c2c_m', 'wavelength_m', 'profile_id', 'passed',
                   'elapsed_seconds', 'maximum_real_neff_error', 'maximum_loss_error_db',
                   'minimum_transverse_e_power_overlap', 'pair_subspace_overlap', 'maximum_te_fraction_error',
                   'overlap_grid_aliasing_error', 'delta_beta_error_rad_per_m', 'kappa_error_rad_per_m']
        with (self.out/'convergence_results.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
            writer.writeheader(); writer.writerows(self.rows)
        rows = ''.join('<tr>'+''.join('<td>'+html.escape(str(row.get(k, '')))+'</td>' for k in columns)+'</tr>' for row in self.rows)
        page = ('<!doctype html><meta charset="utf-8"><title>Block 0 MODE calibration</title>'
                '<style>body{font:15px system-ui;margin:2rem}table{border-collapse:collapse;font-size:12px}'
                'td,th{border:1px solid #ccc;padding:.4rem}th{position:sticky;top:0;background:#eee}</style>'
                '<h1>Block 0 MODE calibration</h1><p>Status: <b>'+html.escape(certificate['status'])+'</b></p>'
                '<p>'+html.escape(certificate.get('reason', ''))+'</p><p>Trial: '+str(self.trial)+
                '; synthetic: '+str(self.evaluator.synthetic)+'</p><pre>'+html.escape(json.dumps(clean(assessments), indent=2))+
                '</pre><table><thead><tr>'+''.join('<th>'+k+'</th>' for k in columns)+'</tr></thead><tbody>'+rows+'</tbody></table>')
        (self.out/'report.html').write_text(page, encoding='utf-8')
        if certificate['status'] == 'passed':
            config = apply_certificate(self.config, certificate)
            write_json(self.out/'solver_profiles.json', {'certificate_id': certificate['certificate_id'], 'profiles': certificate['profiles']})
            savemat(self.out/'solver_profiles.mat', {'block0_profiles': certificate['profiles'],
                    'block0_certificate_id': certificate['certificate_id']}, long_field_names=True)
            raw = config.as_dict()
            raw['project_root'] = str(config.project_root)
            for key in ('silicon_data_csv', 'sin_extinction_csv', 'liquid_dispersion_csv'):
                path = config.resolved_path(getattr(config.materials, key))
                raw['materials'][key] = str(path) if path else None
            write_json(self.out/'calibrated_configuration.json', raw)
        else:
            for name in ('solver_profiles.json', 'solver_profiles.mat', 'calibrated_configuration.json'):
                (self.out/name).unlink(missing_ok=True)
        self.checkpoint(status=certificate['status'])
        return certificate

    def run(self):
        cases = test_cases(self.config, trial=self.trial)
        bases = base_profiles(self.config)
        bases = {kind: replace(p, **comparison_grid(self.config, p, kind)) for kind, p in bases.items()}
        selected, assessments = {}, {}
        try:
            if self.trial:
                for kind in KINDS:
                    base = bases[kind]
                    for case in cases[kind]:
                        # Three distinct settings; no convergence claim from a trial.
                        fine = replace(base, dx_m=base.dx_m*.75, dy_m=base.dy_m*.75)
                        expanded = margins(self.config, base, kind, 1.25)
                        self.compare(kind, [case], base, fine, 'trial_mesh')
                        self.compare(kind, [case], base, expanded, 'trial_domain')
                return self.finish({'status': 'incomplete', 'execution_completed': True,
                                    'reason': 'Diagnostic trial completed; full convergence certification was not requested.', 'profiles': {}}, assessments)
            for kind in KINDS:
                profile, assessment = self.certify_kind(kind, cases[kind], bases[kind])
                assessments[kind] = assessment
                if profile is None:
                    return self.finish({'status': 'failed', 'execution_completed': True,
                                        'reason': assessment['reason'], 'profiles': {}}, assessments)
                selected[kind] = asdict(profile)
            passed = not self.evaluator.synthetic
            return self.finish({'status': 'passed' if passed else 'incomplete', 'execution_completed': True,
                                'reason': 'Full sampled envelope passed.' if passed else 'Synthetic tests cannot certify production settings.',
                                'profiles': selected if passed else {}, 'reference_profiles': self.references}, assessments)
        except BudgetExceeded as exc:
            return self.finish({'status': 'incomplete', 'execution_completed': False, 'reason': str(exc), 'profiles': {}}, assessments)
        except Exception as exc:
            self.finish({'status': 'failed', 'execution_completed': False,
                         'reason': f'{type(exc).__name__}: {exc}', 'profiles': {}}, assessments)
            raise


def run_calibration(config, output, *, trial=False, resume=False, evaluator_factory=NativeEvaluator):
    # Calibrating a calibrated config would make baseline comparisons ambiguous.
    if config.solver.calibration is not None:
        raise ValueError('Use the original uncalibrated configuration for Block 0.')
    config.validate()
    evaluator = evaluator_factory(config)
    try:
        return CalibrationRun(config, output, evaluator, trial=trial, resume=resume).run()
    finally:
        evaluator.close()


def validate_certificate(config, certificate):
    if (certificate.get('schema') != SCHEMA or certificate.get('method') != METHOD
            or certificate.get('status') != 'passed' or certificate.get('trial') is not False
            or certificate.get('synthetic') is not False or not certificate.get('execution_completed')):
        raise ValueError('Block 0 requires a passed, full, real MODE certificate.')
    if certificate.get('certificate_id') != digest({k: v for k, v in certificate.items() if k != 'certificate_id'}):
        raise ValueError('Block 0 certificate checksum mismatch.')
    if certificate.get('input_id') != input_identity(config):
        raise ValueError('Block 0 certificate does not match geometry, materials, solver settings or accuracy limits.')
    if certificate.get('implementation_id') != implementation_identity():
        raise ValueError('Block 0 certificate predates this solver implementation; recalibrate.')
    if certificate.get('installation_id') != installation_identity(config):
        raise ValueError('Block 0 certificate does not match the MODE installation.')
    if set(certificate.get('profiles', {})) != set(KINDS):
        raise ValueError('Block 0 certificate must contain isolated and coupled profiles.')
    if set(certificate.get('reference_profiles', {})) != set(KINDS):
        raise ValueError('Block 0 certificate must retain both reference profiles.')
    if any(certificate.get('assessments', {}).get(kind, {}).get('status') != 'passed' for kind in KINDS):
        raise ValueError('Block 0 certificate is missing passing envelope assessments.')
    for kind, raw in [*certificate['profiles'].items(), *certificate['reference_profiles'].items()]:
        p = SolverProfile(**raw)
        numbers = list(asdict(p).values())
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in numbers):
            raise ValueError('Non-finite Block 0 profile.')
        if (p.dx_m <= 0 or p.dy_m <= 0 or p.x_span_m <= 0 or p.y_min_m >= -config.geometry.box_thickness_m
                or p.y_max_m <= 2*config.geometry.h_wg_m+config.geometry.h_gap_bounds_m[1]
                or p.pml_layers < 1 or int(p.pml_layers) != p.pml_layers or p.tracking_nx < 16 or p.tracking_ny < 16
                or int(p.tracking_nx) != p.tracking_nx or int(p.tracking_ny) != p.tracking_ny
                or p.tracking_x_span_m < p.x_span_m or p.tracking_y_min_m > p.y_min_m or p.tracking_y_max_m < p.y_max_m
                or p.local_dx_m < 0 or p.local_dy_m < 0 or p.local_padding_m <= 0):
            raise ValueError('Invalid Block 0 domain, mesh or tracking grid.')
        envelope = max(config.geometry.w1_max_m, config.geometry.w1_lower_m)
        if kind == 'coupled':
            envelope += 2*max(abs(v) for v in config.geometry.d_c2c_bounds_m)
        if p.x_span_m <= envelope:
            raise ValueError('Block 0 lateral profile does not enclose the required guide envelope.')


def apply_certificate(config, source):
    if not isinstance(source, dict):
        path = Path(source)
        source = json.loads((path/'certificate.json' if path.is_dir() else path).read_text(encoding='utf-8'))
    validate_certificate(config, source)
    return replace(config, solver=replace(config.solver, calibration=source))


def certified_profile(config, kind):
    calibration = config.solver.calibration
    return SolverProfile(**calibration['profiles'][kind]) if calibration else None


def mesh_override_script(profile, geometry, include_upper, include_lower):
    """The original Python adapter uses the same override geometry as LSF."""
    if not (profile.local_dx_m or profile.local_dy_m):
        return []
    lines = []
    for index, enabled in enumerate((include_lower, include_upper)):
        if not enabled:
            continue
        x = geometry.d_c2c_m if index else 0.
        y = geometry.upper_bottom_y_m+geometry.h_wg_m/2 if index else geometry.h_wg_m/2
        width = geometry.w1_m if index else geometry.w2_m
        values = [('x', x), ('x span', width+2*profile.local_padding_m), ('y', y),
                  ('y span', geometry.h_wg_m+2*profile.local_padding_m), ('z', 0), ('z span', 1e-6),
                  ('override x mesh', int(profile.local_dx_m>0)), ('override y mesh', int(profile.local_dy_m>0)),
                  ('override z mesh', 0), ('set maximum mesh step', 1)]
        if profile.local_dx_m:
            values.append(('dx', profile.local_dx_m))
        if profile.local_dy_m:
            values.append(('dy', profile.local_dy_m))
        lines.append(f'addmesh; set("name","vlac_local_{index+1}");')
        lines.extend(f'set("{key}",{value:.16g});' for key, value in values)
    return lines+['select("FDE");']


def verify_saved_design(config, output, *, endpoint_dir=None, profile_dir=None, evaluator_factory=NativeEvaluator):
    """Fresh reference comparisons on actual selected endpoints/profile geometry."""
    from .block_io import load_block
    certificate = config.solver.calibration
    if certificate is None:
        raise ValueError('A certified Block 0 configuration is required for design verification.')
    if bool(endpoint_dir) == bool(profile_dir):
        raise ValueError('Choose exactly one endpoint_dir or profile_dir.')
    endpoint, solution, _, _ = load_block(profile_dir or endpoint_dir, 2 if profile_dir else 1, config)
    return verify_design(config, output, endpoint, solution, evaluator_factory=evaluator_factory)


def verify_design(config, output, endpoint, solution=None, *, bar=None, evaluator_factory=NativeEvaluator, resume=False,
                  endpoint_wavelengths_only=False):
    """Bounded fresh validation; this never creates another calibration certificate."""
    certificate = config.solver.calibration
    if certificate is None:
        raise ValueError('A certified Block 0 configuration is required for design verification.')
    config.validate()
    if solution is None:
        widths = [endpoint.w1_input_m, endpoint.w1_output_m]
        offsets = [config.geometry.d_c2c_input_m, config.geometry.d_c2c_output_m]
        wavelengths = [config.device.wavelength_min_m, config.device.wavelength_max_m]
        points = list(zip(widths, offsets))
    else:
        z = np.unique(np.r_[solution.z_m, (solution.z_m[:-1]+solution.z_m[1:])/2])
        wp, dp = solution_profiles(config, solution)
        points = list(zip(wp(z), dp(z)))
        wavelengths = ([config.device.wavelength_min_m,config.device.wavelength_max_m] if endpoint_wavelengths_only else
                       np.unique(np.r_[solution.wavelengths_m, (config.device.wavelength_min_m+config.device.wavelength_max_m)/2]))
    evaluator = evaluator_factory(config)
    run = None
    try:
        run = CalibrationRun(config, output, evaluator, trial=False, resume=resume)
        passed = True
        for kind in (('isolated',) if solution is None else KINDS):
            cases = [dict(w1_m=float(width), d_c2c_m=float(offset), wavelength_m=float(wavelength),
                          h_rib_m=endpoint.h_rib_m, h_gap_m=endpoint.h_gap_m, state=state)
                     for (width, offset), wavelength, state in itertools.product(points, wavelengths, ('cross', 'bar'))]
            ok, _ = run.compare(kind, cases, certified_profile(config, kind),
                                SolverProfile(**certificate['reference_profiles'][kind]), 'selected_design_gate')
            passed &= ok
        # Check saved results to catch branch selection errors in the actual run.
        scalar_checks = []
        if solution is None:
            for row in endpoint.endpoint_records:
                case = dict(w1_m=endpoint.w1_input_m if row['end']=='input' else endpoint.w1_output_m,
                            d_c2c_m=config.geometry.d_c2c_input_m if row['end']=='input' else config.geometry.d_c2c_output_m,
                            wavelength_m=row['wavelength_m'], h_rib_m=endpoint.h_rib_m, h_gap_m=endpoint.h_gap_m, state=row['state'])
                reference = run.sample('isolated', case, SolverProfile(**certificate['reference_profiles']['isolated']))
                saved = {name: complex(row[name+'_real_rad_per_m'], row[name+'_imag_rad_per_m'])
                         for name in ('beta_upper', 'beta_lower')}
                saved['delta_beta'] = saved['beta_upper']-saved['beta_lower']
                scalar_checks.append(dict(case, **compare_saved_coefficients(config, saved, reference)))
        else:
            for state, grid in [('cross', solution.cross_coefficients)]+([('bar', bar)] if bar is not None else []):
                for iw, wavelength in enumerate(solution.wavelengths_m):
                    for iz, z in enumerate(solution.z_m):
                        case = dict(w1_m=float(wp(z)), d_c2c_m=float(dp(z)), wavelength_m=float(wavelength),
                                    h_rib_m=endpoint.h_rib_m, h_gap_m=endpoint.h_gap_m, state=state)
                        reference = run.sample('coupled', case, SolverProfile(**certificate['reference_profiles']['coupled']))
                        saved = {key: getattr(grid[iw][iz], key) for key in ('beta_upper', 'beta_lower', 'beta_plus', 'beta_minus', 'delta_beta', 'kappa')}
                        saved['flag'] = grid[iw][iz].extraction_flag
                        scalar_checks.append(dict(case, **compare_saved_coefficients(config, saved, reference)))
        passed &= all(row['passed'] for row in scalar_checks)
        report = {'status': ('incomplete' if evaluator.synthetic else 'passed') if passed else 'failed',
                  'synthetic': bool(evaluator.synthetic), 'certificate_id': certificate['certificate_id'],
                  'scope': 'Actual saved design checked in both liquid states against the Block 0 reference.',
                  'wavelength_policy':'endpoints_only' if endpoint_wavelengths_only else 'saved_grid_plus_midpoint',
                  'wavelengths_m':[float(v) for v in wavelengths],
                  'checks': run.rows, 'stored_coefficient_checks': scalar_checks}
    except BudgetExceeded as exc:
        report = {'status': 'incomplete', 'reason': str(exc), 'checks': run.rows if run else []}
    except Exception as exc:
        write_json(Path(output)/'design_verification.json', {'status': 'failed',
            'reason': f'{type(exc).__name__}: {exc}', 'checks': run.rows if run else []})
        raise
    finally:
        evaluator.close()
    write_json(Path(output)/'design_verification.json', report)
    return report


def require_design_verification(config, output, endpoint, solution=None, *, backend=None, bar=None):
    if config.solver.calibration is None:
        return None
    from .provenance import checkpoint_identity
    from .runtime import checkpoint_directory
    from .block_io import geometry_data, endpoint_data
    design = geometry_data(config, endpoint, solution) if solution is not None else endpoint_data(endpoint)
    coordinates = {'z': solution.z_m, 'wavelengths': solution.wavelengths_m} if solution is not None else None
    criteria = {k:v for k,v in config.as_dict()['convergence'].items() if k not in {'max_solves','max_seconds','timing_repeats'}}
    key = checkpoint_identity(config, 2, {'design': design, 'coordinates': coordinates,
                                       'verification': criteria, 'verification_code': implementation_identity(),
                                       'wavelength_policy':'endpoints_only'})
    directory = (checkpoint_directory(config)/'verification'/key if config.execution.reuse_verification
                 else Path(output)/'block0_checks'/uuid.uuid4().hex[:12])
    report = verify_design(config, directory, endpoint, solution, bar=bar,
                           evaluator_factory=lambda cfg: NativeEvaluator(cfg, backend=backend),
                           resume=(directory/"checkpoint.json").exists(),endpoint_wavelengths_only=True)
    if report['status'] == 'incomplete':
        raise BudgetExceeded(f'Block 0 selected-design verification incomplete; resume evidence in {directory}.')
    if report['status'] != 'passed':
        raise ValueError(f'Block 0 selected-design verification {report["status"]}; see {directory}. Review mode selection and recalibrate before publishing this design.')
    return {'status': 'passed', 'certificate_id': config.solver.calibration['certificate_id'],
            'report': str(directory/'design_verification.json'), 'checks': len(report['checks']),
            'wavelength_policy':'endpoints_only'}
