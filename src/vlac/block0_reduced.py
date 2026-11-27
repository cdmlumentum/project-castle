"""Scoped, nine-setting MODE convergence alternative with a 200-call lifetime cap.

This method produces evidence for its recorded stress cases, never the full
Block 0 certificate or production configuration. All optical data come from MODE.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np

from . import block0 as full
from .block_io import clean, digest, write_json

try:
    from .runtime import BudgetExceeded as RuntimeBudgetExceeded
except ModuleNotFoundError:
    # The reduced method also works in the v25 checkout, before shared budgets.
    RuntimeBudgetExceeded = full.BudgetExceeded

METHOD = 'mode_mesh_lateral_sentinel_v1'
HARD_MAX_SOLVES = 200
SETTINGS = (
    ('A', 2e-6, 1.25, 1.25, False),
    ('B', 2e-6, 1.00, 1.25, False),
    ('C', 2e-6, 0.80, 1.25, False),
    ('D', 2e-6, 0.80, 1.00, False),
    ('E', 2e-6, 0.80, 0.80, False),
    ('F', 3e-6, 0.80, 0.80, False),
    ('G', 4e-6, 0.80, 0.80, False),
    ('H', 5e-6, 0.64, 0.64, False),
    ('I', 5e-6, 0.64, 0.64, True),
)
COMPARISONS = (
    ('mesh_x_1', 'A', 'B'), ('mesh_x_2', 'B', 'C'),
    ('mesh_y_1', 'C', 'D'), ('mesh_y_2', 'D', 'E'),
    ('lateral_1', 'E', 'F'), ('lateral_2', 'F', 'G'),
    ('joint_guard', 'G', 'H'), ('pml_guard', 'H', 'I'),
    ('candidate_to_reference', 'A', 'I'),
)
SCOPE = ('Transverse x/y mesh and lateral-domain convergence at four recorded '
         'cases per solve family; fixed vertical domain. No full-envelope '
         'certification, global fastest-mesh claim, or physical device validation.')


@contextmanager
def _output_lock(output):
    """Serialize the journal; OS locks release automatically after a crash."""
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    with (path/'.reduced.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'\0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise FileExistsError('Reduced Block 0 output is already in use by another run.') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def stress_cases(config, source=None):
    """Use explicit SI cases, or declared provisional corners/midpoint cases."""
    g, d = config.geometry, config.device
    if len(set(g.h_gap_bounds_m)) != 1 or len(set(g.h_rib_bounds_m)) != 1:
        raise ValueError('Reduced Block 0 requires fixed gap and rib height.')
    zero = float(np.clip(0., *g.d_c2c_bounds_m))
    defaults = {'h_gap_m': g.h_gap_bounds_m[0], 'h_rib_m': g.h_rib_bounds_m[0]}
    if source is None:
        mid = (g.w1_min_m+g.w1_max_m)/2
        wl, wh = d.wavelength_min_m, d.wavelength_max_m
        rows = [(g.w1_min_m, wh, 'cross'), (g.w1_max_m, wl, 'cross'),
                (g.w1_min_m, wl, 'bar'), (g.w1_max_m, wh, 'bar')]
        raw = {'isolated': [dict(w1_m=w, d_c2c_m=zero, wavelength_m=wave, state=state)
                            for w, wave, state in rows],
               'coupled': [dict(w1_m=w, d_c2c_m=offset, wavelength_m=wave, state=state)
                           for w, offset, wave, state in (
                               (g.w1_min_m, g.d_c2c_bounds_m[1], wh, 'cross'),
                               (mid, zero, wl, 'cross'),
                               (g.w1_min_m, zero, wl, 'bar'),
                               (g.w1_max_m, g.d_c2c_bounds_m[0], wh, 'bar'))]}
    else:
        raw = json.loads(Path(source).read_text(encoding='utf-8')) if not isinstance(source, dict) else source
    if not isinstance(raw, dict) or set(raw) != set(full.KINDS):
        raise ValueError('Cases must contain exactly isolated and coupled lists.')
    expected = {(state, wave) for state in ('cross', 'bar')
                for wave in (d.wavelength_min_m, d.wavelength_max_m)}
    result = {}
    for kind in full.KINDS:
        if not isinstance(raw[kind], list) or len(raw[kind]) != 4:
            raise ValueError('Reduced Block 0 requires four cases per family.')
        result[kind] = []
        for row in raw[kind]:
            if not isinstance(row, dict):
                raise ValueError('Each stress case must be an SI-valued object.')
            case = {**defaults, **row}
            if set(case) != {'w1_m', 'd_c2c_m', 'wavelength_m', 'state', 'h_gap_m', 'h_rib_m'}:
                raise ValueError('Stress cases require width, offset, wavelength and state; gap/rib are optional.')
            for key in set(case)-{'state'}:
                if isinstance(case[key], bool) or not isinstance(case[key], (int, float)) or not math.isfinite(case[key]):
                    raise ValueError('Stress-case coordinates must be finite SI numbers.')
                case[key] = float(case[key])
            full.make_geometry(g, liquid_index=1., **{k: case[k] for k in
                ('w1_m', 'd_c2c_m', 'h_gap_m', 'h_rib_m', 'state')})
            if kind == 'isolated' and case['d_c2c_m'] != zero:
                raise ValueError('Isolated stress cases must use the centered offset.')
            result[kind].append(case)
        if {(case['state'], case['wavelength_m']) for case in result[kind]} != expected:
            raise ValueError('Each family must cover both states at both band edges exactly once.')
    return result


def solver_settings(config):
    """Keep physical dx/dy independent of x span, and all fields on one grid."""
    c, g = config.convergence, config.geometry
    result = {}
    for kind, base in full.base_profiles(config).items():
        envelope = max(g.w1_max_m, g.w1_lower_m)
        if kind == 'coupled':
            envelope += 2*max(abs(v) for v in g.d_c2c_bounds_m)
        span = envelope+10e-6
        tracking = dict(tracking_x_span_m=span, tracking_y_min_m=base.y_min_m,
                        tracking_y_max_m=base.y_max_m,
                        tracking_nx=max(17, 2*math.ceil(span/c.comparison_dx_m/2)+1),
                        tracking_ny=max(17, 2*math.ceil((base.y_max_m-base.y_min_m)/c.comparison_dy_m/2)+1))
        result[kind] = {label: replace(base, x_span_m=envelope+2*padding,
            dx_m=base.dx_m*fx, dy_m=base.dy_m*fy,
            pml_layers=c.pml_layers+(c.pml_test_extra_layers if extra_pml else 0), **tracking)
            for label, padding, fx, fy, extra_pml in SETTINGS}
        # Domain comparisons must preserve the ACTUAL grid spacing and its
        # registration at x=0, not merely a requested maximum cell step.
        anchor = result[kind]['E']
        actual_dx = anchor.x_span_m/anchor.nx
        for label, padding in (('F', 3e-6), ('G', 4e-6)):
            requested = (envelope+2*padding)/actual_dx
            cells = max(2, math.ceil(requested-1e-12*max(1., requested)))
            cells += (cells-anchor.nx) % 2
            result[kind][label] = replace(anchor, x_span_m=cells*actual_dx,
                dx_m=float(np.nextafter(actual_dx, math.inf)))
        # One outward ULP above actual_dx prevents old SolverProfile versions
        # from adding a roundoff cell; the resulting physical lattice is exact.
        largest = max(p.x_span_m for p in result[kind].values())
        result[kind] = {label: replace(p, tracking_x_span_m=largest,
            tracking_nx=max(17, 2*math.ceil(largest/c.comparison_dx_m/2)+1))
            for label, p in result[kind].items()}
    return result


def prepare_plan(config, cases=None):
    config.validate()
    if config.solver.calibration is not None:
        raise ValueError('Use the original uncalibrated configuration for reduced Block 0.')
    checked = stress_cases(config, cases)
    profiles = solver_settings(config)
    plan = {'schema': 1, 'method': METHOD, 'scope': SCOPE, 'production_certified': False,
        'input_id': full.input_identity(config),
        'implementation_id': digest({'shared': full.implementation_identity(),
            'reduced': hashlib.sha256(Path(__file__).read_text(encoding='utf-8').replace('\r\n', '\n').encode()).hexdigest()}),
        'installation_id': full.installation_identity(config),
        'case_selection': ('Explicit user-supplied stress cases.' if cases is not None else
            'Provisional bounds and midpoint; midpoint is NOT a verified phase-matching width. '
            'Use --cases with existing modal evidence to choose near-phase-matched/weakly guided cases.'),
        'cases': checked, 'settings': {kind: {label: asdict(p) for label, p in values.items()}
                                     for kind, values in profiles.items()},
        'comparisons': [dict(stage=stage, candidate=a, reference=b) for stage, a, b in COMPARISONS],
        'lateral_mesh_policy': 'F/G expand the E lattice outward to equal-parity cell counts, preserving actual spacing and registration.',
        'planned_fde_calls': 180, 'hard_lifetime_fde_call_limit': HARD_MAX_SOLVES,
        'reserve_fde_calls': 20,
        'omitted_checks': ['vertical-domain convergence', 'full geometry/wavelength envelope',
                           'adaptive phase-matching scan', 'fastest-candidate search', 'fresh timing replicates']}
    plan['plan_id'] = digest(plan)
    return plan


class ReducedRun:
    def __init__(self, config, output, evaluator, plan, *, resume=False):
        self.config, self.evaluator, self.plan = config, evaluator, plan
        self.out = Path(output).resolve()
        self.started, self.solves, self.executed = time.monotonic(), 0, 0
        self.total_solves, self.pending, self.rows = 0, None, []
        self.identity = {'method': METHOD, 'plan_id': plan['plan_id'],
            'synthetic': bool(evaluator.synthetic),
            'timing_environment': {'host': platform.node(), 'threads': config.solver.threads}}
        checkpoint = self.out/'checkpoint.json'
        if checkpoint.exists():
            if not resume:
                raise FileExistsError('Reduced Block 0 output exists; use --resume or a new directory.')
            saved = json.loads(checkpoint.read_text(encoding='utf-8'))
            checksum = saved.pop('checkpoint_id', None)
            if checksum != digest(saved):
                raise ValueError('Reduced Block 0 checkpoint checksum mismatch.')
            if saved['identity'] != self.identity:
                raise ValueError('Reduced Block 0 resume inputs, implementation or MODE installation changed.')
            self.total_solves = saved['total_solves']
            if type(self.total_solves) is not int or not 0 <= self.total_solves <= HARD_MAX_SOLVES:
                raise ValueError('Invalid lifetime FDE-call accounting.')
        elif resume:
            raise FileNotFoundError('No reduced Block 0 checkpoint exists to resume.')
        elif self.out.exists() and any(p.name != '.reduced.lock' for p in self.out.iterdir()):
            raise FileExistsError('Choose an empty output directory for reduced Block 0.')
        self.samples = self.out/'samples'
        self.samples.mkdir(parents=True, exist_ok=True)
        self.checkpoint()
        write_json(self.out/'plan.json', plan)
        write_json(self.out/'test_cases.json', plan['cases'])

    def checkpoint(self):
        value = {'identity': self.identity, 'total_solves': self.total_solves,
                 'session_solves': self.solves, 'pending_reservation': self.pending}
        value['checkpoint_id'] = digest(value)
        write_json(self.out/'checkpoint.json', value)

    def sample(self, kind, case, profile):
        key = digest({'identity': self.identity, 'kind': kind, 'case': case, 'profile': asdict(profile)})
        meta, fields = self.samples/(key+'.json'), self.samples/(key+'.npz')
        if meta.exists() and fields.exists():
            metadata = json.loads(meta.read_text(encoding='utf-8'))
            if metadata['key'] != key or metadata['npz_sha256'] != full.file_hash(fields):
                raise ValueError('Reduced Block 0 sample checksum mismatch.')
            check = {k: v for k, v in metadata.items() if k != 'metadata_id'}
            if metadata.get('metadata_id') != digest(check):
                raise ValueError('Reduced Block 0 sample metadata checksum mismatch.')
            with np.load(fields, allow_pickle=False) as arrays:
                value = {name: arrays[name].copy() for name in ('neff', 'fields', 'te')}
            value.update(metadata['values'])
            for name in ('delta_beta', 'kappa'):
                if name in value:
                    value[name] = complex(**value[name])
            return value
        base_calls = 2 if kind == 'isolated' else 3
        reserved = base_calls*(1+self.config.solver.zero_mode_retries)
        limits = self.config.convergence
        if (self.total_solves+reserved > HARD_MAX_SOLVES
                or self.solves+reserved > min(limits.max_solves, HARD_MAX_SOLVES)
                or time.monotonic()-self.started >= limits.max_seconds):
            raise full.BudgetExceeded('Reduced Block 0 solve/time budget reached; lifetime cap remains 200 across resumes.')
        # Persist the worst case BEFORE dispatch. A crash or failed batch keeps
        # its reservation charged; a restart cannot reset the lifetime cap.
        self.total_solves += reserved
        self.solves += reserved
        self.pending = {'key': key, 'reserved_fde_calls': reserved}
        self.checkpoint()
        value = self.evaluator(kind, case, profile)
        calls = value['solves']
        if type(calls) is not int or not base_calls <= calls <= reserved:
            raise ValueError('Evaluator returned invalid FDE-call accounting.')
        if not all(np.all(np.isfinite(value[name])) for name in ('neff', 'fields', 'te')):
            raise ValueError('Reduced Block 0 sample contains non-finite modes or fields.')
        temporary = fields.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **{name: value[name] for name in ('neff', 'fields', 'te')})
        temporary.replace(fields)
        metadata = {'key': key, 'npz_sha256': full.file_hash(fields),
                    'values': clean({k: v for k, v in value.items() if k not in {'neff', 'fields', 'te'}})}
        metadata['metadata_id'] = digest(metadata)
        write_json(meta, metadata)
        self.total_solves -= reserved-calls
        self.solves -= reserved-calls
        self.executed += 1
        self.pending = None
        self.checkpoint()
        if not (self.out/'pilot.json').exists():
            write_json(self.out/'pilot.json', {'synthetic': bool(self.evaluator.synthetic),
                'findmodes_calls': calls, 'elapsed_seconds': value['elapsed_seconds'],
                'seconds_per_findmodes_including_processing': value['elapsed_seconds']/calls,
                'note': 'One sample timing; finer reference settings can take longer.'})
        return value

    def finish(self, status, reason, completed):
        checks_passed = len(self.rows) == 72 and all(row['passed'] for row in self.rows)
        report = {**self.identity, 'schema': 1, 'status': status, 'reason': reason,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': SCOPE,
            'execution_completed': completed, 'checks_passed': checks_passed,
            'convergence_demonstrated': checks_passed and not self.evaluator.synthetic and completed,
            'production_certified': False, 'total_solves': self.total_solves,
            'session_solves': self.solves, 'session_samples': self.executed,
            'elapsed_seconds': time.monotonic()-self.started, 'plan_id': self.plan['plan_id'],
            'accuracy_limits': asdict(self.config.convergence), 'checks': self.rows,
            'candidate_profiles': {kind: self.plan['settings'][kind]['A'] for kind in full.KINDS},
            'reference_profiles': {kind: self.plan['settings'][kind]['I'] for kind in full.KINDS},
            'call_accounting': 'Includes actual successful attempts and conservative reservations for failed/interrupted samples; lifetime cap 200.',
            'pending_reservation': self.pending}
        report['report_id'] = digest(report)
        write_json(self.out/'reduced_report.json', report)
        write_json(self.out/'convergence_results.json', self.rows)
        columns = ['stage', 'kind', 'case_index', 'candidate', 'reference', 'state',
                   'w1_m', 'd_c2c_m', 'wavelength_m', 'passed', 'maximum_real_neff_error',
                   'delta_beta_error_rad_per_m', 'kappa_error_rad_per_m',
                   'minimum_transverse_e_power_overlap', 'maximum_loss_error_db']
        with (self.out/'convergence_results.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.rows)
        page = ('<!doctype html><meta charset="utf-8"><title>Reduced Block 0 convergence</title>'
                '<style>body{font:15px system-ui;margin:2rem}pre{white-space:pre-wrap}</style>'
                '<h1>Reduced Block 0 convergence</h1><p>'+html.escape(SCOPE)+'</p>'
                '<p>Production certified: false</p><pre>'+html.escape(json.dumps(clean(report), indent=2))+'</pre>')
        (self.out/'report.html').write_text(page, encoding='utf-8')
        self.checkpoint()
        return report

    def run(self):
        try:
            for label, *_ in SETTINGS:
                for kind in full.KINDS:
                    profile = full.SolverProfile(**self.plan['settings'][kind][label])
                    for index, case in enumerate(self.plan['cases'][kind]):
                        current = self.sample(kind, case, profile)
                        for stage, earlier, later in COMPARISONS:
                            if later != label:
                                continue
                            previous = self.sample(kind, case, full.SolverProfile(**self.plan['settings'][kind][earlier]))
                            metrics = full.compare_samples(self.config, previous, current)
                            self.rows.append(dict(stage=stage, kind=kind, case_index=index,
                                candidate=earlier, reference=later, **case, **metrics))
            if not all(row['passed'] for row in self.rows):
                return self.finish('failed', 'At least one required convergence comparison failed; candidate A is not accepted.', True)
            if self.evaluator.synthetic:
                return self.finish('incomplete', 'Synthetic test execution only; no real MODE convergence established.', True)
            return self.finish('passed', 'All required comparisons passed at the recorded stress cases.', True)
        except (full.BudgetExceeded, RuntimeBudgetExceeded) as exc:
            return self.finish('incomplete', str(exc), False)
        except Exception as exc:
            self.finish('failed', f'{type(exc).__name__}: {exc}', False)
            raise


def run_reduced(config, output, *, cases=None, resume=False, evaluator_factory=full.NativeEvaluator):
    plan = prepare_plan(config, cases)
    with _output_lock(output):
        evaluator = evaluator_factory(config)
        try:
            return ReducedRun(config, output, evaluator, plan, resume=resume).run()
        finally:
            evaluator.close()
