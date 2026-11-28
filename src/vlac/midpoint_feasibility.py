"""Aligned midpoint width screen and separate numerical sensitivity probes.

This diagnostic preserves the Block 1 target and never publishes new endpoints
or Block 2 geometry. All optical samples are real native FDE calculations.
"""
import argparse
from contextlib import closing
from dataclasses import asdict, replace
import csv
import hashlib
from pathlib import Path
import sqlite3
import time

import numpy as np

from .block_io import clean, load_block, write_json
from .checkpoints import decode
from .coefficient_stability import compare_coefficients
from .config import load_config
from .geometry import make_geometry, solution_profiles
from .materials import build_material_bank
from .progress import ProgressReporter
from .runtime import BudgetExceeded
from .tracking import CandidateRejected, require_anchor_match


class RawFallback:
    """Read exact-key raw packets without touching the source cache."""
    def __init__(self, primary, source):
        self.primary, self.source, self.hits = primary, Path(source).resolve(), 0

    def get(self, key, *, kind='other'):
        value = self.primary.get(key, kind=kind)
        if value is not None or kind != 'native_raw_modes':
            return value
        with closing(sqlite3.connect(self.source.as_uri()+'?mode=ro', uri=True)) as db:
            row = db.execute('SELECT payload,sha256 FROM cache_v2 WHERE cache_key=?', (key,)).fetchone()
        if row is None:
            return None
        if hashlib.sha256(row[0]).hexdigest() != row[1]:
            raise ValueError('Source raw cache checksum mismatch')
        value = decode(row[0])
        self.primary.put(key, value, kind=kind)
        self.hits += 1
        return value

    def put(self, key, value, *, kind='other'):
        return self.primary.put(key, value, kind=kind)


def candidates(records, waves, scale):
    latest = {}
    for r in records:
        if r['setting'] == 'production':
            latest[(r['width_m'], r['wavelength_m'])] = r
    result = []
    for width in sorted({key[0] for key in latest}):
        rows = [latest.get((width, float(w))) for w in waves]
        if any(r is None or not r['valid'] for r in rows):
            continue
        cs = [r['coefficients'] for r in rows]
        ks = [complex(c['kappa']['real'], c['kappa']['imag']) for c in cs]
        delta = [c['delta_beta']['real'] for c in cs]
        coupling_error = max(abs(k-scale/2)/(scale/2) for k in ks)
        delta_error = max(abs(v)/scale for v in delta)
        result.append({'width_m': width, 'kappa_real_rad_per_m': [k.real for k in ks],
            'kappa_imag_rad_per_m': [k.imag for k in ks],
            'D_over_abs_kappa': [scale/abs(k) if abs(k)>0 else None for k in ks],
            'delta_beta_real_rad_per_m': delta, 'coupling_error': coupling_error,
            'detuning_error': delta_error, 'midpoint_objective': max(coupling_error, delta_error),
            'minimum_real_kappa_rad_per_m': min(k.real for k in ks)})
    return result


def refinement_widths(rows):
    """Estimate each detuning zero and the common-width minimax crossing.

    These are proposals for additional DIRECT solves, never reported optics.
    Rejected samples split the intervals in the caller's coarse grid.
    """
    proposed = []
    for a, b in zip(rows, rows[1:]):
        da, db = a['delta_beta_real_rad_per_m'], b['delta_beta_real_rad_per_m']
        for va, vb in zip([*da, sum(da)], [*db, sum(db)]):
            if va*vb < 0:
                proposed.append(a['width_m']-va*(b['width_m']-a['width_m'])/(vb-va))
    return sorted(set(proposed))


def summary(records, waves, scale):
    rows = candidates(records, waves, scale)
    def best(key, reverse=False):
        return sorted(rows, key=lambda r: (r[key], r['width_m']), reverse=reverse)[0] if rows else None
    return {'candidates': rows, 'best_detuning_fit': best('detuning_error'),
        'best_coupling_fit': best('coupling_error'), 'best_full_midpoint_fit': best('midpoint_objective'),
        'strongest_common_width': best('minimum_real_kappa_rad_per_m', True),
        'scope': 'Direct samples in the fixed endpoint width interval; no continuous optimum or full-device claim.'}


def run(config, profile_dir, out, *, samples=17, old_cache=None):
    from .native_backend import NativeLumapiBackend
    from .provenance import file_digest
    endpoint, solution, _, artifact = load_block(profile_dir, 2, config)
    if samples < 3 or config.solver.engine != 'native' or config.solver.tracking_method != 'guarded_translation_v1':
        raise ValueError('At least three widths and the guarded native backend are required')
    out = Path(out).resolve()
    if (out/'report.json').exists():
        raise FileExistsError('Use a new diagnostic output directory')
    out.mkdir(parents=True, exist_ok=True)
    waves = np.array([config.device.wavelength_min_m, config.device.wavelength_max_m])
    center = float(solution_profiles(config, solution)[0](config.device.length_m/2))
    lo, hi = endpoint.w1_input_m, endpoint.w1_output_m
    prior = np.asarray(solution.w1_control_m)
    widths = np.unique(np.r_[np.linspace(lo, hi, samples), prior[(prior>=lo)&(prior<=hi)], center])
    scale = endpoint.delta_beta_max_rad_per_m
    progress = ProgressReporter('vlac check-midpoint', out)
    backend = NativeLumapiBackend(config, cache_path=out/'cache/native_fde_cache.sqlite', progress=progress)
    if old_cache:
        backend.cache = RawFallback(backend.cache, old_cache)
    bank, live = build_material_bank(config), {}
    report = {'method': 'aligned_midpoint_feasibility_v1', 'status': 'running', 'execution_completed': False,
        'D_rad_per_m': scale, 'kappa_target_rad_per_m': scale/2, 'required_D_over_kappa': 2.,
        'offset_m': 0., 'fixed_endpoint_widths_m': [lo, hi], 'initial_widths_m': widths,
        'reference_width_m': center, 'wavelengths_m': waves,
        'target_amplitudes_changed': False, 'physical_device_validated': False,
        'source_profile_sha256': artifact['sha256'], 'source_profile_content_id': artifact['content_id'],
        'diagnostic_source_sha256': file_digest(__file__), 'records': [], 'sensitivity_comparisons': [],
        'reused_raw_cache_read_only': str(Path(old_cache).resolve()) if old_cache else None,
        'budget': asdict(config.execution), 'numerical_scope': 'Production width screen, then isolated mesh and x-domain sensitivity; no Block 0 certificate.'}
    write_json(out/'configuration.json', config.as_dict())
    started = time.monotonic()
    def save():
        report.update(elapsed_seconds=time.monotonic()-started, fresh_fde_solves=backend.budget.solves,
            runtime=progress.runtime_stats, raw_cache_packets_reused=getattr(backend.cache, 'hits', 0),
            summary=summary(report['records'], waves, scale))
        write_json(out/'report.json', report)
    def probe(width, wave, refs=(), kappa=None, setting='production'):
        g = make_geometry(config.geometry, w1_m=float(width), d_c2c_m=0.,
            h_rib_m=endpoint.h_rib_m, h_gap_m=endpoint.h_gap_m, state='cross',
            liquid_index=float(bank.liquid_cross.index(wave)))
        c = ref = None
        error = None
        try:
            c, ref = backend._guarded_coefficients(g, float(wave), tuple(refs), kappa, diagnostic=True)
            if not ref.valid:
                error = 'Rejected extraction: '+c.extraction_flag
        except CandidateRejected as exc:
            error = str(exc)
        report['records'].append({'setting': setting, 'width_m': float(width), 'wavelength_m': float(wave),
            'valid': error is None, 'error': error, 'coefficients': clean(asdict(c)) if c else None,
            'mesh_cells': [backend.config.solver.mesh_cells_x, backend.config.solver.mesh_cells_y],
            'domain_m': [backend.x_span_m, backend.y_min_m, backend.y_max_m]})
        save()
        progress.note(f'{setting}: width={width*1e6:.7g} um, {wave*1e9:.0f} nm; '
            +(f'kappa={c.kappa:.7g}; delta={c.delta_beta.real:.7g}; valid={error is None}' if c else str(error)), force=True)
        if error:
            raise CandidateRejected(error)
        if setting == 'production':
            live[(float(width), float(wave))] = (c, ref)
        return c, ref
    def tracked(width, wave, previous, depth):
        previous_width, c, ref = previous
        across = live.get((float(width), float(waves[0]))) if wave != waves[0] else None
        refs = [ref]+([across[1]] if across else [])
        try:
            return probe(width, wave, refs, c.kappa)
        except CandidateRejected:
            if depth <= 0:
                raise
            middle = (previous_width+width)/2
            cm, rm = tracked(middle, wave, previous, depth-1)
            return tracked(width, wave, (middle, cm, rm), depth-1)
    save()
    try:
        with backend:
            progress.start_stage('MODE and strict endpoint preflight')
            backend.preflight(float(waves[0])); backend.require_endpoint_consistency(endpoint, progress=progress)
            progress.finish_stage()
            progress.start_stage('Aligned production width sweep')
            for wave in waves:
                other = live.get((center, float(waves[0]))) if wave != waves[0] else None
                seed = probe(center, wave, [other[1]] if other else ())
                for direction in (widths[widths<center][::-1], widths[widths>center]):
                    previous = (center, *seed)
                    for width in direction:
                        try:
                            c, ref = tracked(float(width), wave, previous, config.solver.tracking_recovery_depth)
                        except CandidateRejected:
                            continue
                        previous = (float(width), c, ref)
            # Only interpolate proposals between adjacent valid coarse samples.
            coarse = candidates(report['records'], waves, scale)
            pairs = [coarse[i:i+2] for i in range(len(coarse)-1)
                     if not any(coarse[i]['width_m']<w<coarse[i+1]['width_m'] for w in widths)]
            proposals = sorted(set(w for pair in pairs for w in refinement_widths(pair)))
            report['refined_widths_m'] = proposals
            for width in proposals:
                for wave in waves:
                    nearest = min((w for (w, lam) in live if lam == wave), key=lambda w: abs(w-width))
                    try:
                        tracked(width, wave, (nearest, *live[(nearest, float(wave))]), config.solver.tracking_recovery_depth)
                    except CandidateRejected:
                        pass
            progress.finish_stage()
            selected = summary(report['records'], waves, scale)
            sentinels = sorted(set(r['width_m'] for r in (selected['best_detuning_fit'], selected['strongest_common_width']) if r))
            report['sensitivity_widths_m'] = sentinels
            progress.start_stage('Sentinel mesh and x-domain sensitivity')
            base_x, base_y = backend.x_span_m, (backend.y_min_m, backend.y_max_m)
            settings = [('mesh_1p25', 1.25, 1.), ('mesh_1p5', 1.5, 1.),
                        ('x_domain_1p25', 1., 1.25), ('x_domain_1p5', 1., 1.5)]
            previous = {}
            for label, mesh, domain in settings:
                updated = replace(config, solver=replace(config.solver,
                    mesh_cells_x=round(config.solver.mesh_cells_x*mesh*domain),
                    mesh_cells_y=round(config.solver.mesh_cells_y*mesh)))
                backend.reconfigure(updated)
                with backend._temporary_domain(x_span_m=base_x*domain, y_min_m=base_y[0], y_max_m=base_y[1]):
                    for width in sentinels:
                        for wave in waves:
                            baseline, ref = live[(width, float(wave))]
                            try:
                                c, current = probe(width, wave, [ref], baseline.kappa, label)
                                overlaps = [require_anchor_match(a, b, config) for a, b in zip(
                                    (current.upper, current.lower, *current.supermodes), (ref.upper, ref.lower, *ref.supermodes))]
                                axis = 'mesh' if mesh>1 else 'x_domain'
                                old = previous.get((axis, width, float(wave)), baseline)
                                comparison = {'setting': label, 'width_m': width, 'wavelength_m': float(wave),
                                    'anchor_field_overlaps': overlaps, 'baseline': compare_coefficients(config, c, baseline, wave),
                                    'previous_setting': compare_coefficients(config, c, old, wave)}
                                previous[(axis, width, float(wave))] = c
                            except CandidateRejected as exc:
                                comparison = {'setting': label, 'width_m': width, 'wavelength_m': float(wave), 'error': str(exc)}
                            report['sensitivity_comparisons'].append(comparison); save()
            progress.finish_stage()
        report.update(status='completed', execution_completed=True)
        report['sentinel_checks_passed'] = bool(report['sensitivity_comparisons']) and all(
            r.get('baseline', {}).get('passed', False) and r.get('previous_setting', {}).get('passed', False)
            for r in report['sensitivity_comparisons'])
        progress.finish()
    except Exception as exc:
        report.update(status='incomplete' if isinstance(exc, BudgetExceeded) else 'failed', error=f'{type(exc).__name__}: {exc}')
        progress.fail(exc)
    finally:
        save(); backend.close()
    export(out, report)
    return report


def export(out, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = report['summary']['candidates']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    with (out/'width_sweep.csv').open('w', encoding='utf-8', newline='') as f:
        fields = ['width_m', 'wavelength_m', 'kappa_real_rad_per_m', 'kappa_imag_rad_per_m', 'D_over_abs_kappa', 'delta_beta_real_rad_per_m']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for i, wave in enumerate(report['wavelengths_m']):
            for r in rows:
                writer.writerow({'width_m': r['width_m'], 'wavelength_m': wave, **{k:r[k][i] for k in fields[2:]}})
            x = [r['width_m']*1e6 for r in rows]
            for ax, key in zip(axes.flat, fields[2:]):
                # Points only: no implied optical interpolation across rejected widths.
                ax.plot(x, [r[key][i] for r in rows], 'o', label=f'{wave*1e9:.0f} nm')
                ax.set_title(key); ax.set_xlabel('Upper width (um)'); ax.grid(alpha=.25)
    axes[0, 0].axhline(report['kappa_target_rad_per_m'], color='k', ls='--', label='D/2 target')
    axes[1, 0].axhline(2, color='k', ls='--', label='Required ratio')
    axes[1, 1].axhline(0, color='k', ls='--', label='Midpoint detuning target')
    for ax in axes.flat: ax.legend(fontsize=8)
    fig.suptitle('Direct FDE aligned width feasibility; fixed endpoints and target; production mesh')
    for suffix in ('png', 'pdf'): fig.savefig(out/f'width_sweep.{suffix}', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='vlac check-midpoint', description=__doc__)
    for name in ('config', 'profile-dir', 'out'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--reuse-cache', type=Path)
    parser.add_argument('--width-samples', type=int, default=17)
    parser.add_argument('--max-solves', type=int, default=300)
    parser.add_argument('--max-seconds', type=float, default=5400.)
    args = parser.parse_args(argv)
    if args.max_solves<1 or not np.isfinite(args.max_seconds) or args.max_seconds<=0:
        parser.error('Positive solve and finite time budgets required')
    config = load_config(args.config)
    config = replace(config, execution=replace(config.execution, max_solves=args.max_solves, max_seconds=args.max_seconds),
                     solver=replace(config.solver, hide_gui=True))
    report = run(config, args.profile_dir, args.out, samples=args.width_samples, old_cache=args.reuse_cache)
    print(f'Midpoint feasibility {report["status"]}: {Path(args.out).resolve()/"report.json"}', flush=True)
    return 0 if report['execution_completed'] else 3 if report['status']=='incomplete' else 2
