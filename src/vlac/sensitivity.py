"""Process-supplied perturbations of frozen controls; never re-optimize cases."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from .block_io import load_block, digest, geometry_data, write_json, validate_grid
from .checkpoints import Checkpoint
from .provenance import checkpoint_identity
from .runtime import output_attempt, checkpoint_directory, resolved_cache, RunBudget
from .materials import build_material_bank
from .profiles import evaluate_design_profiles
from .propagate import propagate_state
from .export import _write_csv

ALLOWED = {'upper_width_bias_m', 'lower_width_bias_m', 'thickness_bias_m',
           'gap_bias_m', 'rib_bias_m', 'alignment_bias_m', 'sin_index_offset', 'liquid_index_offset'}


def load_cases(path):
    specification = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(specification.get('source'), str) or not specification['source'].strip():
        raise ValueError('Sensitivity specification must identify the process/metrology source; no inferred tolerances.')
    cases = specification.get('cases', [])
    if not cases: raise ValueError('Supply at least one process case')
    names = set()
    for case in cases:
        name = case.get('name')
        if not isinstance(name, str) or not name or name in names: raise ValueError('Case names must be unique nonempty strings')
        names.add(name)
        if 'block0_dir' in case and not isinstance(case['block0_dir'], str): raise ValueError('block0_dir must be a certificate path')
        if set(case)-ALLOWED-{'name','block0_dir'}: raise ValueError('Unsupported sensitivity parameter')
        if not all(isinstance(v, (int,float)) and not isinstance(v,bool) and np.isfinite(v) for k,v in case.items() if k not in {'name','block0_dir'}):
            raise ValueError('Perturbations must be finite SI values (index offsets dimensionless)')
    return specification


def perturb(config, endpoint, solution, case):
    g, m = config.geometry, config.materials
    dw, dl = case.get('upper_width_bias_m',0.), case.get('lower_width_bias_m',0.)
    dh, dg, dr, dx = (case.get(k,0.) for k in ('thickness_bias_m','gap_bias_m','rib_bias_m','alignment_bias_m'))
    widths = solution.w1_control_m+dw
    offsets = solution.d_c2c_control_m+dx
    rib, gap = endpoint.h_rib_m+dr, endpoint.h_gap_m+dg
    geometry = replace(g, w1_min_m=min(g.w1_min_m,float(min(widths))), w1_max_m=max(g.w1_max_m,float(max(widths))),
        w1_lower_m=g.w1_lower_m+dl, h_wg_m=g.h_wg_m+dh, h_gap_candidates_m=[gap], h_gap_bounds_m=[gap,gap],
        h_rib_candidates_m=[rib], h_rib_bounds_m=[rib,rib], d_c2c_input_m=float(offsets[0]), d_c2c_output_m=float(offsets[-1]),
        d_c2c_bounds_m=[min(g.d_c2c_bounds_m[0],float(min(offsets))),max(g.d_c2c_bounds_m[1],float(max(offsets)))])
    if case.get('liquid_index_offset',0.) and m.liquid_dispersion_csv:
        raise ValueError('A constant liquid-index offset cannot override a supplied dispersion table')
    materials = replace(m, sin_absolute_index_offset=m.sin_absolute_index_offset+case.get('sin_index_offset',0.),
        liquid_cross_n=m.liquid_cross_n+case.get('liquid_index_offset',0.), liquid_bar_n=m.liquid_bar_n+case.get('liquid_index_offset',0.))
    varied = replace(config, geometry=geometry, materials=materials)
    if case.get('block0_dir'):
        from .block0 import apply_certificate
        varied = replace(varied,solver=replace(varied.solver,calibration=None))
        varied = apply_certificate(varied,case['block0_dir'])
    elif config.solver.calibration and any(case.get(k,0.) for k in ALLOWED):
        raise ValueError('Perturbed geometry/materials require a matching case block0_dir certificate.')
    varied.validate()
    return varied, replace(endpoint, w1_input_m=float(widths[0]),w1_output_m=float(widths[-1]),h_gap_m=gap,h_rib_m=rib), widths, offsets


def run(config, source, specification, output, backend_factory):
    endpoint, solution, _, artifact = load_block(source, 2, config)
    # Validate every case before opening MODE.
    prepared = [(case, perturb(config,endpoint,solution,case)) for case in specification['cases']]
    budget = RunBudget(config.execution)
    cases, rows = [], []
    with output_attempt(output, 'vlac sensitivity') as attempt:
        for case, (varied, ep, widths, offsets) in prepared:
            key = checkpoint_identity(varied, 3, {'nominal_geometry': geometry_data(config,endpoint,solution),
                'case': case, 'propagation': varied.as_dict()['propagation'], 'sampling': [solution.z_m,solution.wavelengths_m], 'source': specification['source']})
            checkpoint = Checkpoint(checkpoint_directory(config), 'sensitivity', key)
            saved = checkpoint.load() if config.execution.resume else None
            if saved is None:
                bank = build_material_bank(varied)
                result, quality = [], {}
                with backend_factory(varied, cache_path=resolved_cache(config)) as backend:
                    backend.budget = budget
                    for state in ('cross','bar'):
                        grid = evaluate_design_profiles(backend,varied,bank,ep,widths,offsets,solution.z_m,solution.wavelengths_m,state=state)
                        quality[state] = validate_grid(grid,solution,varied)
                        result.extend(propagate_state(solution.z_m,grid,solution.wavelengths_m,state=state,settings=config.propagation))
                saved = {'case': case, 'performance': result, 'quality': quality,
                         'case_id': key, 'reoptimized': False}
                checkpoint.save(saved)
            cases.append(saved)
            rows.extend(dict(case=case['name'], **row) for row in saved['performance'])
            write_json(attempt/'sensitivity_checkpoint.json', {'completed_cases': [v['case']['name'] for v in cases]})
        report = {'status': 'review', 'nominal_geometry_id': geometry_data(config,endpoint,solution)['geometry_id'],
            'source': specification['source'], 'cases': cases,
            'scope': 'User-supplied deterministic process cases on frozen controls; no probability/yield or new grid-convergence claim.',
            'physical_device_validation': False, 'upstream_validation': artifact['validation']}
        report['sha256'] = digest(report)
        write_json(attempt/'sensitivity.json',report)
        _write_csv(attempt/'sensitivity_by_launch.csv', list(rows[0]), rows)
    return report
