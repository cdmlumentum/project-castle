"""Offline workflow inspection and reporting; no solver session is constructed."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from .block_io import load_block, source_record
from .runtime import resolved_cache, resolve_published, output_attempt

BLOCKS = {1: 'endpoint search', 2: 'cross-state optimization',
          3: 'frozen-profile characterization', 4: 'two-mode propagation'}


def plan(config, profile_dir=None):
    nz, nw = config.device.z_samples_initial, 2
    verification = None
    if profile_dir:
        _, solution, _, _ = load_block(profile_dir, 2, config)
        nz, nw = len(solution.z_m), len(solution.wavelengths_m)
        check_nz = 2*nz-1
        check_nw = len(np.unique(np.r_[solution.wavelengths_m,
            (config.device.wavelength_min_m+config.device.wavelength_max_m)/2]))
        verification = check_nz*check_nw*2*(2+3)*2
    return {'blocks': BLOCKS, 'cache': str(resolved_cache(config)),
            'budgets': config.as_dict()['execution'],
            'initial_optimization_grid': [nw, nz], 'cold_solves_per_objective': 3*nw*nz,
            'selected_profile_verification_cold_solves': verification,
            'worst_case_retry_multiplier': 1+config.solver.zero_mode_retries,
            'characterization': config.as_dict()['characterization'],
            'scope': 'Cost estimates before cache reuse; excludes startup, inspection and retries. This is not a convergence certificate.',
            'wall_time_policy': 'Soft limit checked between solver batches; an active eigensolve may finish after the limit.'}


def inspect(config, source, block):
    endpoint, solution, bar, artifact = load_block(source, block, config)
    result = {'block': block, 'source': str(resolve_published(source)),
              'content_id': artifact.get('content_id'), 'provenance': artifact.get('provenance'),
              'validation': artifact.get('validation'), 'endpoint': {
                  'w1_input_m': endpoint.w1_input_m, 'w1_output_m': endpoint.w1_output_m}}
    if solution is not None:
        result.update(geometry_id=artifact['payload'].get('geometry',{}).get('geometry_id'),
                      optimization_shape=[len(solution.wavelengths_m), len(solution.z_m)])
    dataset = artifact.get('payload',{}).get('characterization')
    if dataset:
        result['characterization'] = {'shape': [len(dataset['wavelengths_m']),len(dataset['z_m'])],
                                     'validation': dataset['validation']}
    return result


def report(config, source, block, output):
    from .blocks import run_propagation
    from .block_exports import export_profiles
    from .materials import build_material_bank
    endpoint, solution, bar, artifact = load_block(source, block, config)
    with output_attempt(output, 'vlac report') as attempt:
        if block == 2:
            export_profiles(attempt, config, endpoint, solution)
        else:
            run_propagation(config, build_material_bank(config), endpoint, solution, bar, attempt,
                source=source_record(artifact), characterization=artifact.get('payload',{}).get('characterization'))
    return Path(output).resolve()
