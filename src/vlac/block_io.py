"""Portable JSON handoffs, compatibility checks and v21 endpoint import.

Artifacts contain scalar coefficients, never pickles or mode-field arrays.
An atomic result file is the final commit of an exported block.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import config_from_dict
from .endpoint import EndpointSearchResult
from .endpoint_constraints import require_endpoint_constraints
from .modes import Coefficients
from .profiles import ProfileSolution, target_profiles

SCHEMA_VERSION = 2
MODEL = 'full_width_slabs_v21;negative_to_positive_cross_target'
COMPLEX_FIELDS = {'beta_upper','beta_lower','beta_plus','beta_minus','delta_beta','delta_pm','delta_12','kappa'}
OPTIONAL_FLOATS = {'kappa_profile_check','overlap_upper_lower'}


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, complex): return {'real': clean(value.real), 'imag': clean(value.imag)}
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value),indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def physics_signature(config, block):
    raw = config.as_dict()
    material = {k:v for k,v in raw['materials'].items() if not k.startswith('silicon_') and k!='sin_index_sensitivity_offsets'}
    for key in ('liquid_dispersion_csv','sin_extinction_csv'):
        path = config.resolved_path(material[key])
        material[key] = None if path is None else {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    solver_keys = ('number_of_trial_modes','mesh_accuracy','mesh_cells_x','mesh_cells_y',
                   'domain_margin_y_bottom_m','endpoint_compact_x','tracking_grid_nx','tracking_grid_ny',
                   'minimum_te_fraction','minimum_mode_overlap','field_overlap_crosscheck_max_basis_overlap')
    endpoint_keys = ('cross_magnitude_balance_rtol','pso_lower_mode_state_independent','pso_guided_neff_margin')
    signature = {'model': MODEL,'geometry': raw['geometry'],'materials': material,
                 'solver': {k:raw['solver'][k] for k in solver_keys},
                 'endpoint_constraints': {k:raw['endpoint_search'][k] for k in endpoint_keys},
                 'wavelength_bounds_m': [config.device.wavelength_min_m,config.device.wavelength_max_m]}
    if config.solver.calibration is not None:
        signature['block0_certificate_id'] = config.solver.calibration['certificate_id']
    if block>=2:
        signature['length_m'] = config.device.length_m
        signature['grid_convergence_rtol'] = config.device.grid_convergence_rtol
        if config.solver.tracking_method != 'legacy' or config.optimizer.method != 'bounded_powell':
            signature['profile_methods'] = {'tracking':config.solver.tracking_method,
                'recovery_depth':config.solver.tracking_recovery_depth,'optimizer':config.optimizer.method}
            if config.solver.endpoint_domain_policy != 'strict':
                signature['profile_methods']['endpoint_domain_policy'] = config.solver.endpoint_domain_policy
    if block>=4: signature['propagation'] = raw['propagation']
    return clean(signature)


def assert_compatible(saved, config, block):
    current = physics_signature(config,block)
    if saved!=current:
        changed = [key for key in current if saved.get(key)!=current[key]]
        raise ValueError('Saved block is incompatible with the requested configuration: '+', '.join(changed)+
                         '. Recompute the affected upstream block with matching settings.')


def endpoint_data(endpoint):
    return clean(asdict(endpoint))


def decode_endpoint(raw):
    values = dict(raw)
    values['wavelengths_m'] = np.asarray(values['wavelengths_m'],dtype=float)
    return EndpointSearchResult(**values)


def coefficient_data(grid):
    rows = []
    for row in grid:
        encoded = []
        for coefficient in row:
            values = asdict(coefficient)
            for key in COMPLEX_FIELDS: values[key] = complex(values[key])
            encoded.append(clean(values))
        rows.append(encoded)
    return rows


def solution_data(solution):
    values = asdict(solution)
    values['cross_coefficients'] = coefficient_data(solution.cross_coefficients)
    return clean(values)


def decode_coefficients(raw):
    result = []
    expected = {f.name for f in fields(Coefficients)}
    for row in raw:
        decoded = []
        for entry in row:
            if set(entry)!=expected: raise ValueError('Saved coefficient fields do not match the supported schema.')
            values = dict(entry)
            for key in COMPLEX_FIELDS:
                value = values[key]
                if not isinstance(value,dict) or set(value)!={'real','imag'}:
                    raise ValueError(f'Malformed complex coefficient {key}.')
                if value['real'] is None or value['imag'] is None:
                    raise ValueError(f'Non-finite core coefficient {key}.')
                values[key] = complex(value['real'],value['imag'])
            for key in OPTIONAL_FLOATS:
                if values[key] is None: values[key] = float('nan')
            values['supermode_identities'] = tuple(values['supermode_identities'])
            decoded.append(Coefficients(**values))
        result.append(decoded)
    return result


def decode_solution(raw):
    values = dict(raw)
    for key in ('w1_control_m','d_c2c_control_m','z_m','wavelengths_m','delta_target_rad_per_m','kappa_target_rad_per_m'):
        values[key] = np.asarray(values[key],dtype=float)
    values['cross_coefficients'] = decode_coefficients(values['cross_coefficients'])
    return ProfileSolution(**values)


def validate_endpoint(endpoint, config):
    rows = endpoint.endpoint_records
    wavelengths = [config.device.wavelength_min_m,config.device.wavelength_max_m]
    required = {(end,state,w) for end in ('input','output') for state in ('cross','bar') for w in wavelengths}
    if len(rows)!=8 or {(r['end'],r['state'],float(r['wavelength_m'])) for r in rows}!=required:
        raise ValueError('Saved endpoint must contain eight unique matching state/end/band-edge cases.')
    if not np.array_equal(endpoint.wavelengths_m,wavelengths): raise ValueError('Saved endpoint wavelength array is inconsistent.')
    g = config.geometry
    for end,width in (('input',endpoint.w1_input_m),('output',endpoint.w1_output_m)):
        if not np.isfinite(width) or not g.w1_min_m<=width<=g.w1_max_m: raise ValueError('Saved endpoint width is out of bounds.')
        for row in [r for r in rows if r['end']==end]:
            values = [float(row[key]) for key in ('beta_upper_real_rad_per_m','beta_lower_real_rad_per_m','delta_beta_real_rad_per_m')]
            if not np.all(np.isfinite(values)) or not np.isclose(values[0]-values[1],values[2],rtol=1e-8,atol=1e-3):
                raise ValueError('Saved endpoint betas and detuning disagree.')
            for key,expected in (('w1_m',width),('h_rib_m',endpoint.h_rib_m),('h_gap_m',endpoint.h_gap_m),
                                  ('d_c2c_m',g.d_c2c_input_m if end=='input' else g.d_c2c_output_m)):
                if not np.isclose(float(row[key]),expected,rtol=0,atol=1e-15): raise ValueError(f'Saved endpoint {key} is inconsistent.')
            for key in ('beta_upper_imag_rad_per_m','beta_lower_imag_rad_per_m','delta_beta_imag_rad_per_m'):
                if not np.isfinite(row[key]): raise ValueError('Saved endpoint has non-finite imaginary beta data.')
            for key in ('upper_tracking_overlap','lower_tracking_overlap'):
                if not np.isfinite(row[key]) or row[key]<config.solver.minimum_mode_overlap:
                    raise ValueError('Saved endpoint tracking overlap fails the configured threshold.')
            for key in ('upper_te_fraction','lower_te_fraction'):
                if not np.isfinite(row[key]) or row[key]<config.solver.minimum_te_fraction: raise ValueError('Saved endpoint TE fraction fails the configured threshold.')
    if not g.h_rib_bounds_m[0]<=endpoint.h_rib_m<=g.h_rib_bounds_m[1] or not g.h_gap_bounds_m[0]<=endpoint.h_gap_m<=g.h_gap_bounds_m[1]:
        raise ValueError('Saved endpoint height/gap is incompatible.')
    objective = min(abs(r['delta_beta_real_rad_per_m']) for r in rows)
    if not np.isfinite(endpoint.delta_beta_max_rad_per_m) or objective<=0 or not np.isclose(objective,endpoint.delta_beta_max_rad_per_m,rtol=1e-6,atol=1e-3):
        raise ValueError('Saved endpoint objective does not match the eight verified cases.')
    return require_endpoint_constraints(rows,config.endpoint_search)


def validate_grid(grid, solution, config):
    if len(grid)!=len(solution.wavelengths_m) or any(len(row)!=len(solution.z_m) for row in grid):
        raise ValueError('Saved coefficient grid dimensions do not match its coordinates.')
    flags, low = {}, 0
    for row in grid:
        for c in row:
            for key in COMPLEX_FIELDS:
                value = complex(getattr(c,key))
                if not np.isfinite(value.real) or not np.isfinite(value.imag): raise ValueError('Non-finite core coefficient '+key)
            if not np.isfinite(c.min_mode_overlap) or not 0<=c.min_mode_overlap<=1: raise ValueError('Invalid tracking overlap.')
            if not np.isclose(c.delta_beta,c.beta_upper-c.beta_lower,rtol=1e-6,atol=1e-3): raise ValueError('Inconsistent saved delta_beta.')
            if not np.isclose(c.delta_12,c.delta_beta/2,rtol=1e-6,atol=1e-3): raise ValueError('Inconsistent saved delta_12.')
            if not np.isclose(c.delta_pm,(c.beta_plus-c.beta_minus)/2,rtol=1e-6,atol=1e-3): raise ValueError('Inconsistent saved delta_pm.')
            if not np.isclose(c.kappa**2,c.delta_pm**2-c.delta_12**2,rtol=1e-5,atol=1e-2): raise ValueError('Inconsistent saved kappa extraction.')
            flags[c.extraction_flag] = flags.get(c.extraction_flag,0)+1
            low += c.min_mode_overlap<config.solver.minimum_mode_overlap
    return {'passed': not low and all(k=='ok' for k in flags), 'low_overlap_cases': int(low),'extraction_flags': flags}


def validate_solution(solution, endpoint, config):
    for values, minimum in ((solution.z_m,3),(solution.wavelengths_m,2)):
        if values.ndim!=1 or len(values)<minimum or not np.all(np.isfinite(values)) or np.any(np.diff(values)<=0):
            raise ValueError(f'Saved grid coordinates must be finite, increasing 1D arrays with at least {minimum} points.')
    if not np.allclose(solution.z_m[[0,-1]],[0,config.device.length_m],rtol=0,atol=1e-15): raise ValueError('Saved z grid does not cover the configured device length.')
    if not np.allclose(solution.wavelengths_m[[0,-1]],endpoint.wavelengths_m,rtol=0,atol=1e-15): raise ValueError('Saved wavelength grid does not cover the band.')
    g = config.geometry
    for values,bounds,ends in ((solution.w1_control_m,(g.w1_min_m,g.w1_max_m),(endpoint.w1_input_m,endpoint.w1_output_m)),
                              (solution.d_c2c_control_m,g.d_c2c_bounds_m,(g.d_c2c_input_m,g.d_c2c_output_m))):
        if values.ndim!=1 or len(values)<4 or not np.all(np.isfinite(values)) or np.any(values<bounds[0]-1e-15) or np.any(values>bounds[1]+1e-15):
            raise ValueError('Saved profile controls are invalid or outside bounds.')
        if not np.allclose(values[[0,-1]],ends,rtol=0,atol=1e-15): raise ValueError('Saved profile endpoint controls changed.')
        if config.optimizer.method in {'monotone_powell','sequential_width_offset_v1'}:
            from .monotone import is_monotone
            if not is_monotone(values):
                raise ValueError('Saved profile violates monotone control ordering.')
    expected = target_profiles(solution.z_m,config.device.length_m,endpoint.delta_beta_max_rad_per_m)
    for actual,target in zip((solution.delta_target_rad_per_m,solution.kappa_target_rad_per_m),expected):
        if actual.shape!=target.shape or not np.allclose(actual,target,rtol=1e-10,atol=1e-6): raise ValueError('Saved coefficient targets are inconsistent.')
    if not np.isfinite(solution.objective) or solution.objective<0: raise ValueError('Saved profile objective is invalid.')
    from .profiles import minimax_objective
    from .optimize import _check_fabrication_constraints
    from .geometry import ClampedCubicProfile
    actual_objective = minimax_objective(solution.cross_coefficients,solution.z_m,config.device.length_m,endpoint.delta_beta_max_rad_per_m)[0]
    if not np.isclose(solution.objective,actual_objective,rtol=1e-9,atol=1e-7):
        raise ValueError('Saved objective does not match its coefficient grid.')
    if _check_fabrication_constraints(config,ClampedCubicProfile(config.device.length_m,solution.w1_control_m),
                                     ClampedCubicProfile(config.device.length_m,solution.d_c2c_control_m)):
        raise ValueError('Saved profiles violate the current slope/curvature constraints.')
    quality = validate_grid(solution.cross_coefficients,solution,config)
    if config.solver.tracking_method == 'guarded_translation_v1':
        from .tracking import endpoint_consistency
        quality['endpoint_mode_consistency'] = endpoint_consistency(solution.cross_coefficients,solution.wavelengths_m,endpoint,config)
        quality['passed'] = quality['passed'] and quality['endpoint_mode_consistency']['passed']
    return quality


def write_artifact(output, block, config, endpoint, *, solution=None, bar=None, performance=None, validation=None, source=None, characterization=None):
    validate_endpoint(endpoint,config)
    payload = {'endpoint': endpoint_data(endpoint)}
    if solution is not None:
        validate_solution(solution,endpoint,config)
        payload['solution'] = solution_data(solution)
    if bar is not None:
        validate_grid(bar,solution,config)
        payload['bar_coefficients'] = coefficient_data(bar)
    if performance is not None: payload['performance'] = clean(performance)
    if solution is not None:
        payload['geometry'] = geometry_data(config, endpoint, solution)
    if characterization is not None:
        from .characterize import validate_characterization
        validate_characterization(characterization, config, endpoint, solution)
        payload['characterization'] = characterization
    from .provenance import artifact_identity
    from .runtime import overall_status
    validation = dict(validation or {})
    validation['overall_status'] = overall_status(validation)
    artifact = {'schema_version':SCHEMA_VERSION,'block':block,'status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
                'physics_signature':physics_signature(config,block),'config':config.as_dict(),
                'payload':payload,'validation':validation or {},'source':source or {}}
    artifact['provenance'] = artifact_identity(config)
    artifact['content_id'] = digest({'physics': artifact['physics_signature'], 'payload': payload})
    artifact['sha256'] = digest(artifact)
    write_json(Path(output)/f'block{block}_result.json',artifact)
    return artifact


def _read_artifact(path, block, config):
    artifact = json.loads(path.read_text(encoding='utf-8'))
    if artifact.get('schema_version') not in {1,SCHEMA_VERSION} or artifact.get('block')!=block or artifact.get('status')!='complete':
        raise ValueError('Unsupported, incomplete, or wrong-block artifact: '+str(path))
    expected = artifact.pop('sha256',None)
    if expected!=digest(artifact): raise ValueError('Saved artifact checksum mismatch: '+str(path))
    artifact['sha256'] = expected
    if artifact['schema_version'] >= 2:
        content = digest({'physics': artifact['physics_signature'], 'payload': artifact['payload']})
        if content != artifact.get('content_id'): raise ValueError('Saved artifact content identity mismatch')
    assert_compatible(artifact['physics_signature'],config,block)
    endpoint = decode_endpoint(artifact['payload']['endpoint'])
    validate_endpoint(endpoint,config)
    solution = None
    bar = None
    if block>=2:
        solution = decode_solution(artifact['payload']['solution'])
        validate_solution(solution,endpoint,config)
        if artifact['schema_version'] >= 2 and artifact['payload'].get('geometry') != geometry_data(config, endpoint, solution):
            raise ValueError('Saved frozen geometry disagrees with optimized controls')
    if block>=3:
        bar = decode_coefficients(artifact['payload']['bar_coefficients'])
        validate_grid(bar,solution,config)
    if 'characterization' in artifact['payload']:
        from .characterize import validate_characterization
        validate_characterization(artifact['payload']['characterization'], config, endpoint, solution)
    return endpoint,solution,bar,artifact


def load_block(directory, block, config):
    from .runtime import resolve_published
    path = resolve_published(directory)
    if path.is_file():
        if path.name!=f'block{block}_result.json': raise ValueError('Expected a block result directory or matching block result JSON.')
    else: path = path/f'block{block}_result.json'
    if path.exists(): return _read_artifact(path,block,config)
    if block!=1: raise ValueError(f'Missing block{block}_result.json in {path.parent}; run the preceding block first.')
    return load_v21_endpoint(path.parent,config)


def load_v21_endpoint(directory, config):
    """Import the complete existing v21 endpoint export without running MODE."""
    folder = Path(directory)
    raw_config = json.loads((folder/'configuration.json').read_text(encoding='utf-8'))
    if 'rib_slab_overhang_m' in raw_config.get('geometry',{}):
        raise ValueError('Finite-slab endpoint results predate v21 and cannot be reused with full-width slabs.')
    saved_config = config_from_dict(raw_config,project_root=config.project_root)
    assert_compatible(physics_signature(saved_config,1),config,1)
    summary = json.loads((folder/'endpoint_summary.json').read_text(encoding='utf-8'))
    if summary.get('status')!='passed': raise ValueError('Saved endpoint summary is not passed.')
    method = summary.get('search_method')
    if method not in {'adaptive_1d','python_pso'}: raise ValueError('Legacy import requires a v21 adaptive/PSO fresh-verification report.')
    filename = 'endpoint_adaptive_verification.json' if method=='adaptive_1d' else 'endpoint_pso_verification.json'
    verification = json.loads((folder/filename).read_text(encoding='utf-8'))
    if verification.get('status')!='passed' or not verification.get('fresh_fde_solves'):
        raise ValueError('Saved endpoint fresh verification did not pass.')
    with (folder/'endpoint_detunings.csv').open(newline='',encoding='utf-8-sig') as stream:
        rows = [{k:(v if k in {'end','state'} or k.endswith('_identity') else float(v)) for k,v in row.items()}
                for row in csv.DictReader(stream)]
    search = summary.get('search_summary') or summary.get('pso_summary') or {}
    endpoint = EndpointSearchResult(h_gap_m=summary['h_gap_m'],h_rib_m=summary['h_rib_m'],
        w1_input_m=summary['w1_input_m'],w1_output_m=summary['w1_output_m'],delta_beta_max_rad_per_m=summary['delta_beta_max_rad_per_m'],
        wavelengths_m=np.asarray(summary['wavelengths_m']),endpoint_records=rows,candidate_records=[],
        converged=bool(search.get('converged',False)),refinement_history=[],isolated_mode_domain=summary['isolated_mode_domain'],
        search_method=method,pso_summary=summary.get('pso_summary'),search_summary=summary.get('search_summary'))
    validate_endpoint(endpoint,config)
    expected = {(r['end'],r['state'],float(r['wavelength_m'])):r for r in verification.get('endpoint_records',[])}
    if len(expected)!=8: raise ValueError('Fresh verification must contain eight distinct endpoint records.')
    for row in rows:
        verified = expected[(row['end'],row['state'],row['wavelength_m'])]
        for key in ('w1_m','beta_upper_real_rad_per_m','beta_lower_real_rad_per_m','delta_beta_real_rad_per_m'):
            if not np.isclose(row[key],verified[key],rtol=1e-10,atol=1e-12 if key=='w1_m' else 1e-3):
                raise ValueError('Endpoint CSV disagrees with its fresh-verification record.')
    source = {'format':'v21_endpoint_export','path':str(folder),'legacy_import':True,
              'note':'Legacy export has no material-file hashes; current supplied material tables are assumed unchanged.',
              'sha256':digest({'summary':summary,'records':rows,'verification':verification,'config':raw_config}),
              'validation':{'status':'passed','endpoint_constraints':require_endpoint_constraints(rows,config.endpoint_search)}}
    return endpoint,None,None,source


def source_record(artifact):
    return {'block':artifact.get('block',1),'sha256':artifact['sha256'],
            'content_id':artifact.get('content_id'), 'provenance':artifact.get('provenance'),
            'validation':artifact.get('validation',{}),'format':artifact.get('format','vlac_block_json'),
            'backend_name':artifact.get('source',{}).get('backend_name','saved_fde_coefficients')}


def geometry_data(config, endpoint, solution):
    """Frozen controls are independent of optimization/characterization grids."""
    value = {'schema_version': 1, 'units': 'SI', 'length_m': config.device.length_m,
             'w1_control_m': clean(solution.w1_control_m), 'd_c2c_control_m': clean(solution.d_c2c_control_m),
             'h_gap_m': endpoint.h_gap_m, 'h_rib_m': endpoint.h_rib_m,
             'geometry': clean(config.as_dict()['geometry'])}
    return dict(value, geometry_id=digest(value))
