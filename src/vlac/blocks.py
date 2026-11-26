"""Execute and export individual blocks, reusing portable upstream results."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import numpy as np

from .block_io import (load_block, source_record, validate_endpoint, validate_grid,
                       validate_solution, write_artifact, write_json)
from .block_exports import export_profiles
from .export import export_run
from .materials import build_material_bank
from .optimize import optimize_profiles, _axis_validation_errors, _grid_refine
from .profiles import evaluate_design_profiles
from .propagate import propagate_state


def invalidate(output, first_block):
    """Remove only published downstream results from a reused output folder."""
    out = Path(output)
    for block in range(first_block,5):
        for suffix in ('result','summary','validation_report','failure'):
            (out/f'block{block}_{suffix}.json').unlink(missing_ok=True)
    for name in ('performance_by_launch.csv','performance_summary.csv','wavelength_performance.png','metadata.json'):
        (out/name).unlink(missing_ok=True)
    if first_block<=3:
        for name in ('bar_coefficients.csv','coefficients.csv','profiles_and_coefficients.png'):
            (out/name).unlink(missing_ok=True)
    if first_block<=2:
        for name in ('cross_coefficients.csv','coefficients.csv','optimized_profiles.csv','optimizer_history.json','profiles_and_coefficients.png'):
            (out/name).unlink(missing_ok=True)
    if first_block<=1:
        from .endpoint_diagnostics import FILES
        for name in FILES:
            (out/name).unlink(missing_ok=True)
        for name in ('endpoint_summary.json','endpoint_detunings.csv','endpoint_adaptive_best.json','endpoint_pso_best.json'):
            (out/name).unlink(missing_ok=True)
    (out/'run_failure.json').unlink(missing_ok=True)


def export_endpoint(output,config,endpoint,*,backend_name,source=None,backend=None,mode_profiles=None,inspection_project=None):
    from .export import _write_csv
    out = Path(output)
    out.mkdir(parents=True,exist_ok=True)
    constraints = validate_endpoint(endpoint,config)
    from .block0 import require_design_verification
    calibration_check = require_design_verification(config, out, endpoint, backend=backend)
    report = {'block':1,'status':'passed','execution_completed':True,'search_converged':endpoint.converged,
              'scope':'endpoint constraints and completed search verification; no global optimality claim',
              'endpoint_constraint_verification':constraints}
    if calibration_check is not None:
        report['block0_verification'] = calibration_check
    from .endpoint_diagnostics import export_endpoint_diagnostics
    report['diagnostics'] = export_endpoint_diagnostics(out, config, endpoint, backend=backend,
        profiles=mode_profiles, inspection_project=inspection_project)
    write_json(out/'configuration.json',config.as_dict())
    _write_csv(out/'endpoint_detunings.csv',list(endpoint.endpoint_records[0]),endpoint.endpoint_records)
    summary = {'status':'passed','scope':'simulation block 1 only','search_method':endpoint.search_method,
               'h_gap_m':endpoint.h_gap_m,'h_rib_m':endpoint.h_rib_m,'w1_input_m':endpoint.w1_input_m,
               'w1_output_m':endpoint.w1_output_m,'delta_beta_max_rad_per_m':endpoint.delta_beta_max_rad_per_m,
               'wavelengths_m':endpoint.wavelengths_m.tolist(),'isolated_mode_domain':endpoint.isolated_mode_domain,
               'search_summary':endpoint.search_summary,'pso_summary':endpoint.pso_summary,
               'cross_state_sign_verification':constraints['cross_state_sign_verification'],
               'endpoint_constraint_verification':constraints,'converged':endpoint.converged}
    summary['diagnostics'] = report['diagnostics']
    write_json(out/'endpoint_summary.json',summary)
    write_json(out/'block1_summary.json',report)
    return write_artifact(out,1,config,endpoint,validation=report,source=dict(source or {},backend_name=backend_name))


def profile_report(solution,endpoint,config):
    quality = validate_solution(solution,endpoint,config)
    optimization = bool(solution.optimizer_history and solution.optimizer_history[-1]['success'])
    passed = optimization and solution.grid_converged and quality['passed']
    constraints = validate_endpoint(endpoint,config)
    edges_only = len(solution.wavelengths_m)==2
    return {'block':2,'status':'passed' if passed else 'review','execution_completed':True,'usable':True,
            'optimization_converged':optimization,'grid_converged':bool(solution.grid_converged),
            'coefficient_quality':quality,'objective_achieved':solution.objective,
            'target_fit_acceptance_threshold':None,
            'scope':('cross-state profile optimization and z coefficient interpolation checks at the band edges'
                     if edges_only else 'cross-state profile optimization on the saved wavelength grid'),
            'wavelength_validation':('band edges only; interior wavelengths not checked in Block 2'
                                     if edges_only else 'see saved grid history for wavelength checks'),
            'grid_tolerance':config.device.grid_convergence_rtol,'grid_history':solution.grid_convergence_history,
            'fixed_endpoint_widths_m':[endpoint.w1_input_m,endpoint.w1_output_m],
            'endpoint_common_target_checks':constraints['block2_common_target_checks'],
            'note':'Grid convergence covers z sampling at the saved wavelengths; it does not establish interior-wavelength accuracy, a small target-fit objective or full-device performance.'}


def run_profile(backend,config,bank,endpoint,output,*,progress=None,source=None):
    solution = optimize_profiles(backend,config,bank,endpoint,progress=progress)
    report = profile_report(solution,endpoint,config)
    report['upstream_validation'] = (source or {}).get('validation',{})
    from .block0 import require_design_verification
    calibration_check = require_design_verification(config, output, endpoint, solution, backend=backend)
    if calibration_check is not None:
        report['block0_verification'] = calibration_check
    export_profiles(output,config,endpoint,solution)
    write_json(Path(output)/'configuration.json',config.as_dict())
    write_json(Path(output)/'block2_summary.json',report)
    artifact = write_artifact(output,2,config,endpoint,solution=solution,validation=report,source=source)
    if progress: progress.checkpoint('cross_state_optimization',report)
    return solution,artifact


def check_bar_grid(backend,config,bank,endpoint,solution,bar,*,progress=None):
    errors = {}
    for name,axis,current,limit in (('z',1,solution.z_m,config.device.z_samples_max),
                                    ('wavelength',0,solution.wavelengths_m,config.device.wavelength_samples_max)):
        refined = _grid_refine(current)
        if len(refined)>limit:
            errors[name] = None
            continue
        if progress:
            progress.start_stage(f'Bar-state {name} grid validation',unit='fixed-profile FDE checks')
        checked = evaluate_design_profiles(backend,config,bank,endpoint,solution.w1_control_m,solution.d_c2c_control_m,
            refined if name=='z' else solution.z_m,refined if name=='wavelength' else solution.wavelengths_m,state='bar')
        errors[name] = _axis_validation_errors(bar,checked,current,refined,axis,endpoint.delta_beta_max_rad_per_m)
        quality = validate_grid(checked,replace(solution,
            z_m=refined if name=='z' else solution.z_m,
            wavelengths_m=refined if name=='wavelength' else solution.wavelengths_m),config)
        if not quality['passed']: errors[name] = max(errors[name],1.)
        if progress: progress.finish_stage(f'normalized interpolation error={errors[name]:.6g}')
    return {'passed':all(v is not None and v<=config.device.grid_convergence_rtol for v in errors.values()),
            'z_coefficient_interpolation_error':errors['z'],'wavelength_coefficient_interpolation_error':errors['wavelength'],
            'tolerance':config.device.grid_convergence_rtol,
            'scope':'fixed saved geometry; no profile optimization or endpoint search',
            'note':'A failed or unavailable check needs a denser coefficient grid; targets are not imposed on the bar state.'}


def run_bar(backend,config,bank,endpoint,solution,output,*,progress=None,source=None):
    bar = evaluate_design_profiles(backend,config,bank,endpoint,solution.w1_control_m,solution.d_c2c_control_m,
                                  solution.z_m,solution.wavelengths_m,state='bar',progress=progress)
    quality = validate_grid(bar,solution,config)
    checked = check_bar_grid(backend,config,bank,endpoint,solution,bar,progress=progress)
    from .block0 import require_design_verification
    calibration_check = require_design_verification(config, output, endpoint, solution, backend=backend, bar=bar)
    report = {'block':3,'status':'passed' if quality['passed'] and checked['passed'] else 'review',
              'execution_completed':True,'usable':True,'coefficient_quality':quality,'grid_validation':checked,
              'upstream_validation':(source or {}).get('validation',{}),
              'profile_optimized_this_block':False,'scope':'bar-state coefficients for the saved Block 2 geometry'}
    if calibration_check is not None:
        report['block0_verification'] = calibration_check
    export_profiles(output,config,endpoint,solution,bar)
    write_json(Path(output)/'configuration.json',config.as_dict())
    write_json(Path(output)/'block3_summary.json',report)
    artifact = write_artifact(output,3,config,endpoint,solution=solution,bar=bar,validation=report,source=source)
    if progress: progress.checkpoint('bar_state_coefficients',report)
    return bar,artifact


def run_propagation(config,bank,endpoint,solution,bar,output,*,progress=None,source=None,backend_name='saved_fde_coefficients'):
    performance = []
    for state,grid in (('cross',solution.cross_coefficients),('bar',bar)):
        performance.extend(propagate_state(solution.z_m,grid,solution.wavelengths_m,state=state,
                                            settings=config.propagation,progress=progress))
    finite = all(np.all(np.isfinite([r['desired_power'],r['undesired_power'],r['guided_output_power']])) for r in performance)
    converged = all(r['integration_converged'] for r in performance)
    significant_gain = any(r['guided_output_power']>1+max(1e-6,10*config.propagation.power_convergence_rtol) for r in performance)
    if not finite: raise ValueError('Propagation produced non-finite output powers; increase integration resolution or review coefficients.')
    report = {'block':4,'status':'passed' if converged and not significant_gain else 'review','execution_completed':True,
              'integration_converged':converged,'finite_output_powers':finite,'significant_guided_power_above_one':significant_gain,
              'performance_cases':len(performance),'mode_required_for_block4':False,
              'upstream_validation':(source or {}).get('validation',{}),
              'scope':'two-mode RK4 step convergence for both states, all saved wavelengths and both launches',
              'nonfinite_metric_encoding':'JSON null represents an undefined or infinite IL/XT metric; CSV retains nan/inf.',
              'full_device_validation':False}
    if progress: progress.start_stage('Export',total=1,unit='output bundles')
    export_run(output,config,bank,endpoint,solution,bar,performance,backend_name=backend_name,fde_run_completed=True)
    write_json(Path(output)/'block4_summary.json',report)
    artifact = write_artifact(output,4,config,endpoint,solution=solution,bar=bar,performance=performance,validation=report,source=source)
    if progress:
        progress.update(1,detail=str(output))
        progress.finish_stage()
        progress.checkpoint('propagation',report)
    return artifact


def run_single_block(block,config,output,source_dir,*,backend_factory,progress,validation_only=False):
    """Run only this block; all input validation precedes a MODE launch."""
    out = Path(output)
    out.mkdir(parents=True,exist_ok=True)
    invalidate(out,block)
    endpoint,solution,bar,source = load_block(source_dir,block-1,config)
    write_json(out/'configuration.json',config.as_dict())
    bank = build_material_bank(config)
    source = source_record(source)
    progress.checkpoint('loaded_upstream_block',source)
    if source.get('validation',{}).get('status')=='review':
        progress.note('Loaded upstream result needs validation review; its status is retained in this block report.',force=True)
    if block==4:
        artifact = run_propagation(config,bank,endpoint,solution,bar,out,progress=progress,source=source,
                                   backend_name=source.get('backend_name','saved_fde_coefficients'))
    else:
        progress.start_stage('MODE API/license preflight',total=1,unit='startup checks')
        with backend_factory(config,cache_path=config.resolved_path(config.output.cache_sqlite),progress=progress) as backend:
            backend.preflight(.5*(config.device.wavelength_min_m+config.device.wavelength_max_m))
            progress.update(1)
            progress.finish_stage()
            source['backend_name'] = backend.backend_name
            if block==2:
                _,artifact = run_profile(backend,config,bank,endpoint,out,progress=progress,source=source)
            else:
                _,artifact = run_bar(backend,config,bank,endpoint,solution,out,progress=progress,source=source)
    report = artifact['validation']
    metadata_path = out/'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    metadata.update(command=progress.command,blocks_executed_this_command=[block],mode_launched_this_command=block in (2,3),
                    loaded_upstream=source,block_validation=report,config=config.as_dict())
    write_json(metadata_path,metadata)
    if validation_only: write_json(out/f'block{block}_validation_report.json',report)
    progress.finish()
    return artifact
