"""Bounded derivative-free minimax optimization and grid refinement."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.interpolate import PchipInterpolator
from collections import OrderedDict
import time
from .profile_planning import ProfilePlanner, FinalizeProfile

from .config import WorkflowConfig
from .endpoint import EndpointSearchResult
from .geometry import ClampedCubicProfile
from .materials import MaterialBank
from .profiles import (
    ProfileSolution,
    evaluate_design_profiles,
    minimax_objective,
)


def _grid_refine(values: np.ndarray) -> np.ndarray:
    return np.linspace(float(values[0]), float(values[-1]), 2 * len(values) - 1)


def _check_fabrication_constraints(
    config: WorkflowConfig, w: ClampedCubicProfile, d: ClampedCubicProfile
) -> float:
    limit_slope = config.optimizer.maximum_slope_m_per_m
    limit_curvature = config.optimizer.maximum_curvature_per_m
    if limit_slope is None and limit_curvature is None:
        return 0.0
    dense_z = np.linspace(0, config.device.length_m, 401)
    penalty = 0.0
    for profile in (w, d):
        if limit_slope is not None:
            violation = np.maximum(np.abs(profile.derivative(dense_z)) - limit_slope, 0)
            if np.any(violation):
                penalty += 1e3 + float(np.max(violation) / max(limit_slope, 1e-30))
        if limit_curvature is not None:
            violation = np.maximum(
                np.abs(profile.derivative(dense_z, order=2)) - limit_curvature, 0
            )
            if np.any(violation):
                penalty += 1e3 + float(
                    np.max(violation) / max(limit_curvature, 1e-30)
                )
    return penalty


def _normalized_controls(physical, lower, upper):
    return np.clip((np.asarray(physical)-lower)/(upper-lower), 0., 1.)


def _axis_validation_errors(base, checked, coordinates, refined, axis, delta_scale, kappa_scale=None):
    """Compare fresh midpoint coefficients with interpolation on one axis."""
    error = 0.
    for attr, scale in (("delta_beta", delta_scale), ("kappa", .5*delta_scale if kappa_scale is None else kappa_scale)):
        values = np.asarray([[getattr(c, attr) for c in row] for row in base], dtype=complex)
        actual = np.asarray([[getattr(c, attr) for c in row] for row in checked], dtype=complex)
        predicted = (PchipInterpolator(coordinates, values.real, axis=axis)(refined)
                     + 1j*PchipInterpolator(coordinates, values.imag, axis=axis)(refined))
        error = max(error, float(np.max(np.abs(actual-predicted)))/max(scale,1e-30))
    if any(c.extraction_flag == "low_mode_overlap" for row in checked for c in row):
        error = max(error, 1.)
    return error


def optimize_profiles(backend, config: WorkflowConfig, bank: MaterialBank, endpoint: EndpointSearchResult, *, progress=None, checkpoint_dir=None):
    """Joint or sequential Powell, with fresh coefficient midpoint checks."""
    if config.optimizer.method == 'tabulated_width_offset_v1':
        from .tabulated import optimize_tabulated
        return optimize_tabulated(backend,config,bank,endpoint,progress=progress,checkpoint_dir=checkpoint_dir)
    g, d, o = config.geometry, config.device, config.optimizer
    sequential = o.method == 'sequential_width_offset_v1'
    if config.solver.tracking_method == 'guarded_translation_v1':
        # A fixed-endpoint inconsistency cannot be repaired by interior controls.
        # Diagnose it once, before spending the optimization allowance.
        backend.require_endpoint_consistency(endpoint,progress=progress)
    planner = ProfilePlanner(o.complete_within_seconds, started=getattr(progress, 'started', None))
    ncp = o.control_points
    w0 = np.linspace(endpoint.w1_input_m, endpoint.w1_output_m, ncp)
    dc0 = np.linspace(g.d_c2c_input_m, g.d_c2c_output_m, ncp)
    lower = np.r_[np.full(ncp-2,g.w1_min_m),np.full(ncp-2,g.d_c2c_bounds_m[0])]
    upper = np.r_[np.full(ncp-2,g.w1_max_m),np.full(ncp-2,g.d_c2c_bounds_m[1])]
    initial = _normalized_controls(np.r_[w0[1:-1],dc0[1:-1]],lower,upper)
    if o.method == 'monotone_powell' or sequential:
        from .monotone import linear_coordinates
        initial = linear_coordinates(ncp) if sequential else np.tile(linear_coordinates(ncp),2)
    z = np.linspace(0,d.length_m,d.z_samples_initial)
    wavelengths = np.array([d.wavelength_min_m, d.wavelength_max_m])
    history, grid_history = [], []
    converged = False
    evaluations_completed = 0
    checkpoint = None
    start_refinement = 0
    best = float('inf')
    width_state = None

    def validation_count():
        return (2*len(z)-1)*len(wavelengths)
    if checkpoint_dir is not None:
        from .checkpoints import Checkpoint
        from .provenance import checkpoint_identity
        from .block_io import endpoint_data
        from dataclasses import asdict
        settings = asdict(o)
        settings.pop('max_function_evaluations_per_grid'); settings.pop('max_grid_refinements')
        identity = checkpoint_identity(config, 2, {'endpoint': endpoint_data(endpoint), 'optimizer': settings,
            'initial_z': d.z_samples_initial, 'maximum_z': d.z_samples_max})
        checkpoint = Checkpoint(checkpoint_dir, 'powell', identity)
        saved = checkpoint.load() if config.execution.resume else None
        if saved:
            initial, z = saved['controls'], saved['z']
            start_refinement = saved['refinement']
            history, grid_history = saved['history'], saved['grid_history']
            width_state = saved.get('width_state')
            wavelengths = saved.get('wavelengths', wavelengths)
            if not np.array_equal(wavelengths,[d.wavelength_min_m,d.wavelength_max_m]):
                raise ValueError('Block 2 cannot resume an interior-wavelength grid.')
            if saved.get('converged_solution') is not None:
                return saved['converged_solution']
            if progress: progress.note('Resuming Powell from saved feasible controls; optimizer directions restart.', force=True)

    def save_checkpoint(controls, refinement, completed=None):
        if checkpoint:
            checkpoint.save({'controls': np.asarray(controls), 'z': z, 'refinement': refinement,
                'history': history, 'grid_history': grid_history, 'converged_solution': completed,
                'width_state':width_state, 'wavelengths':wavelengths,
                'resume_method': 'Powell warm restart; not exact optimizer continuation'})
    if progress:
        progress.start_stage("Cross-state profile optimization", unit="objective evaluations")

    def unpack(unit):
        if o.method == 'monotone_powell' or sequential:
            from .monotone import controls
            unit = np.clip(np.asarray(unit),0,1)
            if sequential:
                return (np.asarray(width_state['controls']),
                        controls(unit,g.d_c2c_input_m,g.d_c2c_output_m))
            return (controls(unit[:ncp-2],endpoint.w1_input_m,endpoint.w1_output_m),
                    controls(unit[ncp-2:],g.d_c2c_input_m,g.d_c2c_output_m))
        physical = lower + np.clip(np.asarray(unit),0,1)*(upper-lower)
        return (np.r_[endpoint.w1_input_m,physical[:ncp-2],endpoint.w1_output_m],
                np.r_[g.d_c2c_input_m,physical[ncp-2:],g.d_c2c_output_m])

    for refinement in range(start_refinement, max(start_refinement, o.max_grid_refinements)+1):
        if sequential and (width_state is None or width_state['refinement'] != refinement):
            from .sequential import fit_width_profile
            from .monotone import linear_coordinates
            width_initial = (linear_coordinates(ncp) if width_state is None else
                             np.asarray(width_state['report']['normalized_controls']))
            width_controls, width_report = fit_width_profile(backend,config,bank,endpoint,z,wavelengths,
                width_initial,max_evaluations=o.max_function_evaluations_per_grid//2,
                planner=planner,validation_samples=validation_count(),progress=progress)
            width_state = {'refinement':refinement,'controls':width_controls,'report':width_report}
            evaluations_completed += width_report['evaluations']
            save_checkpoint(initial, refinement)
            if progress:
                progress.checkpoint(f'sequential_width_grid_{refinement+1}',width_report)
        memo = OrderedDict()
        evaluations = 0
        best = float("inf")
        best_entry = None
        best_unit = None
        def evaluate(unit):
            key = tuple(float(v) for v in unit)
            if key in memo:
                memo.move_to_end(key)
                return memo[key]
            w, dc = unpack(unit)
            penalty = _check_fabrication_constraints(config,ClampedCubicProfile(d.length_m,w),ClampedCubicProfile(d.length_m,dc))
            if penalty:
                return penalty,None,None,None
            sample_started = time.monotonic()
            from .tracking import CandidateRejected, EndpointConsistencyError
            try:
                coefficients = evaluate_design_profiles(backend,config,bank,endpoint,w,dc,z,wavelengths,state="cross")
            except EndpointConsistencyError:
                raise
            except CandidateRejected as exc:
                if progress:
                    progress.note('Candidate rejected before objective/reference reuse: '+str(exc),force=True)
                    progress.checkpoint('rejected_profile_candidate',{'w_control_m':w.tolist(),
                        'd_control_m':dc.tolist(),'reason':str(exc)})
                planner.observe(time.monotonic()-sample_started,len(z)*len(wavelengths))
                return 1e6,None,None,None
            planner.observe(time.monotonic()-sample_started, len(z)*len(wavelengths))
            value,delta_target,kappa_target = minimax_objective(coefficients,z,d.length_m,endpoint.delta_beta_max_rad_per_m)
            if sequential:
                # Width is frozen. Fit full complex coupling only; the published
                # objective below is still recomputed from BOTH residuals.
                value = float(np.max(np.abs(np.asarray([[c.kappa for c in row] for row in coefficients])
                    -kappa_target[None,:]))/(.5*endpoint.delta_beta_max_rad_per_m))
            memo[key] = value,coefficients,delta_target,kappa_target
            while len(memo)>4:
                memo.popitem(last=False)
            return memo[key]

        def objective(unit):
            nonlocal evaluations,evaluations_completed,best,best_entry,best_unit
            if best_entry is not None and not planner.allows(len(z)*len(wavelengths), validation_count()):
                raise FinalizeProfile()
            if hasattr(backend, '_budget'): backend._budget().check()
            evaluations += 1
            entry = evaluate(unit)
            value = entry[0]
            if entry[1] is not None and value < best:
                best = value
                best_entry, best_unit = entry, np.asarray(unit).copy()
                save_checkpoint(unit, refinement)
                if progress and o.complete_within_seconds is not None:
                    from .block_io import write_json, solution_data
                    candidate_w, candidate_dc = unpack(unit)
                    combined = minimax_objective(entry[1],z,d.length_m,endpoint.delta_beta_max_rad_per_m)[0]
                    candidate = ProfileSolution(candidate_w,candidate_dc,z,wavelengths,entry[2],entry[3],entry[1],
                        combined,history,grid_history,False)
                    write_json(progress.output_dir/'block2_best_candidate.json',
                        {'status':'candidate_only', 'execution_completed':False,
                         'solution':solution_data(candidate), 'planning':planner.report(len(z)*len(wavelengths),validation_count())})
            evaluations_completed += 1
            if progress:
                label = 'offset coupling' if sequential else 'normalized Powell'
                progress.update(evaluations_completed,detail=f"grid {refinement+1}, {label} evaluation {evaluations}; J={value:.6g}")
                if o.complete_within_seconds is not None:
                    progress.checkpoint('profile_work_plan',planner.report(len(z)*len(wavelengths),validation_count()))
            return value

        planned_finish = False
        try:
            result = minimize(objective,initial,method="Powell",bounds=Bounds(np.zeros_like(initial),np.ones_like(initial)),
                              options={"maxfev":(o.max_function_evaluations_per_grid-o.max_function_evaluations_per_grid//2
                                                  if sequential else o.max_function_evaluations_per_grid),"xtol":o.minimum_control_step_fraction,
                                       "ftol":d.grid_convergence_rtol,"disp":False})
        except FinalizeProfile:
            from types import SimpleNamespace
            planned_finish = True
            result = SimpleNamespace(x=best_unit, success=False,
                message='Optimization allowance complete; reserved time used for validation and exports.')
            memo[tuple(float(v) for v in best_unit)] = best_entry
        if sequential:
            if best_entry is None:
                raise RuntimeError('No optically valid offset candidate; rejected extractions were not fitted.')
            if not np.array_equal(result.x,best_unit) and not np.isclose(
                    getattr(result,'fun',float('inf')),best_entry[0],rtol=1e-12,atol=1e-14):
                result.success = False
                result.message = 'Returned best valid offset candidate from the completed allowance.'
            result.x = best_unit
            value,coefficients,delta_target,kappa_target = best_entry
        else:
            value,coefficients,delta_target,kappa_target = evaluate(result.x)
        final_w,final_dc = unpack(result.x)
        if coefficients is None:
            if best_entry is None:
                raise RuntimeError('No optically valid profile candidate was found; rejected extractions were not fitted.')
            result.x = best_unit
            value,coefficients,delta_target,kappa_target = best_entry
            final_w,final_dc = unpack(best_unit)
            result.success = False
            result.message = 'Returned best valid candidate after optimizer ended on a rejected candidate.'
        offset_objective = value
        if sequential:
            value = minimax_objective(coefficients,z,d.length_m,endpoint.delta_beta_max_rad_per_m)[0]
            planned_finish = planned_finish or width_state['report']['planned_finalization']
        history.append({"grid_refinement":refinement,"n_z":len(z),"n_wavelengths":len(wavelengths),
                        "evaluations":evaluations,"objective":value,"success":bool(result.success),
                        "message":str(result.message),"control_w1_m":final_w.tolist(),"control_d_c2c_m":final_dc.tolist(),
                        "coordinate_system":("monotone_stick_breaking_v1" if o.method=='monotone_powell' or sequential else "unit_interval"),
                        "method":o.method,"normalized_controls":result.x.tolist()})
        if sequential:
            history[-1].update(width_stage=width_state['report'], offset_objective=offset_objective,
                offset_success=bool(result.success),width_frozen_during_offset_fit=True,
                offset_evaluations=evaluations,
                evaluations=evaluations+width_state['report']['evaluations'],
                success=bool(result.success and width_state['report']['success']))
        if o.complete_within_seconds is not None:
            history[-1]['work_plan'] = planner.report(len(z)*len(wavelengths),validation_count())
            history[-1]['planned_finalization'] = planned_finish
        if progress:
            progress.checkpoint(f"profile_optimization_grid_{refinement+1}",history[-1])
        errors, checked_counts = {}, {}
        # Blocks 2/3 use only the two endpoint wavelengths. Longitudinal
        # midpoint cross-sections reuse exact raw solves with tracking reapplied.
        axes = [("z",1,z,d.z_samples_max)]
        for name, axis, current, limit in axes:
            refined = _grid_refine(current)
            if len(refined)>limit:
                errors[name] = None
                checked_counts[name] = 0
                continue
            check_z = refined if name == "z" else z
            check_wavelengths = refined if name == "wavelength" else wavelengths
            checked = evaluate_design_profiles(backend,config,bank,endpoint,final_w,final_dc,
                                               check_z,check_wavelengths,state="cross")
            errors[name] = _axis_validation_errors(coefficients,checked,current,refined,axis,
                                                    endpoint.delta_beta_max_rad_per_m)
            checked_counts[name] = len(refined)
        requests = [name for name,error in errors.items() if error is not None and error>d.grid_convergence_rtol]
        converged = errors["z"] is not None and errors["z"]<=d.grid_convergence_rtol
        row = {"grid_refinement":refinement,"n_z":len(z),"n_wavelengths":len(wavelengths),
               "z_coefficient_interpolation_error":errors["z"],
               "wavelength_coefficient_interpolation_error":errors.get('wavelength'),
               "wavelength_check":"band_edges_only_no_interior_check",
               "validation_axis_counts":checked_counts,"requested_refinement_axes":requests,
               "grid_converged":converged,"tolerance":d.grid_convergence_rtol}
        grid_history.append(row)
        if progress:
            progress.checkpoint(f"profile_grid_validation_{refinement+1}",row)
        save_checkpoint(result.x, refinement)
        if converged:
            row["stop_reason"] = "longitudinal_check_passed"
            break
        if planned_finish:
            row['stop_reason'] = 'completed_within_time_plan'
            break
        if refinement==o.max_grid_refinements or not requests:
            row["stop_reason"] = "refinement_budget_or_axis_limit"
            break
        # Refining commits to a larger optimization grid AND its midpoint check.
        # Keep the completed, checked grid if that work will consume finalization time.
        next_samples = (2*len(z)-1)*len(wavelengths)
        next_validation = (4*len(z)-3)*len(wavelengths)
        if not planner.allows(next_samples, next_validation):
            row['stop_reason'] = 'completed_within_time_plan_refinement_deferred'
            break
        if "z" in requests:
            z = _grid_refine(z)
        initial = np.clip(result.x,0,1)
        save_checkpoint(initial, refinement+1)
    if progress:
        progress.finish_stage(f"J={value:.6g}; {evaluations_completed} objective evaluations; grid_converged={converged}")
    solution = ProfileSolution(final_w,final_dc,z,wavelengths,delta_target,kappa_target,coefficients,value,
                               history,grid_history,converged)
    save_checkpoint(result.x, refinement, solution if converged and history[-1]['success'] else None)
    return solution
