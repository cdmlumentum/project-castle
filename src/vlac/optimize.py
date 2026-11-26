"""Bounded derivative-free minimax optimization and grid refinement."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.interpolate import PchipInterpolator
from collections import OrderedDict

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


def _axis_validation_errors(base, checked, coordinates, refined, axis, delta_scale):
    """Compare fresh midpoint coefficients with interpolation on one axis."""
    error = 0.
    for attr, scale in (("delta_beta", delta_scale), ("kappa", .5*delta_scale)):
        values = np.asarray([[getattr(c, attr) for c in row] for row in base], dtype=complex)
        actual = np.asarray([[getattr(c, attr) for c in row] for row in checked], dtype=complex)
        predicted = (PchipInterpolator(coordinates, values.real, axis=axis)(refined)
                     + 1j*PchipInterpolator(coordinates, values.imag, axis=axis)(refined))
        error = max(error, float(np.max(np.abs(actual-predicted)))/max(scale,1e-30))
    if any(c.extraction_flag == "low_mode_overlap" for row in checked for c in row):
        error = max(error, 1.)
    return error


def optimize_profiles(backend, config: WorkflowConfig, bank: MaterialBank, endpoint: EndpointSearchResult, *, progress=None, checkpoint_dir=None):
    """Bounded Powell at the band edges, with a longitudinal FDE grid check."""
    g, d, o = config.geometry, config.device, config.optimizer
    ncp = o.control_points
    w0 = np.linspace(endpoint.w1_input_m, endpoint.w1_output_m, ncp)
    dc0 = np.linspace(g.d_c2c_input_m, g.d_c2c_output_m, ncp)
    lower = np.r_[np.full(ncp-2,g.w1_min_m),np.full(ncp-2,g.d_c2c_bounds_m[0])]
    upper = np.r_[np.full(ncp-2,g.w1_max_m),np.full(ncp-2,g.d_c2c_bounds_m[1])]
    initial = _normalized_controls(np.r_[w0[1:-1],dc0[1:-1]],lower,upper)
    z = np.linspace(0,d.length_m,d.z_samples_initial)
    wavelengths = np.array([d.wavelength_min_m, d.wavelength_max_m])
    history, grid_history = [], []
    converged = False
    evaluations_completed = 0
    checkpoint = None
    start_refinement = 0
    best = float('inf')
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
            if saved.get('converged_solution') is not None:
                return saved['converged_solution']
            if progress: progress.note('Resuming Powell from saved feasible controls; optimizer directions restart.', force=True)

    def save_checkpoint(controls, refinement, completed=None):
        if checkpoint:
            checkpoint.save({'controls': np.asarray(controls), 'z': z, 'refinement': refinement,
                'history': history, 'grid_history': grid_history, 'converged_solution': completed,
                'resume_method': 'Powell warm restart; not exact optimizer continuation'})
    if progress:
        progress.start_stage("Cross-state profile optimization", unit="objective evaluations")

    def unpack(unit):
        physical = lower + np.clip(np.asarray(unit),0,1)*(upper-lower)
        return (np.r_[endpoint.w1_input_m,physical[:ncp-2],endpoint.w1_output_m],
                np.r_[g.d_c2c_input_m,physical[ncp-2:],g.d_c2c_output_m])

    for refinement in range(start_refinement, max(start_refinement, o.max_grid_refinements)+1):
        memo = OrderedDict()
        evaluations = 0
        best = float("inf")
        def evaluate(unit):
            key = tuple(float(v) for v in unit)
            if key in memo:
                memo.move_to_end(key)
                return memo[key]
            w, dc = unpack(unit)
            penalty = _check_fabrication_constraints(config,ClampedCubicProfile(d.length_m,w),ClampedCubicProfile(d.length_m,dc))
            if penalty:
                return penalty,None,None,None
            coefficients = evaluate_design_profiles(backend,config,bank,endpoint,w,dc,z,wavelengths,state="cross")
            value,delta_target,kappa_target = minimax_objective(coefficients,z,d.length_m,endpoint.delta_beta_max_rad_per_m)
            memo[key] = value,coefficients,delta_target,kappa_target
            while len(memo)>4:
                memo.popitem(last=False)
            return memo[key]

        def objective(unit):
            nonlocal evaluations,evaluations_completed,best
            if hasattr(backend, '_budget'): backend._budget().check()
            evaluations += 1
            entry = evaluate(unit)
            value = entry[0]
            if entry[1] is not None and value < best:
                best = value
                save_checkpoint(unit, refinement)
            evaluations_completed += 1
            if progress:
                progress.update(evaluations_completed,detail=f"grid {refinement+1}, normalized Powell evaluation {evaluations}; J={value:.6g}")
            return value

        result = minimize(objective,initial,method="Powell",bounds=Bounds(np.zeros_like(initial),np.ones_like(initial)),
                          options={"maxfev":o.max_function_evaluations_per_grid,"xtol":o.minimum_control_step_fraction,
                                   "ftol":d.grid_convergence_rtol,"disp":False})
        final_w,final_dc = unpack(result.x)
        value,coefficients,delta_target,kappa_target = evaluate(result.x)
        if coefficients is None:
            raise RuntimeError("Profile optimizer returned controls violating the configured fabrication limits.")
        history.append({"grid_refinement":refinement,"n_z":len(z),"n_wavelengths":len(wavelengths),
                        "evaluations":evaluations,"objective":value,"success":bool(result.success),
                        "message":str(result.message),"control_w1_m":final_w.tolist(),"control_d_c2c_m":final_dc.tolist(),
                        "coordinate_system":"unit_interval", "normalized_controls":result.x.tolist()})
        if progress:
            progress.checkpoint(f"profile_optimization_grid_{refinement+1}",history[-1])
        errors, checked_counts = {}, {}
        # Check longitudinal midpoints at the two band edges only.
        # Unchanged cross-sections reuse raw FDE data with tracking reapplied.
        for name, axis, current, limit in (("z",1,z,d.z_samples_max),):
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
               "wavelength_coefficient_interpolation_error":None,
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
        if refinement==o.max_grid_refinements or not requests:
            row["stop_reason"] = "refinement_budget_or_axis_limit"
            break
        if "z" in requests:
            z = _grid_refine(z)
        initial = np.clip(result.x,0,1)
        save_checkpoint(initial, refinement+1)
    if progress:
        progress.finish_stage(f"J={value:.6g}; {evaluations_completed} objective evaluations; grid_converged={converged}")
    solution = ProfileSolution(final_w,final_dc,z,wavelengths,delta_target,kappa_target,coefficients,value,
                               history,grid_history,converged)
    save_checkpoint(result.x, refinement, solution if converged and result.success else None)
    return solution
