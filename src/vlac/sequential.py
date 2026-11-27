"""Isolated-mode width stage of the sequential_width_offset_v1 alternative.

The two stages fit one longitudinal spline at a time. They are not two scalar
root solves: each spline retains n_control-2 monotone coordinates so smoothness,
fixed endpoints and optional fabrication constraints are enforced throughout.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import Bounds, minimize

from .geometry import ClampedCubicProfile, make_geometry
from .monotone import controls
from .profiles import target_profiles
from .profile_planning import FinalizeProfile
from .tracking import CandidateRejected, EndpointConsistencyError, require_anchor_match

METHOD = 'sequential_width_offset_v1'


def isolated_detuning_grid(backend, config, bank, endpoint, width_controls, z, wavelengths, *, width_samples=None):
    """Track isolated modes on the full production domain, with upper at x=0.

    No coupled solves or compact-domain substitutions are made in this stage.
    Each trial starts a new reference chain; rejected modes never advance it.
    Lower modes are invariant with upper width in the current material map and
    are solved once per wavelength. Native raw caching handles further reuse.
    """
    profile = (ClampedCubicProfile(config.device.length_m, width_controls) if width_samples is None
               else lambda position: np.interp(position, z, width_samples))
    previous_lambda = [None] * len(z)
    grid = []
    for wavelength in wavelengths:
        wavelength = float(wavelength)

        def geometry(width):
            return make_geometry(config.geometry, w1_m=float(width), h_rib_m=endpoint.h_rib_m,
                h_gap_m=endpoint.h_gap_m, d_c2c_m=0., state='cross',
                liquid_index=float(bank.liquid_cross.index(wavelength)))

        def solve(width, guide, reference):
            mode, overlap = backend.solve_isolated_mode(geometry(width), wavelength,
                guide=guide, reference=reference, compact_x=False)
            if (not np.isfinite(mode.neff) or not np.isfinite(overlap)
                    or overlap < config.solver.minimum_mode_overlap
                    or mode.te_fraction < config.solver.minimum_te_fraction
                    or not np.all(np.isfinite(mode.ex)) or not np.all(np.isfinite(mode.ey))
                    or np.sum(abs(mode.ex)**2+abs(mode.ey)**2) <= 0):
                raise CandidateRejected('Invalid isolated '+guide+' mode in width stage.')
            return mode

        lower = solve(float(profile(z[0])), 'lower', None)

        def tracked(width, previous_width, reference, depth):
            try:
                return solve(width, 'upper', reference)
            except CandidateRejected:
                if reference is None or depth <= 0 or width == previous_width:
                    raise
                middle = (width+previous_width)/2
                middle_mode = tracked(middle, previous_width, reference, depth-1)
                return tracked(width, middle, middle_mode, depth-1)

        row, modes = [], []
        previous, previous_width = None, None
        for index, position in enumerate(z):
            width = float(profile(position))
            reference = previous if previous is not None else previous_lambda[index]
            upper = tracked(width, previous_width if previous is not None else width,
                            reference, config.solver.tracking_recovery_depth)
            # The full-domain endpoint anchors were independently verified in
            # the mandatory preflight. Translate fields, never propagation constants.
            if index in (0, len(z)-1):
                offset = config.geometry.d_c2c_input_m if index == 0 else config.geometry.d_c2c_output_m
                anchor = getattr(backend, '_endpoint_full_modes', {}).get(
                    (wavelength, width, offset, geometry(width).liquid_index))
                if anchor is not None:
                    require_anchor_match(upper, anchor[0], config)
                    require_anchor_match(lower, anchor[1], config)
            row.append(2*np.pi*(upper.neff-lower.neff)/wavelength)
            modes.append(upper)
            previous, previous_width = upper, width
        previous_lambda = modes
        grid.append(row)
    return np.asarray(grid, dtype=complex)


def fit_width_profile(backend, config, bank, endpoint, z, wavelengths, initial, *,
                      max_evaluations, planner, validation_samples, progress=None):
    """Fit only detuning; preserve a best valid width when the allowance ends."""
    from .optimize import _check_fabrication_constraints

    d, g = config.device, config.geometry
    target = target_profiles(z, d.length_m, endpoint.delta_beta_max_rad_per_m)[0]
    scale = endpoint.delta_beta_max_rad_per_m
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Sequential detuning scale must be finite and positive.')
    fixed_offset = ClampedCubicProfile(d.length_m,
        np.full(config.optimizer.control_points, g.d_c2c_input_m))
    evaluations, rejected = 0, 0
    best = None
    # Only a few tiny scalar/grid results are retained; raw fields stay in the
    # existing checksummed backend cache and are never fabricated/interpolated.
    memo = {}

    def objective(unit):
        nonlocal evaluations, rejected, best
        key = tuple(float(v) for v in unit)
        if best is not None and not planner.allows(len(z)*len(wavelengths), validation_samples):
            raise FinalizeProfile()
        if hasattr(backend, '_budget'):
            backend._budget().check()
        evaluations += 1
        if key in memo:
            return memo[key]
        width = controls(np.clip(unit, 0, 1), endpoint.w1_input_m, endpoint.w1_output_m)
        penalty = _check_fabrication_constraints(config, ClampedCubicProfile(d.length_m, width), fixed_offset)
        if penalty:
            return penalty
        started = time.monotonic()
        try:
            delta = isolated_detuning_grid(backend, config, bank, endpoint, width, z, wavelengths)
        except EndpointConsistencyError:
            raise
        except CandidateRejected as exc:
            rejected += 1
            if progress:
                progress.note('Sequential width candidate rejected: '+str(exc), force=True)
            return 1e6
        finally:
            # Before coupled costs are measured reserve conservatively for the
            # upper/lower/coupled packets still required in the offset stage.
            planner.observe(3*(time.monotonic()-started), len(z)*len(wavelengths))
        residual = delta.real-target[None, :]
        value = float(np.max(np.abs(residual))/scale)
        if best is None or value < best[0]:
            best = value, np.asarray(unit).copy(), delta
        memo[key] = value
        if len(memo) > 8:
            memo.pop(next(iter(memo)))
        if progress:
            progress.note(f'Sequential width evaluation {evaluations}: normalized detuning error={value:.6g}')
        return value

    planned_finish = False
    try:
        result = minimize(objective, initial, method='Powell',
            bounds=Bounds(np.zeros_like(initial), np.ones_like(initial)),
            options={'maxfev':max_evaluations, 'xtol':config.optimizer.minimum_control_step_fraction,
                     'ftol':d.grid_convergence_rtol, 'disp':False})
    except FinalizeProfile:
        from types import SimpleNamespace
        planned_finish = True
        result = SimpleNamespace(success=False, message='Width allowance ended to reserve offset/validation/export work.')
    if best is None:
        raise RuntimeError('No valid isolated-mode width candidate; offset stage was not started.')
    value, unit, delta = best
    success = bool(result.success and (np.array_equal(result.x, unit) or
        np.isclose(getattr(result,'fun',float('inf')),value,rtol=1e-12,atol=1e-14))) if not planned_finish else False
    residual = delta.real-target[None, :]
    report = {'stage':'isolated_width', 'method':METHOD, 'evaluations':evaluations,
        'evaluation_limit':max_evaluations, 'rejected_candidates':rejected,
        'success':success, 'message':str(result.message),
        'planned_finalization':planned_finish, 'objective':value,
        'normalized_controls':unit.tolist(), 'domain':'full_production_domain_upper_centered_at_x_zero',
        'coupled_solves_for_width_fit':0, 'target_fit_acceptance_threshold':None,
        'endpoint_range_diagnostics':[{'wavelength_m':float(wavelength),
            'isolated_endpoint_delta_range_rad_per_m':[float(min(values.real[[0,-1]])),float(max(values.real[[0,-1]]))],
            'target_range_rad_per_m':[float(min(target)),float(max(target))],
            'endpoints_bracket_target_range':bool(min(values.real[[0,-1]])<=min(target)
                                                 and max(values.real[[0,-1]])>=max(target)),
            'sampled_delta_monotone':bool(np.all(np.diff(values.real)>=0) or np.all(np.diff(values.real)<=0))}
            for wavelength, values in zip(wavelengths,delta)],
        'errors_by_wavelength':[{'wavelength_m':float(wavelength),
            'max_abs_error_rad_per_m':float(np.max(abs(errors))),
            'rms_error_rad_per_m':float(np.sqrt(np.mean(errors**2)))}
            for wavelength, errors in zip(wavelengths, residual)]}
    return controls(unit, endpoint.w1_input_m, endpoint.w1_output_m), report
