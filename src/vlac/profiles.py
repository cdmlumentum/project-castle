"""Spline-profile evaluation, minimax objective and grid refinement."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import WorkflowConfig
from .endpoint import EndpointSearchResult
from .geometry import profile_from_values, blackman_theta, make_geometry
from .materials import MaterialBank
from .modes import Coefficients, ModeReference


@dataclass
class ProfileSolution:
    w1_control_m: np.ndarray
    d_c2c_control_m: np.ndarray
    z_m: np.ndarray
    wavelengths_m: np.ndarray
    delta_target_rad_per_m: np.ndarray
    kappa_target_rad_per_m: np.ndarray
    cross_coefficients: list[list[Coefficients]]
    objective: float
    optimizer_history: list[dict]
    grid_convergence_history: list[dict]
    grid_converged: bool
    profile_kind: str = "clamped_cubic_v1"
    anchor_z_m: np.ndarray | None = None
    sweep_report: dict | None = None


def target_profiles(z_m: np.ndarray, length_m: float, delta_beta_max: float):
    theta = blackman_theta(z_m, length_m)
    delta = -delta_beta_max * np.cos(theta)
    kappa = 0.5 * delta_beta_max * np.sin(theta)
    return delta, kappa


def _profile_coefficients(
    backend,
    config: WorkflowConfig,
    bank: MaterialBank,
    endpoint: EndpointSearchResult,
    w_control_m: np.ndarray,
    d_control_m: np.ndarray,
    z_m: np.ndarray,
    wavelengths_m: np.ndarray,
    *,
    state: str,
    progress=None,
    anchor_z_m=None,
) -> list[list[Coefficients]]:
    g, d = config.geometry, config.device
    guarded = config.solver.tracking_method == 'guarded_translation_v1'
    w_profile = profile_from_values(d.length_m, w_control_m, anchor_z_m)
    dc_profile = profile_from_values(d.length_m, d_control_m, anchor_z_m)
    liquid = bank.liquid_cross if state == "cross" else bank.liquid_bar
    all_coeffs: list[list[Coefficients]] = []
    previous_lambda_references: list[ModeReference | None] = [None] * len(z_m)
    completed = 0
    if progress:
        progress.start_stage(
            f"{state.capitalize()}-state coefficient extraction",
            total=len(z_m) * len(wavelengths_m),
            unit="z-wavelength samples",
        )
    for wavelength_index, wavelength in enumerate(wavelengths_m):
        row: list[Coefficients] = []
        current_lambda_references: list[ModeReference | None] = []
        previous_z_reference: ModeReference | None = None
        previous_z_kappa: complex | None = None
        for iz, z in enumerate(z_m):
            def sample_at(position, last_position, last_reference, last_kappa, depth, lambda_reference=None):
                from .tracking import CandidateRejected, require_candidate
                local_geometry = make_geometry(g,w1_m=float(w_profile(position)),h_rib_m=endpoint.h_rib_m,
                    h_gap_m=endpoint.h_gap_m,d_c2c_m=float(dc_profile(position)),state=state,
                    liquid_index=float(liquid.index(wavelength)))
                local_refs = tuple(r for r in (last_reference,lambda_reference) if r is not None)
                try:
                    value, ref = backend.evaluate_coefficients(local_geometry,float(wavelength),
                        references=local_refs,previous_kappa=last_kappa)
                    if guarded:
                        require_candidate(value,config)
                        if not ref.valid:
                            raise CandidateRejected('Backend returned a rejected reference.')
                    return value,ref
                except CandidateRejected as exc:
                    # The rejected reference is never advanced. Bridge only from
                    # the last accepted point; recovery does not relax tolerances.
                    if not guarded or last_reference is None or depth<=0 or last_position==position:
                        raise type(exc)(f'z={position*1e6:.6g} um, wavelength={wavelength*1e9:.6g} nm: {exc}') from exc
                    midpoint = (last_position+position)/2
                    middle, middle_ref = sample_at(midpoint,last_position,last_reference,last_kappa,depth-1)
                    if progress:
                        progress.note(f'Tracking recovery inserted z={midpoint*1e6:.6g} um')
                    return sample_at(position,midpoint,middle_ref,middle.kappa,depth-1,lambda_reference)

            coefficients, reference = sample_at(float(z),float(z_m[iz-1]) if iz else float(z),
                previous_z_reference,previous_z_kappa,config.solver.tracking_recovery_depth,
                previous_lambda_references[iz])
            row.append(coefficients)
            completed += 1
            if progress:
                progress.update(
                    completed,
                    detail=f"{state} {wavelength * 1e9:.1f} nm, z={z * 1e6:.1f} um",
                )
            current_lambda_references.append(reference)
            previous_z_reference = reference
            previous_z_kappa = coefficients.kappa
        all_coeffs.append(row)
        previous_lambda_references = current_lambda_references
    if guarded:
        from .tracking import endpoint_consistency, enforce_endpoint_report
        check = endpoint_consistency(all_coeffs,wavelengths_m,endpoint,config,state=state)
        enforce_endpoint_report(check,config)
    if progress:
        progress.finish_stage()
    return all_coeffs


def minimax_objective(
    coefficients: list[list[Coefficients]],
    z_m: np.ndarray,
    length_m: float,
    delta_beta_max: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    if delta_beta_max <= 0:
        raise ValueError("Delta_beta_max must be positive before profile optimization.")
    delta_target, kappa_target = target_profiles(z_m, length_m, delta_beta_max)
    delta_errors, kappa_errors = [], []
    for row in coefficients:
        delta_errors.append(
            [
                abs(value.delta_beta.real - delta_target[j]) / delta_beta_max
                for j, value in enumerate(row)
            ]
        )
        kappa_errors.append(
            [
                abs(value.kappa - kappa_target[j]) / (0.5 * delta_beta_max)
                for j, value in enumerate(row)
            ]
        )
    objective = float(max(np.max(delta_errors), np.max(kappa_errors)))
    return objective, delta_target, kappa_target


def evaluate_design_profiles(
    backend,
    config: WorkflowConfig,
    bank: MaterialBank,
    endpoint: EndpointSearchResult,
    w_control_m: np.ndarray,
    d_control_m: np.ndarray,
    z_m: np.ndarray,
    wavelengths_m: np.ndarray,
    *,
    state: str = "cross",
    progress=None,
    anchor_z_m=None,
) -> list[list[Coefficients]]:
    if hasattr(backend, "evaluate_profile_grid"):
        return backend.evaluate_profile_grid(
            config, bank, endpoint, w_control_m, d_control_m, z_m, wavelengths_m,
            state=state, progress=progress,
            **({"anchor_z_m": anchor_z_m} if anchor_z_m is not None else {}),
        )
    return _profile_coefficients(
        backend,
        config,
        bank,
        endpoint,
        w_control_m,
        d_control_m,
        z_m,
        wavelengths_m,
        state=state,
        progress=progress,
        anchor_z_m=anchor_z_m,
    )
