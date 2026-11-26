"""Spline-profile evaluation, minimax objective and grid refinement."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import WorkflowConfig
from .endpoint import EndpointSearchResult
from .geometry import ClampedCubicProfile, blackman_theta, make_geometry
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
) -> list[list[Coefficients]]:
    g, d, m = config.geometry, config.device, config.materials
    w_profile = ClampedCubicProfile(d.length_m, w_control_m)
    dc_profile = ClampedCubicProfile(d.length_m, d_control_m)
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
            geometry = make_geometry(
                g,
                w1_m=float(w_profile(z)),
                h_rib_m=endpoint.h_rib_m,
                h_gap_m=endpoint.h_gap_m,
                d_c2c_m=float(dc_profile(z)),
                state=state,
                liquid_index=float(liquid.index(wavelength)),
            )
            refs = []
            if previous_z_reference is not None:
                refs.append(previous_z_reference)
            if previous_lambda_references[iz] is not None:
                refs.append(previous_lambda_references[iz])
            coefficients, reference = backend.evaluate_coefficients(
                geometry,
                float(wavelength),
                references=tuple(refs),
                previous_kappa=previous_z_kappa,
            )
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
) -> list[list[Coefficients]]:
    if hasattr(backend, "evaluate_profile_grid"):
        return backend.evaluate_profile_grid(
            config, bank, endpoint, w_control_m, d_control_m, z_m, wavelengths_m,
            state=state, progress=progress,
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
    )
