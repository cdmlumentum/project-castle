"""Coupled-amplitude propagation, convergence and port metrics."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import PchipInterpolator

from .config import PropagationConfig
from .modes import Coefficients


@dataclass
class PropagationResult:
    output_amplitudes: np.ndarray
    output_powers: np.ndarray
    total_guided_power: float
    steps: int
    converged: bool
    relative_change: float


def _complex_pchip(x: np.ndarray, values: np.ndarray):
    values = np.asarray(values, dtype=complex)
    real = PchipInterpolator(x, values.real, extrapolate=False)
    imag = PchipInterpolator(x, values.imag, extrapolate=False)

    def evaluate(at):
        return real(at) + 1j * imag(at)

    return evaluate


def propagate_rk4(
    z_samples_m: np.ndarray,
    beta_upper: np.ndarray,
    beta_lower: np.ndarray,
    kappa: np.ndarray,
    *,
    launch_layer: int,
    steps: int,
) -> PropagationResult:
    """Integrate dA/dz = i [[beta1,kappa],[kappa,beta2]] A with fixed-step RK4.

    Positive imaginary beta gives attenuation under the exp(+i beta z) phase
    convention. Launch amplitudes are power-normalized to one in the selected
    basis mode. Cross-section absorption and other imaginary-neff losses are
    included in beta; radiation outside this two-mode basis is not reconstructed.
    """
    if launch_layer not in (0, 1):
        raise ValueError("launch_layer must be 0 (upper) or 1 (lower).")
    z_samples_m = np.asarray(z_samples_m, dtype=float)
    if len(z_samples_m) < 2 or np.any(np.diff(z_samples_m) <= 0):
        raise ValueError("z samples must be strictly increasing with at least 2 points.")
    b1 = _complex_pchip(z_samples_m, beta_upper)
    b2 = _complex_pchip(z_samples_m, beta_lower)
    kap = _complex_pchip(z_samples_m, kappa)
    z0, z1 = float(z_samples_m[0]), float(z_samples_m[-1])
    h = (z1 - z0) / steps
    reduced_amplitude = np.zeros(2, dtype=complex)
    reduced_amplitude[launch_layer] = 1.0

    def rhs(z, a):
        # Remove the common propagation phase/attenuation before RK4. This is
        # exact for the scalar mean beta and avoids resolving O(1e7 rad/m)
        # carrier phase to calculate power transfer.
        delta = (b1(z) - b2(z)) / 2
        matrix = np.array([[delta, kap(z)], [kap(z), -delta]], dtype=complex)
        return 1j * (matrix @ a)

    common_attenuation_integral = 0.0
    for i in range(steps):
        z = z0 + i * h
        z_mid = z + h / 2
        z_next = z0 + (i + 1) * h
        k1 = rhs(z, reduced_amplitude)
        k2 = rhs(z_mid, reduced_amplitude + h * k1 / 2)
        k3 = rhs(z_mid, reduced_amplitude + h * k2 / 2)
        k4 = rhs(z_next, reduced_amplitude + h * k3)
        reduced_amplitude = reduced_amplitude + h * (
            k1 + 2 * k2 + 2 * k3 + k4
        ) / 6
        beta_mean_im = ((b1(z) + b2(z)) / 2).imag
        beta_mid_im = ((b1(z_mid) + b2(z_mid)) / 2).imag
        beta_next_im = ((b1(z_next) + b2(z_next)) / 2).imag
        common_attenuation_integral += h * (
            beta_mean_im + 4 * beta_mid_im + beta_next_im
        ) / 6
    # The discarded common phase is irrelevant to port powers. Retain common
    # amplitude attenuation so absolute guided output includes material loss.
    amplitude = reduced_amplitude * np.exp(-common_attenuation_integral)
    powers = np.abs(amplitude) ** 2
    return PropagationResult(
        output_amplitudes=amplitude,
        output_powers=powers,
        total_guided_power=float(np.sum(powers)),
        steps=steps,
        converged=False,
        relative_change=float("nan"),
    )


def propagate_converged(
    z_samples_m: np.ndarray,
    beta_upper: np.ndarray,
    beta_lower: np.ndarray,
    kappa: np.ndarray,
    *,
    launch_layer: int,
    settings: PropagationConfig,
) -> PropagationResult:
    steps = settings.rk4_steps_initial
    previous = None
    while True:
        current = propagate_rk4(
            z_samples_m,
            beta_upper,
            beta_lower,
            kappa,
            launch_layer=launch_layer,
            steps=steps,
        )
        if previous is not None:
            denominator = np.maximum(
                np.maximum(current.output_powers, previous.output_powers),
                settings.power_convergence_rtol * 1e-3,
            )
            change = float(
                np.max(
                    np.abs(current.output_powers - previous.output_powers)
                    / denominator
                )
            )
            current.relative_change = change
            current.converged = change <= settings.power_convergence_rtol
            if current.converged or steps >= settings.rk4_steps_max:
                return current
        if steps >= settings.rk4_steps_max:
            current.converged = False
            return current
        previous = current
        steps = min(2 * steps, settings.rk4_steps_max)


def port_metrics(
    result: PropagationResult,
    *,
    state: str,
    launch_layer: int,
    xt_power_floor: float,
) -> dict:
    if state not in {"cross", "bar"}:
        raise ValueError("state must be 'cross' or 'bar'.")
    desired_layer = 1 - launch_layer if state == "cross" else launch_layer
    undesired_layer = 1 - desired_layer
    p_desired = float(result.output_powers[desired_layer])
    p_undesired = float(result.output_powers[undesired_layer])
    il = float("inf") if p_desired <= 0 else -10 * np.log10(p_desired)
    if p_desired <= 0:
        xt = float("nan")
    else:
        xt_ratio = max(p_undesired / p_desired, xt_power_floor)
        xt = 10 * np.log10(xt_ratio)
    return {
        "state": state,
        "launch_layer": "upper" if launch_layer == 0 else "lower",
        "desired_layer": "upper" if desired_layer == 0 else "lower",
        "undesired_layer": "upper" if undesired_layer == 0 else "lower",
        "desired_power": p_desired,
        "undesired_power": p_undesired,
        "guided_output_power": result.total_guided_power,
        "not_in_two_mode_guided_basis_power": max(0.0, 1 - result.total_guided_power),
        "il_db": float(il),
        "xt_db": float(xt),
        "integration_steps": result.steps,
        "integration_converged": result.converged,
        "integration_relative_change": result.relative_change,
        "xt_floor_power_ratio": xt_power_floor,
    }


def propagate_state(
    z_samples_m: np.ndarray,
    coefficients_by_wavelength: list[list[Coefficients]],
    wavelengths_m: np.ndarray,
    *,
    state: str,
    settings: PropagationConfig,
    progress=None,
) -> list[dict]:
    rows = []
    completed = 0
    if progress:
        progress.start_stage(
            f"{state.capitalize()}-state propagation",
            total=2 * len(wavelengths_m),
            unit="wavelength/launch cases",
        )
    for iw, wavelength in enumerate(wavelengths_m):
        coeff_row = coefficients_by_wavelength[iw]
        b1 = np.array([c.beta_upper for c in coeff_row], dtype=complex)
        b2 = np.array([c.beta_lower for c in coeff_row], dtype=complex)
        kappa = np.array([c.kappa for c in coeff_row], dtype=complex)
        for launch_layer in (0, 1):
            result = propagate_converged(
                z_samples_m,
                b1,
                b2,
                kappa,
                launch_layer=launch_layer,
                settings=settings,
            )
            row = port_metrics(
                result,
                state=state,
                launch_layer=launch_layer,
                xt_power_floor=settings.xt_power_floor,
            )
            row["wavelength_m"] = float(wavelength)
            row["wavelength_nm"] = float(wavelength * 1e9)
            rows.append(row)
            completed += 1
            if progress:
                progress.update(
                    completed,
                    detail=(
                        f"{state} {wavelength * 1e9:.1f} nm, "
                        f"launch {'upper' if launch_layer == 0 else 'lower'}"
                    ),
                )
    if progress:
        progress.finish_stage()
    return rows
