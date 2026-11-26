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
    launch_layer: int | None,
    steps: int,
) -> PropagationResult:
    """Integrate dA/dz = i [[beta1,kappa],[kappa,beta2]] A with fixed-step RK4.

    Positive imaginary beta gives attenuation under the exp(+i beta z) phase
    convention. Launch amplitudes are power-normalized to one in the selected
    basis mode. Cross-section absorption and other imaginary-neff losses are
    included in beta; radiation outside this two-mode basis is not reconstructed.
    """
    if launch_layer not in (None, 0, 1):
        raise ValueError("launch_layer must be 0 (upper) or 1 (lower).")
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    z_samples_m = np.asarray(z_samples_m, dtype=float)
    if len(z_samples_m) < 2 or np.any(np.diff(z_samples_m) <= 0):
        raise ValueError("z samples must be strictly increasing with at least 2 points.")
    b1 = _complex_pchip(z_samples_m, beta_upper)
    b2 = _complex_pchip(z_samples_m, beta_lower)
    kap = _complex_pchip(z_samples_m, kappa)
    z0, z1 = float(z_samples_m[0]), float(z_samples_m[-1])
    h = (z1 - z0) / steps
    reduced_amplitude = np.eye(2, dtype=complex) if launch_layer is None else np.eye(2, dtype=complex)[:,launch_layer].copy()

    # Vectorize interpolation over the RK4 half-step lattice once per solve.
    lattice = np.linspace(z0, z1, 2*steps+1)
    upper_values, lower_values, coupling = b1(lattice), b2(lattice), kap(lattice)
    detuning = (upper_values-lower_values)/2
    mean_imag = ((upper_values+lower_values)/2).imag
    matrices = np.empty((len(lattice), 2, 2), dtype=complex)
    matrices[:,0,0], matrices[:,1,1] = detuning, -detuning
    matrices[:,0,1] = matrices[:,1,0] = coupling
    matrices *= 1j
    common_attenuation_integral = 0.0
    for i in range(steps):
        left, mid, right = matrices[2*i:2*i+3]
        k1 = left @ reduced_amplitude
        k2 = mid @ (reduced_amplitude+h*k1/2)
        k3 = mid @ (reduced_amplitude+h*k2/2)
        k4 = right @ (reduced_amplitude+h*k3)
        reduced_amplitude += h*(k1+2*k2+2*k3+k4)/6
        common_attenuation_integral += h*(mean_imag[2*i]+4*mean_imag[2*i+1]+mean_imag[2*i+2])/6
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


def propagate_both(z, upper, lower, kappa, settings):
    """Share interpolation and RK4 matrices, retaining each launch's stopping step."""
    steps, previous, results = settings.rk4_steps_initial, None, [None, None]
    while True:
        pair = propagate_rk4(z, upper, lower, kappa, launch_layer=None, steps=steps)
        if previous is not None:
            denom = np.maximum(np.maximum(pair.output_powers, previous.output_powers), settings.power_convergence_rtol*1e-3)
            changes = np.max(np.abs(pair.output_powers-previous.output_powers)/denom, axis=0)
        else:
            changes = np.full(2, np.inf)
        for launch in (0, 1):
            converged = changes[launch] <= settings.power_convergence_rtol
            if results[launch] is None and (converged or steps >= settings.rk4_steps_max):
                results[launch] = PropagationResult(pair.output_amplitudes[:,launch], pair.output_powers[:,launch],
                    float(sum(pair.output_powers[:,launch])), steps, bool(converged), float(changes[launch]))
        if all(value is not None for value in results): return results
        previous = pair
        steps = min(2*steps, settings.rk4_steps_max)


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
        for launch_layer, result in enumerate(propagate_both(z_samples_m, b1, b2, kappa, settings)):
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
