"""Dispersive material models used by the workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np

from .config import WorkflowConfig

C0 = 299_792_458.0


class MaterialModel:
    name: str
    wavelength_range_m: tuple[float, float]

    def index(self, wavelength_m: float | np.ndarray) -> complex | np.ndarray:
        raise NotImplementedError

    def validate_range(self, wavelengths_m: np.ndarray) -> None:
        lo, hi = self.wavelength_range_m
        if np.min(wavelengths_m) < lo or np.max(wavelengths_m) > hi:
            raise ValueError(
                f"{self.name} is only supported from {lo*1e9:.1f} to "
                f"{hi*1e9:.1f} nm; requested "
                f"{np.min(wavelengths_m)*1e9:.1f}–{np.max(wavelengths_m)*1e9:.1f} nm."
            )


@dataclass
class MalitsonSilica(MaterialModel):
    """Malitson fused-silica Sellmeier model (lambda in micrometres)."""

    name: str = "SiO2_Malitson_1965"
    wavelength_range_m: tuple[float, float] = (0.21e-6, 3.71e-6)

    def index(self, wavelength_m: float | np.ndarray) -> complex | np.ndarray:
        wavelength_m = np.asarray(wavelength_m, dtype=float)
        self.validate_range(wavelength_m)
        lam_um = wavelength_m * 1e6
        lam2 = lam_um**2
        b = np.array([0.6961663, 0.4079426, 0.8974794])
        c2 = np.array([0.0684043**2, 0.1162414**2, 9.896161**2])
        n2 = 1.0 + np.sum(
            b.reshape((-1,) + (1,) * lam2.ndim)
            * lam2
            / (lam2 - c2.reshape((-1,) + (1,) * lam2.ndim)),
            axis=0,
        )
        result = np.sqrt(n2).astype(complex)
        return complex(result) if result.ndim == 0 else result


@dataclass
class LukeSiliconNitride(MaterialModel):
    """Luke et al. 2015 LPCVD SiN Sellmeier model with an optional index shift.

    The paper's fit is a provisional nominal material model. It is not treated as
    process-matched to the user's film. The absolute index offset is an explicit
    sensitivity input. By default the model has k=0 because no process-specific
    extinction data were supplied; an optional CSV can provide wavelength_um,k.
    """

    absolute_index_offset: float = 0.0
    extinction_csv: Path | None = None
    name: str = "Si3N4_Luke_2015_provisional"
    wavelength_range_m: tuple[float, float] = (0.310e-6, 5.504e-6)

    def _n(self, wavelength_m: np.ndarray) -> np.ndarray:
        lam_um = wavelength_m * 1e6
        lam2 = lam_um**2
        # Luke et al. 2015; wavelength is in micrometres.
        n2_minus_1 = (
            3.0249 * lam2 / (lam2 - 0.1353406**2)
            + 40314.0 * lam2 / (lam2 - 1239.842**2)
        )
        return np.sqrt(1.0 + n2_minus_1) + self.absolute_index_offset

    def _k(self, wavelength_m: np.ndarray) -> np.ndarray:
        if self.extinction_csv is None:
            return np.zeros_like(wavelength_m, dtype=float)
        table = np.genfromtxt(self.extinction_csv, delimiter=",", names=True)
        required = {"wavelength_um", "k"}
        if not required.issubset(table.dtype.names or ()):
            raise ValueError(
                "SiN extinction CSV must have wavelength_um and k columns."
            )
        w = np.asarray(table["wavelength_um"], dtype=float) * 1e-6
        kval = np.asarray(table["k"], dtype=float)
        if np.any(kval < 0) or np.any(np.diff(w) <= 0):
            raise ValueError("SiN extinction table must be ordered and k nonnegative.")
        if np.min(wavelength_m) < w[0] or np.max(wavelength_m) > w[-1]:
            raise ValueError("Requested wavelength lies outside the SiN k table.")
        return np.interp(wavelength_m, w, kval)

    def index(self, wavelength_m: float | np.ndarray) -> complex | np.ndarray:
        wavelength_m = np.asarray(wavelength_m, dtype=float)
        self.validate_range(wavelength_m)
        result = self._n(wavelength_m) + 1j * self._k(wavelength_m)
        return complex(result) if result.ndim == 0 else result


@dataclass
class TabulatedSilicon(MaterialModel):
    """Green (2008) intrinsic silicon n,k data at 300 K.

    The Green table reports n and the absorption coefficient alpha; k is
    calculated using k = alpha*lambda/(4*pi). The solver convention here is
    n + i*k for a passive material.
    """

    table_csv: Path
    name: str = "Si_Green_2008_intrinsic_300K"
    wavelength_range_m: tuple[float, float] = (1.26e-6, 1.34e-6)

    def __post_init__(self) -> None:
        table = np.genfromtxt(self.table_csv, delimiter=",", names=True)
        self._w_um = np.asarray(table["wavelength_um"], dtype=float)
        self._n_data = np.asarray(table["n"], dtype=float)
        self._k_data = np.asarray(table["k"], dtype=float)
        if np.any(np.diff(self._w_um) <= 0) or np.any(self._k_data < 0):
            raise ValueError("Green silicon table must be ordered and k nonnegative.")
        self.wavelength_range_m = (self._w_um[0] * 1e-6, self._w_um[-1] * 1e-6)

    def index(self, wavelength_m: float | np.ndarray) -> complex | np.ndarray:
        wavelength_m = np.asarray(wavelength_m, dtype=float)
        self.validate_range(wavelength_m)
        wavelength_um = wavelength_m * 1e6
        n = np.interp(wavelength_um, self._w_um, self._n_data)
        k = np.interp(wavelength_um, self._w_um, self._k_data)
        result = n + 1j * k
        return complex(result) if result.ndim == 0 else result


@dataclass
class LiquidIndex:
    """Liquid index with an optional wavelength_um,n CSV for future dispersion."""

    constant_n: float
    dispersion_csv: Path | None = None

    def index(self, wavelength_m: float | np.ndarray) -> np.ndarray:
        wavelength_m = np.asarray(wavelength_m, dtype=float)
        if self.dispersion_csv is None:
            return np.full_like(wavelength_m, self.constant_n, dtype=float)
        table = np.genfromtxt(self.dispersion_csv, delimiter=",", names=True)
        if not {"wavelength_um", "n"}.issubset(table.dtype.names or ()):
            raise ValueError("Liquid dispersion CSV must have wavelength_um and n columns.")
        x = np.asarray(table["wavelength_um"], dtype=float) * 1e-6
        y = np.asarray(table["n"], dtype=float)
        if np.any(np.diff(x) <= 0) or np.any(y <= 0):
            raise ValueError("Liquid dispersion table must be ordered with positive n.")
        if np.min(wavelength_m) < x[0] or np.max(wavelength_m) > x[-1]:
            raise ValueError("Requested wavelength lies outside the liquid-index table.")
        return np.interp(wavelength_m, x, y)


@dataclass
class MaterialBank:
    silica: MalitsonSilica
    silicon: TabulatedSilicon
    sin: LukeSiliconNitride
    liquid_cross: LiquidIndex
    liquid_bar: LiquidIndex


def build_material_bank(config: WorkflowConfig) -> MaterialBank:
    si_csv = config.resolved_path(config.materials.silicon_data_csv)
    if si_csv is None or not si_csv.exists():
        raise FileNotFoundError(f"Green 2008 silicon CSV not found: {si_csv}")
    sin_k_csv = config.resolved_path(config.materials.sin_extinction_csv)
    liquid_csv = config.resolved_path(config.materials.liquid_dispersion_csv)
    return MaterialBank(
        silica=MalitsonSilica(),
        silicon=TabulatedSilicon(si_csv),
        sin=LukeSiliconNitride(
            absolute_index_offset=config.materials.sin_absolute_index_offset,
            extinction_csv=sin_k_csv,
        ),
        liquid_cross=LiquidIndex(config.materials.liquid_cross_n, liquid_csv),
        liquid_bar=LiquidIndex(config.materials.liquid_bar_n, liquid_csv),
    )


def material_samples(model: MaterialModel, wavelengths_m: np.ndarray) -> np.ndarray:
    """Return an ascending-frequency [frequency_Hz, epsilon] Lumerical table."""
    wavelengths_m = np.asarray(wavelengths_m, dtype=float)
    index = np.asarray(model.index(wavelengths_m), dtype=complex)
    frequency = C0 / wavelengths_m
    eps = index**2
    order = np.argsort(frequency)
    return np.column_stack((frequency[order], eps[order]))


def liquid_samples(model: LiquidIndex, wavelengths_m: np.ndarray) -> np.ndarray:
    wavelengths_m = np.asarray(wavelengths_m, dtype=float)
    index = np.asarray(model.index(wavelengths_m), dtype=complex)
    frequency = C0 / wavelengths_m
    order = np.argsort(frequency)
    return np.column_stack((frequency[order], (index**2)[order]))


def material_report(bank: MaterialBank, wavelengths_m: np.ndarray) -> dict:
    wavelengths_m = np.asarray(wavelengths_m, dtype=float)
    report = {}
    for name, material in (
        ("silica", bank.silica),
        ("silicon", bank.silicon),
        ("sin", bank.sin),
    ):
        values = np.asarray(material.index(wavelengths_m), dtype=complex)
        report[name] = {
            "model": material.name,
            "mode_used_in_fde": name != "silicon",
            "wavelength_range_nm": [
                material.wavelength_range_m[0] * 1e9,
                material.wavelength_range_m[1] * 1e9,
            ],
            "n_at_sampled_wavelengths": values.real.tolist(),
            "k_at_sampled_wavelengths": values.imag.tolist(),
        }
    for name, model in (
        ("liquid_cross", bank.liquid_cross),
        ("liquid_bar", bank.liquid_bar),
    ):
        values = np.asarray(model.index(wavelengths_m), dtype=float)
        report[name] = {
            "mode_used_in_fde": True,
            "model": (
                "constant nondispersive index"
                if model.dispersion_csv is None
                else f"tabulated dispersion from {model.dispersion_csv}"
            ),
            "wavelength_range_nm": [
                float(np.min(wavelengths_m) * 1e9),
                float(np.max(wavelengths_m) * 1e9),
            ],
            "n_at_sampled_wavelengths": values.tolist(),
            "k_at_sampled_wavelengths": np.zeros_like(values).tolist(),
        }
    return report
