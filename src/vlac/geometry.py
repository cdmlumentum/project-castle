"""Cross-section geometry and longitudinal spline parameterization."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline

from .config import GeometryConfig


@dataclass(frozen=True)
class CrossSectionGeometry:
    w1_m: float
    w2_m: float
    h_wg_m: float
    h_rib_m: float
    h_gap_m: float
    d_c2c_m: float
    box_thickness_m: float
    state: str
    liquid_index: float

    @property
    def lower_bottom_y_m(self) -> float:
        return 0.0

    @property
    def lower_top_y_m(self) -> float:
        return self.h_wg_m

    @property
    def upper_bottom_y_m(self) -> float:
        return self.lower_top_y_m + self.h_gap_m

    @property
    def upper_top_y_m(self) -> float:
        return self.upper_bottom_y_m + self.h_wg_m


@dataclass(frozen=True)
class Rectangle:
    name: str
    material: str
    x_center_m: float
    y_center_m: float
    x_span_m: float
    y_span_m: float


def component_enabled(name, include_upper, include_lower, isolated_reference):
    """Only ridges are removed in the common-background reference alternative."""
    if name in {"upper_slab", "lower_slab"} and isolated_reference == "retained_slabs_v1":
        return True
    if name in {"upper_slab", "upper_ridge"}:
        return bool(include_upper)
    if name in {"lower_slab", "lower_ridge"}:
        return bool(include_lower)
    return True


def make_geometry(
    config: GeometryConfig,
    *,
    w1_m: float,
    h_rib_m: float,
    h_gap_m: float,
    d_c2c_m: float,
    state: str,
    liquid_index: float,
) -> CrossSectionGeometry:
    if state not in {"cross", "bar"}:
        raise ValueError("state must be 'cross' or 'bar'.")
    if not config.w1_min_m <= w1_m <= config.w1_max_m:
        raise ValueError("w1 lies outside its configured bounds.")
    if not config.h_rib_bounds_m[0] <= h_rib_m <= config.h_rib_bounds_m[1]:
        raise ValueError("h_rib lies outside its configured bounds.")
    if not config.h_gap_bounds_m[0] <= h_gap_m <= config.h_gap_bounds_m[1]:
        raise ValueError("h_gap lies outside its configured bounds.")
    if not config.d_c2c_bounds_m[0] <= d_c2c_m <= config.d_c2c_bounds_m[1]:
        raise ValueError("d_c2c lies outside its configured bounds.")
    return CrossSectionGeometry(
        w1_m=w1_m,
        w2_m=config.w1_lower_m,
        h_wg_m=config.h_wg_m,
        h_rib_m=h_rib_m,
        h_gap_m=h_gap_m,
        d_c2c_m=d_c2c_m,
        box_thickness_m=config.box_thickness_m,
        state=state,
        liquid_index=liquid_index,
    )


def rectangles_for_cross_section(
    geometry: CrossSectionGeometry,
    *,
    x_span_m: float,
    y_min_m: float,
    y_max_m: float,
) -> list[Rectangle]:
    """Build the cross-section's non-overlapping host layers and SiN ribs.

    The lower guide and the vertical gap are embedded in SiO2. The upper guide
    is placed in a liquid region above its lower face. Both residual slabs span
    the full FDE x domain; their edges do not move with the upper ridge.
    """
    if y_min_m >= -geometry.box_thickness_m:
        raise ValueError(
            "FDE domain must extend below the configured BOX depth so the bottom "
            "boundary remains inside silica."
        )
    rects = [
        Rectangle(
            "buried_oxide_and_gap", "SiO2_Malitson_1965",
            0.0, (y_min_m + geometry.upper_bottom_y_m) / 2,
            x_span_m, geometry.upper_bottom_y_m - y_min_m,
        ),
        Rectangle(
            "upper_liquid_cladding", "Liquid",
            0.0, (geometry.upper_bottom_y_m + y_max_m) / 2,
            x_span_m, y_max_m - geometry.upper_bottom_y_m,
        ),
    ]

    if geometry.h_rib_m > 0:
        rects.append(
            Rectangle(
                "lower_slab", "Si3N4_Luke_2015_provisional",
                0.0, geometry.h_rib_m / 2,
                x_span_m,
                geometry.h_rib_m,
            )
        )
        rects.append(
            Rectangle(
                "upper_slab", "Si3N4_Luke_2015_provisional",
                0.0, geometry.upper_bottom_y_m + geometry.h_rib_m / 2,
                x_span_m,
                geometry.h_rib_m,
            )
        )

    ridge_height = geometry.h_wg_m - geometry.h_rib_m
    rects.extend(
        [
            Rectangle(
                "lower_ridge", "Si3N4_Luke_2015_provisional",
                0.0, geometry.h_rib_m + ridge_height / 2,
                geometry.w2_m, ridge_height,
            ),
            Rectangle(
                "upper_ridge", "Si3N4_Luke_2015_provisional",
                geometry.d_c2c_m,
                geometry.upper_bottom_y_m + geometry.h_rib_m + ridge_height / 2,
                geometry.w1_m, ridge_height,
            ),
        ]
    )
    # The bottom PML now terminates inside an oxide half-space. No silicon
    # substrate rectangle is added to the cross section.
    return rects


class ClampedCubicProfile:
    """Open-clamped cubic B-spline whose coefficients are bounded control points."""

    def __init__(self, length_m: float, control_values: np.ndarray):
        self.length_m = float(length_m)
        self.control_values = np.asarray(control_values, dtype=float)
        if self.control_values.ndim != 1 or len(self.control_values) < 4:
            raise ValueError("A cubic profile needs at least four control values.")
        n = len(self.control_values)
        degree = 3
        interior_count = n - degree - 1
        if interior_count:
            interior = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
        else:
            interior = np.array([], dtype=float)
        knots = np.concatenate(
            (np.zeros(degree + 1), interior, np.ones(degree + 1))
        )
        self._spline = BSpline(knots, self.control_values, degree, extrapolate=False)

    def __call__(self, z_m: float | np.ndarray) -> np.ndarray:
        u = np.asarray(z_m, dtype=float) / self.length_m
        if np.any((u < -1e-12) | (u > 1 + 1e-12)):
            raise ValueError("Profile queried outside 0 <= z <= L.")
        return np.asarray(self._spline(np.clip(u, 0.0, 1.0)))

    def derivative(self, z_m: float | np.ndarray, order: int = 1) -> np.ndarray:
        u = np.asarray(z_m, dtype=float) / self.length_m
        return np.asarray(self._spline.derivative(order)(u)) / self.length_m**order


def blackman_theta(z_m: float | np.ndarray, length_m: float) -> np.ndarray:
    u = np.asarray(z_m, dtype=float) / length_m
    return (
        np.pi * u
        - 0.5952 * np.sin(2 * np.pi * u)
        + 0.0476 * np.sin(4 * np.pi * u)
    )


class AnchoredProfile:
    """Shape-preserving interpolation through measured-design anchors in SI."""
    def __init__(self, length_m, z_m, values):
        from scipy.interpolate import PchipInterpolator
        self.length_m = float(length_m)
        self.z_m = np.asarray(z_m, dtype=float)
        self.values = np.asarray(values, dtype=float)
        if (self.z_m.ndim != 1 or self.values.shape != self.z_m.shape or len(self.z_m) < 3
                or not np.all(np.isfinite(self.z_m)) or not np.all(np.isfinite(self.values))
                or np.any(np.diff(self.z_m) <= 0)
                or not np.allclose(self.z_m[[0,-1]], [0., self.length_m], rtol=0, atol=1e-15)):
            raise ValueError('Invalid PCHIP anchor coordinates or values.')
        self._spline = PchipInterpolator(self.z_m, self.values, extrapolate=False)

    def __call__(self, z_m):
        z = np.asarray(z_m, dtype=float)
        if np.any(~np.isfinite(z)) or np.any((z < 0) | (z > self.length_m)):
            raise ValueError('Anchor profile queried outside 0 <= z <= L.')
        result = np.asarray(self._spline(z))
        low, high = float(np.min(self.values)), float(np.max(self.values))
        tolerance = 32*np.finfo(float).eps*max(abs(low),abs(high),1e-300)
        if np.any(result<low-tolerance) or np.any(result>high+tolerance):
            raise ValueError('PCHIP unexpectedly exceeded its anchor range.')
        result = np.clip(result,low,high)  # roundoff only; PCHIP cannot overshoot in exact arithmetic
        for coordinate,value in zip(self.z_m,self.values):
            result = np.where(z==coordinate,value,result)
        return result

    def derivative(self, z_m, order=1):
        return np.asarray(self._spline.derivative(order)(z_m))

    def max_abs_derivative(self, order):
        # Include both sides of knots: PCHIP second derivatives can jump.
        maximum = 0.
        for i, h in enumerate(np.diff(self.z_m)):
            polynomial = np.polyder(self._spline.c[:,i], order)
            roots = np.roots(np.polyder(polynomial))
            points = [0., h]+[float(r.real) for r in roots if abs(r.imag)<1e-14 and 0<r.real<h]
            maximum = max(maximum, float(np.max(np.abs(np.polyval(polynomial, points)))))
        return maximum


def profile_from_values(length_m, values, anchor_z_m=None):
    return (ClampedCubicProfile(length_m, values) if anchor_z_m is None
            else AnchoredProfile(length_m, anchor_z_m, values))


def solution_profiles(config, solution):
    return tuple(profile_from_values(config.device.length_m, values, solution.anchor_z_m)
                 for values in (solution.w1_control_m, solution.d_c2c_control_m))
