"""Configuration loading and validation.

All internal lengths are SI (metres), wavelengths in metres, propagation
constants/coupling coefficients in rad/m, and angles in radians.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import os
import math
import warnings
from pathlib import Path
from typing import Any


@dataclass
class DeviceConfig:
    length_m: float = 1.5e-3
    wavelength_min_m: float = 1.26e-6
    wavelength_max_m: float = 1.34e-6
    wavelength_samples_initial: int = 5
    z_samples_initial: int = 5
    z_samples_max: int = 33
    wavelength_samples_max: int = 33
    grid_convergence_rtol: float = 0.01


@dataclass
class GeometryConfig:
    w1_min_m: float = 0.3e-6
    w1_max_m: float = 2.0e-6
    w1_lower_m: float = 2.0e-6
    w1_endpoint_candidates_m: list[float] = field(
        default_factory=lambda: [0.3e-6, 1.15e-6, 2.0e-6]
    )
    h_wg_m: float = 350e-9
    h_rib_candidates_m: list[float] = field(
        default_factory=lambda: [175e-9]
    )
    h_gap_candidates_m: list[float] = field(
        default_factory=lambda: [1.0e-6]
    )
    h_gap_bounds_m: tuple[float, float] = (1.0e-6, 1.0e-6)
    h_rib_bounds_m: tuple[float, float] = (175e-9, 175e-9)
    d_c2c_bounds_m: tuple[float, float] = (-8e-6, 8e-6)
    d_c2c_input_m: float = -8e-6
    d_c2c_output_m: float = 8e-6
    box_thickness_m: float = 3e-6
    # Both residual slabs span the complete FDE x domain. The lateral margin
    # preserves the v20 10 um compact and 26 um coupled domain widths.
    domain_margin_x_m: float = 4e-6
    domain_margin_top_m: float = 3e-6
    upper_and_lower_use_same_h_rib: bool = True


@dataclass
class MaterialsConfig:
    silica_model: str = "Malitson 1965 fused silica Sellmeier"
    # Retained for table validation/reference only. Silicon is not installed in
    # MODE in the substrate-free v15 cross section.
    silicon_model: str = "Green 2008 intrinsic Si, 300 K, tabulated"
    silicon_data_csv: str | None = "../data/green_si_2008_1260_1340.csv"
    sin_model: str = "Luke et al. 2015 provisional stoichiometric LPCVD Sellmeier"
    sin_absolute_index_offset: float = 0.0
    sin_index_sensitivity_offsets: list[float] = field(default_factory=list)
    sin_extinction_csv: str | None = None
    liquid_cross_n: float = 1.55
    liquid_bar_n: float = 1.37
    liquid_dispersion_csv: str | None = None
    material_fit_max_coefficients: int = 10
    temperature_k: float = 300.0


@dataclass
class SolverConfig:
    engine: str = "python"
    api_path: str | None = "C:/Program Files/Lumerical/v221/api/python"
    threads: int = 2
    hide_gui: bool = False
    platform: str = "windows"
    number_of_trial_modes: int = 8
    # FDE uses mesh cell counts. mesh_accuracy is retained as a refinement
    # level label: each step from the configured base level doubles both counts.
    mesh_accuracy: int = 3
    mesh_cells_x: int = 528
    # Reduced with the substrate-free y-span to keep the nominal vertical
    # cell size close to the v10 value (~40 nm).
    mesh_cells_y: int = 208
    mesh_accuracy_refinements: list[int] = field(default_factory=lambda: [2, 3, 4])
    domain_scale_factors: list[float] = field(default_factory=lambda: [1.0, 1.25, 1.5])
    domain_margin_y_bottom_m: float = 0.5e-6
    # Endpoint detuning uses isolated guide modes, so its x origin is arbitrary.
    # Recenter those modes and size the x domain around the maximum guide/slab
    # envelope. Coupled-mode FDE samples continue to use the full domain.
    endpoint_compact_x: bool = True
    reuse_mode_project: bool = True
    batch_mode_data: bool = True
    raw_cache_memory_entries: int = 16
    zero_mode_retries: int = 1
    tracking_grid_nx: int = 144
    tracking_grid_ny: int = 72
    minimum_te_fraction: float = 0.45
    minimum_mode_overlap: float = 0.55
    field_overlap_crosscheck_max_basis_overlap: float = 0.95
    large_separation_check_m: float = 7e-6
    mesh_convergence_neff_tolerance: float = 2e-5
    domain_convergence_neff_tolerance: float = 2e-5
    coefficient_convergence_relative_tolerance: float = 0.05
    # A complete, self-contained Block 0 certificate; absent in legacy configs.
    calibration: dict | None = None


@dataclass
class ConvergenceConfig:
    max_solves: int = 1200
    max_seconds: float = 21600.0
    reference_domain_factors: list[float] = field(default_factory=lambda: [1., 1.25, 1.5, 1.75])
    reference_mesh_factors: list[float] = field(default_factory=lambda: [1., .75, .5, .375])
    candidate_margin_factors: list[float] = field(default_factory=lambda: [1., .75, .5])
    candidate_mesh_factors: list[float] = field(default_factory=lambda: [1., 1.25, 1.5])
    neff_atol: float = 2e-5
    coefficient_rtol: float = .01
    coefficient_atol_rad_per_m: float = 2.0
    phase_error_rad: float = .02
    loss_error_db: float = .01
    minimum_power_overlap: float = .999
    polarization_atol: float = .01
    comparison_dx_m: float = 60e-9
    comparison_dy_m: float = 30e-9
    local_mesh_padding_m: float = .4e-6
    pml_layers: int = 12
    pml_test_extra_layers: int = 4
    timing_repeats: int = 2
    reference_sentinels: int = 4


@dataclass
class OptimizerConfig:
    method: str = "bounded_powell"
    control_points: int = 6
    max_function_evaluations_per_grid: int = 120
    max_grid_refinements: int = 2
    minimum_control_step_fraction: float = 0.005
    maximum_slope_m_per_m: float | None = None
    maximum_curvature_per_m: float | None = None
    random_seed: int = 20261120


@dataclass
class EndpointSearchConfig:
    """Shared width table with constrained pair selection; optional Python PSO."""

    method: str = "adaptive_1d"
    adaptive_initial_widths: int = 9
    adaptive_max_widths: int = 33
    adaptive_width_tolerance_m: float = 5e-9
    adaptive_objective_rtol: float = 0.01
    cross_magnitude_balance_rtol: float = 0.10
    pso_generation_size: int = 8
    pso_max_generations: int = 15
    # Zero disables early stopping. Otherwise both relative best-score change
    # and normalized swarm spread must stay below tolerance for three generations.
    pso_tolerance: float = 0.0
    pso_random_seed: int | None = 20261120
    pso_inertia: float = 0.7298
    pso_cognitive: float = 1.49618
    pso_social: float = 1.49618
    pso_velocity_limit: float = 0.2
    pso_lower_mode_state_independent: bool = True
    # Applied to each candidate; no cutoff scan or automatic bound adjustment.
    pso_guided_neff_margin: float = 1e-5


@dataclass
class PropagationConfig:
    rk4_steps_initial: int = 400
    rk4_steps_max: int = 12800
    power_convergence_rtol: float = 1e-6
    xt_power_floor: float = 1e-15


@dataclass
class OutputConfig:
    directory: str = "outputs/run"
    cache_sqlite: str = "outputs/cache/fde_cache.sqlite"
    save_solver_projects: bool = False
    cache_max_bytes: int = 2_000_000_000


@dataclass
class ExecutionConfig:
    max_solves: int | None = None
    max_seconds: float | None = None
    resume: bool = True
    native_batch_size: int = 4
    reuse_verification: bool = True


@dataclass
class CharacterizationConfig:
    enabled: bool = False
    max_refinements: int = 4


@dataclass
class WorkflowConfig:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    materials: MaterialsConfig = field(default_factory=MaterialsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    endpoint_search: EndpointSearchConfig = field(default_factory=EndpointSearchConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    characterization: CharacterizationConfig = field(default_factory=CharacterizationConfig)
    project_root: Path = field(default_factory=Path.cwd, repr=False, compare=False)

    def validate(self) -> None:
        for name in ('max_solves', 'max_seconds'):
            value = getattr(self.execution, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError(f'execution.{name} must be positive or null.')
        if self.execution.max_solves is not None and not isinstance(self.execution.max_solves, int):
            raise ValueError('execution.max_solves must be an integer or null.')
        if isinstance(self.execution.native_batch_size, bool) or not isinstance(self.execution.native_batch_size, int) or not 1 <= self.execution.native_batch_size <= 64:
            raise ValueError('execution.native_batch_size must be in [1,64].')
        if isinstance(self.output.cache_max_bytes, bool) or not isinstance(self.output.cache_max_bytes, int) or self.output.cache_max_bytes < 0:
            raise ValueError('output.cache_max_bytes must be nonnegative (0 means unlimited).')
        if isinstance(self.characterization.max_refinements, bool) or not isinstance(self.characterization.max_refinements, int) or self.characterization.max_refinements < 0:
            raise ValueError('characterization.max_refinements must be nonnegative.')
        if not all(isinstance(value, bool) for value in (self.execution.resume, self.execution.reuse_verification, self.characterization.enabled)):
            raise ValueError('Resume, verification reuse and characterization flags must be booleans.')
        d, g, m, s, o, e, p = (
            self.device,
            self.geometry,
            self.materials,
            self.solver,
            self.optimizer,
            self.endpoint_search,
            self.propagation,
        )
        if d.length_m <= 0 or d.wavelength_min_m <= 0:
            raise ValueError("Device length and wavelengths must be positive SI values.")
        if d.wavelength_max_m <= d.wavelength_min_m:
            raise ValueError("wavelength_max_m must exceed wavelength_min_m.")
        if not (3 <= d.wavelength_samples_initial <= d.wavelength_samples_max):
            raise ValueError("Wavelength sample counts are inconsistent.")
        if not (3 <= d.z_samples_initial <= d.z_samples_max):
            raise ValueError("z sample counts are inconsistent.")
        if not (0 < g.w1_min_m < g.w1_max_m):
            raise ValueError("Upper-guide width bounds must be positive and ordered.")
        if g.w1_lower_m <= 0 or g.h_wg_m <= 0:
            raise ValueError("Lower-guide width and SiN height must be positive.")
        if not (0 <= g.h_rib_bounds_m[0] <= g.h_rib_bounds_m[1] < g.h_wg_m):
            raise ValueError("Residual slab bounds must satisfy 0 <= h_rib < h_wg.")
        if not (0 < g.h_gap_bounds_m[0] <= g.h_gap_bounds_m[1]):
            raise ValueError("Vertical gap bounds must be positive and ordered.")
        if g.upper_and_lower_use_same_h_rib is not True:
            raise ValueError(
                "This implementation uses one shared h_rib for both stacked guides."
            )
        if g.d_c2c_bounds_m[0] >= g.d_c2c_bounds_m[1]:
            raise ValueError("Lateral offset bounds are invalid.")
        if not (
            g.d_c2c_bounds_m[0] <= g.d_c2c_input_m <= g.d_c2c_bounds_m[1]
            and g.d_c2c_bounds_m[0] <= g.d_c2c_output_m <= g.d_c2c_bounds_m[1]
        ):
            raise ValueError("The exact endpoint offsets must lie inside the bounds.")
        if g.box_thickness_m < 3e-6 - 1e-15:
            raise ValueError("The bottom oxide must be at least the specified 3 um.")
        if g.domain_margin_x_m <= 0:
            raise ValueError("The lateral domain margin must be positive.")
        if g.domain_margin_top_m <= 0:
            raise ValueError('The top domain margin must be positive.')
        if s.domain_margin_y_bottom_m <= 0:
            raise ValueError("The bottom-oxide margin must be positive.")
        if not isinstance(s.endpoint_compact_x, bool):
            raise ValueError("solver.endpoint_compact_x must be true or false.")
        if m.liquid_bar_n <= 0 or m.liquid_cross_n <= 0:
            raise ValueError("Liquid refractive indices must be positive.")
        if abs(m.temperature_k - 300.0) > 1e-9:
            raise ValueError(
                "The selected Green silicon table is at 300 K; use a different "
                "model implementation before changing temperature_k."
            )
        if m.material_fit_max_coefficients < 1:
            raise ValueError("At least one material fit coefficient is required.")
        if s.number_of_trial_modes < 2:
            raise ValueError("At least two trial modes are needed for coupled modes.")
        for name in ('mesh_cells_x', 'mesh_cells_y'):
            value = getattr(s, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f'solver.{name} must be an integer >= 2.')
        if s.engine not in {"python", "native"}:
            raise ValueError("solver.engine must be python or native.")
        if s.tracking_grid_nx < 16 or s.tracking_grid_ny < 16:
            raise ValueError("The common field-overlap grid is too small.")
        if o.control_points < 4:
            raise ValueError("Use at least four spline control points per profile.")
        if e.method not in {"adaptive_1d", "python_grid", "python_pso"}:
            raise ValueError("endpoint_search.method must be adaptive_1d, python_grid or python_pso.")
        if e.method == "adaptive_1d" and not s.endpoint_compact_x:
            raise ValueError("adaptive_1d requires solver.endpoint_compact_x=true.")
        if not (isinstance(e.adaptive_initial_widths, int) and isinstance(e.adaptive_max_widths, int)
                and 3 <= e.adaptive_initial_widths <= e.adaptive_max_widths):
            raise ValueError("Adaptive width counts must satisfy 3 <= initial <= maximum.")
        if not math.isfinite(e.adaptive_width_tolerance_m) or e.adaptive_width_tolerance_m <= 0:
            raise ValueError("adaptive_width_tolerance_m must be finite and positive.")
        if not math.isfinite(e.adaptive_objective_rtol) or not 0 < e.adaptive_objective_rtol < 1:
            raise ValueError("adaptive_objective_rtol must lie in (0,1).")
        if not math.isfinite(e.cross_magnitude_balance_rtol) or not 0 < e.cross_magnitude_balance_rtol < 2:
            raise ValueError("cross_magnitude_balance_rtol must lie in (0,2).")
        if not 0 < o.minimum_control_step_fraction < 1:
            raise ValueError("minimum_control_step_fraction must lie in (0,1).")
        if o.max_function_evaluations_per_grid < 1 or o.max_grid_refinements < 0:
            raise ValueError("Profile evaluation and refinement budgets are invalid.")
        if s.raw_cache_memory_entries < 0 or not isinstance(s.raw_cache_memory_entries, int):
            raise ValueError("raw_cache_memory_entries must be a nonnegative integer.")
        if isinstance(s.zero_mode_retries, bool) or not isinstance(s.zero_mode_retries, int) or not 0 <= s.zero_mode_retries <= 2:
            raise ValueError("zero_mode_retries must be an integer from 0 to 2.")
        if not isinstance(s.reuse_mode_project, bool) or not isinstance(s.batch_mode_data, bool):
            raise ValueError("MODE reuse and batch settings must be Boolean.")
        for name, value, minimum in (
            ("pso_generation_size", e.pso_generation_size, 2),
            ("pso_max_generations", e.pso_max_generations, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in ("pso_tolerance", "pso_inertia", "pso_cognitive", "pso_social", "pso_guided_neff_margin"):
            value = getattr(e, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not math.isfinite(e.pso_velocity_limit) or not 0 < e.pso_velocity_limit <= 1:
            raise ValueError("pso_velocity_limit must lie in (0, 1].")
        if e.pso_random_seed is not None and (
            isinstance(e.pso_random_seed, bool)
            or not isinstance(e.pso_random_seed, int) or e.pso_random_seed < 0
        ):
            raise ValueError("pso_random_seed must be a nonnegative integer or null.")
        if e.method in {"adaptive_1d", "python_pso"} and e.pso_lower_mode_state_independent is not True:
            raise ValueError(
                "Adaptive/PSO endpoint search uses state-independent lower modes; choose python_grid "
                "to solve lower modes separately in each liquid state."
            )
        if o.maximum_slope_m_per_m is not None and o.maximum_slope_m_per_m <= 0:
            raise ValueError("A configured slope limit must be positive.")
        if o.maximum_curvature_per_m is not None and o.maximum_curvature_per_m <= 0:
            raise ValueError("A configured curvature limit must be positive.")
        if p.rk4_steps_initial < 1 or p.rk4_steps_max < p.rk4_steps_initial:
            raise ValueError("Propagation integration step settings are inconsistent.")
        if not math.isfinite(p.power_convergence_rtol) or not 0 < p.power_convergence_rtol < 1:
            raise ValueError("Propagation power_convergence_rtol must lie in (0,1).")
        if not (0 < p.xt_power_floor < 1):
            raise ValueError("XT power floor must lie between zero and one.")
        c = self.convergence
        for name in ('max_solves', 'timing_repeats', 'reference_sentinels', 'pml_layers', 'pml_test_extra_layers'):
            value = getattr(c, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'convergence.{name} must be a positive integer.')
        for name in ('max_seconds', 'neff_atol', 'coefficient_rtol', 'coefficient_atol_rad_per_m',
                     'phase_error_rad', 'loss_error_db', 'polarization_atol', 'comparison_dx_m',
                     'comparison_dy_m', 'local_mesh_padding_m'):
            value = getattr(c, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'convergence.{name} must be finite and positive.')
        if not 0 < c.minimum_power_overlap < 1:
            raise ValueError('convergence.minimum_power_overlap must lie in (0,1).')
        for name, direction in (('reference_domain_factors', 1), ('reference_mesh_factors', -1)):
            values = getattr(c, name)
            if (len(values) < 3 or values[0] != 1 or
                    any(not math.isfinite(v) or v <= 0 for v in values) or
                    any(direction * (b-a) <= 0 for a, b in zip(values, values[1:]))):
                raise ValueError(f'convergence.{name} must start at 1 and contain at least three ordered levels.')
        for name in ('candidate_margin_factors', 'candidate_mesh_factors'):
            values = getattr(c, name)
            if not values or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f'convergence.{name} must contain positive finite factors.')
        if s.calibration is not None:
            from .block0 import validate_certificate
            validate_certificate(self, s.calibration)

    def resolved_path(self, raw_path: str | None) -> Path | None:
        if raw_path is None:
            return None
        path = Path(raw_path)
        return path if path.is_absolute() else (self.project_root / path).resolve()

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("project_root", None)
        return values


def _tuple_fields(instance: Any, source: dict[str, Any]) -> Any:
    tuple_fields = {"h_gap_bounds_m", "h_rib_bounds_m", "d_c2c_bounds_m"}
    for key in tuple_fields.intersection(source):
        setattr(instance, key, tuple(source[key]))
    return instance


def load_config(path: str | Path) -> WorkflowConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return config_from_dict(raw, project_root=raw.get('project_root', config_path.parent.parent))


def config_from_dict(raw: dict, *, project_root: str | Path | None = None) -> WorkflowConfig:
    """Rebuild saved configuration without relying on an output folder layout."""
    geometry_raw = raw.get("geometry", {})
    geometry_values = {
        k: v
        for k, v in geometry_raw.items()
        if k not in {
            "h_gap_bounds_m", "h_rib_bounds_m", "d_c2c_bounds_m",
            # Ignore the substrate-thickness setting used by pre-v11 configs.
            "substrate_depth_m",
            "rib_slab_overhang_m", "rib_slab_overhang_is_provisional",
        }
    }
    if "rib_slab_overhang_m" in geometry_raw:
        # Retire finite slabs while preserving an older config's domain width.
        overhang = float(geometry_raw["rib_slab_overhang_m"])
        if not math.isfinite(overhang) or overhang < 0:
            raise ValueError("Legacy rib_slab_overhang_m must be finite and nonnegative.")
        geometry_values["domain_margin_x_m"] = geometry_raw.get("domain_margin_x_m", 2e-6) + overhang
        warnings.warn("Residual slabs now extend to both FDE x boundaries. Legacy slab overhang was folded into domain_margin_x_m to preserve the domain width.", UserWarning, stacklevel=2)
    solver_raw = dict(raw.get("solver", {}))
    env_api_path = os.environ.get("LUMERICAL_PYTHON_API")
    if env_api_path:
        solver_raw["api_path"] = env_api_path
    env_hide_gui = os.environ.get("VLAC_HIDE_GUI")
    if env_hide_gui is not None:
        normalized = env_hide_gui.strip().lower()
        if normalized not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
            raise ValueError("VLAC_HIDE_GUI must be true/false, 1/0, yes/no, or on/off.")
        solver_raw["hide_gui"] = normalized in {"1", "true", "yes", "on"}
    env_platform = os.environ.get("VLAC_PLATFORM")
    if env_platform:
        solver_raw["platform"] = env_platform
    endpoint_raw = dict(raw.get("endpoint_search", {}))
    retired = [key for key in endpoint_raw if key.startswith(("width_preflight_", "width_auto_expand_"))]
    for key in retired:
        endpoint_raw.pop(key)
    migrated = endpoint_raw.get("method") == "lumerical_pso"
    if migrated:
        endpoint_raw["method"] = "python_pso"
    if "pso_reset_random_generator" in endpoint_raw:
        reset = endpoint_raw.pop("pso_reset_random_generator")
        endpoint_raw.setdefault("pso_random_seed", raw.get("optimizer", {}).get("random_seed", 20261120) if reset else None)
        migrated = True
    if migrated or retired:
        warnings.warn(
            "Migrating legacy endpoint settings to Python PSO. Native MODE PSO and "
            "width cutoff preflight settings have been retired; width bounds are preserved.",
            UserWarning, stacklevel=2,
        )
    cfg = WorkflowConfig(
        device=DeviceConfig(**raw.get("device", {})),
        geometry=_tuple_fields(GeometryConfig(**geometry_values), geometry_raw),
        materials=MaterialsConfig(**raw.get("materials", {})),
        solver=SolverConfig(**solver_raw),
        optimizer=OptimizerConfig(**raw.get("optimizer", {})),
        endpoint_search=EndpointSearchConfig(**endpoint_raw),
        propagation=PropagationConfig(**raw.get("propagation", {})),
        output=OutputConfig(**raw.get("output", {})),
        convergence=ConvergenceConfig(**raw.get("convergence", {})),
        execution=ExecutionConfig(**raw.get("execution", {})),
        characterization=CharacterizationConfig(**raw.get("characterization", {})),
        project_root=Path(project_root if project_root is not None else raw.get('project_root', Path.cwd())).resolve(),
    )
    cfg.validate()
    return cfg


def save_config(config: WorkflowConfig, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
