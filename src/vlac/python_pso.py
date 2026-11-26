"""Python-controlled particle swarm and endpoint FDE objective.

MODE performs individual FDE solves through LumapiBackend. Swarm state,
objective evaluation, mode checks, progress and exports stay in Python.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .config import EndpointSearchConfig, WorkflowConfig
from .endpoint import EndpointSearchResult
from .endpoint_constraints import (
    CROSS_STATE_SIGN_REQUIREMENT, EndpointSignError,
    require_cross_state_signs, cross_state_sign_report, EndpointConstraintError,
    endpoint_constraint_report, require_endpoint_constraints,
)
from .geometry import make_geometry
from .materials import MaterialBank
from .modes import InvalidModeError


@dataclass
class SwarmResult:
    position: np.ndarray
    score: float
    generations: int
    evaluations: int
    converged: bool


def run_particle_swarm(
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    settings: EndpointSearchConfig,
    *,
    on_generation: Callable[[dict], None] | None = None,
    initial_positions_normalized: np.ndarray | None = None,
) -> SwarmResult:
    """Maximize a scalar objective; -inf denotes a physically invalid particle.

    Positions and velocities use unit coordinates to avoid SI scaling issues.
    Updates are synchronous and serial. Unexpected objective exceptions propagate.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not np.all(np.isfinite(bounds)):
        raise ValueError("PSO bounds must be a finite (dimensions, 2) array.")
    lo, hi = bounds[:, 0], bounds[:, 1]
    if np.any(hi <= lo):
        raise ValueError("PSO bounds must be strictly ordered.")
    n, dimensions = settings.pso_generation_size, len(lo)
    rng = np.random.default_rng(settings.pso_random_seed)
    positions = rng.uniform(0, 1, (n, dimensions))
    # Include the largest-width pair first so a narrow invalid interval does not
    # make a small smoke run depend entirely on random initialization.
    corners = [np.ones(dimensions), np.arange(dimensions) % 2,
               1 - np.arange(dimensions) % 2, np.zeros(dimensions)]
    for index, corner in enumerate(corners[:n]):
        positions[index] = corner
    if initial_positions_normalized is not None:
        seeds = np.asarray(initial_positions_normalized, dtype=float)
        if (seeds.ndim != 2 or seeds.shape[1] != dimensions or len(seeds) > n
                or not np.all(np.isfinite(seeds)) or np.any((seeds < 0) | (seeds > 1))):
            raise ValueError("Initial normalized PSO positions must fit the population and lie in [0, 1].")
        # Explicit endpoint seeds replace generic corners. Remaining particles
        # retain their random positions, covering the entire allowed interval.
        positions = rng.uniform(0, 1, (n, dimensions))
        positions[:len(seeds)] = seeds
    limit = settings.pso_velocity_limit
    velocities = rng.uniform(-limit, limit, positions.shape)
    personal_positions = positions.copy()
    personal_scores = np.full(n, -np.inf)
    best_score, best_position = -np.inf, positions[0].copy()
    stable, evaluations, converged = 0, 0, False
    for generation in range(1, settings.pso_max_generations + 1):
        previous_score = best_score
        scores = np.asarray([objective(lo + row * (hi - lo)) for row in positions], dtype=float)
        evaluations += n
        if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
            raise RuntimeError("PSO objective returned NaN or positive infinity.")
        improved = scores > personal_scores
        personal_scores[improved] = scores[improved]
        personal_positions[improved] = positions[improved]
        winner = int(np.argmax(personal_scores))
        if personal_scores[winner] > best_score:
            best_score = float(personal_scores[winner])
            best_position = personal_positions[winner].copy()
        relative_change = (
            abs(best_score - previous_score) / max(abs(best_score), 1e-30)
            if np.isfinite(previous_score) and np.isfinite(best_score) else None
        )
        spread = float(np.max(np.ptp(positions, axis=0)))
        if settings.pso_tolerance > 0 and relative_change is not None:
            stable = stable + 1 if max(relative_change, spread) <= settings.pso_tolerance else 0
            converged = stable >= 3
        state = {
            "generation": generation, "evaluations": evaluations,
            "best_position": (lo + best_position * (hi - lo)).tolist(),
            "best_score": best_score if np.isfinite(best_score) else None,
            "positions_normalized": positions.tolist(),
            "velocities_normalized": velocities.tolist(),
            "personal_positions_normalized": personal_positions.tolist(),
            "personal_scores": [float(s) if np.isfinite(s) else None for s in personal_scores],
            "generation_valid_particles": int(np.count_nonzero(np.isfinite(scores))),
            "relative_best_change": relative_change, "normalized_swarm_spread": spread,
            "stable_generations": stable, "converged": converged,
            "rng_state": rng.bit_generator.state,
        }
        if on_generation:
            on_generation(state)
        if converged or generation == settings.pso_max_generations:
            break
        r1, r2 = rng.random(positions.shape), rng.random(positions.shape)
        velocities = np.clip(
            settings.pso_inertia * velocities
            + settings.pso_cognitive * r1 * (personal_positions - positions)
            + settings.pso_social * r2 * (best_position - positions), -limit, limit,
        )
        proposed = positions + velocities
        hit = (proposed < 0) | (proposed > 1)
        positions = np.clip(proposed, 0, 1)
        velocities[hit] *= -0.5
    if not np.isfinite(best_score):
        raise RuntimeError(
            "Python PSO found no candidate satisfying all mode checks and the cross-state "
            "endpoint sign restriction (input < 0, output > 0 at both band edges), increasing width and magnitude balance. "
            "Review endpoint_pso_particles.csv and the configured widths/materials. "
            "Width bounds were not changed."
        )
    return SwarmResult(lo + best_position * (hi - lo), best_score, generation, evaluations, converged)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generation_detuning_fields(records: list[dict]) -> dict:
    """Eight signed edge values and four signed worst-edge summaries."""
    result = {}
    for state in ("cross", "bar"):
        for end, label in (("input", "in"), ("output", "out")):
            selected = sorted(
                (r for r in records if r["state"] == state and r["end"] == end),
                key=lambda r: r["wavelength_m"],
            )
            if len(selected) != 2:
                raise RuntimeError(f"Expected two band edges for {state} {end}.")
            for edge, row in zip(("min", "max"), selected):
                result[f"delta_beta_{state}_{label}_lambda_{edge}_rad_per_m"] = row["delta_beta_real_rad_per_m"]
            worst = min(selected, key=lambda r: abs(r["delta_beta_real_rad_per_m"]))
            result[f"delta_beta_{state}_{label}_rad_per_m"] = worst["delta_beta_real_rad_per_m"]
            result[f"delta_beta_{state}_{label}_worst_wavelength_m"] = worst["wavelength_m"]
    return result


class EndpointEvaluator:
    """Use the same mode selector for every particle and the final fresh solve."""

    def __init__(self, backend, config: WorkflowConfig, bank: MaterialBank, *, progress=None):
        self.backend, self.config, self.bank, self.progress = backend, config, bank, progress
        self.wavelengths = [config.device.wavelength_min_m, config.device.wavelength_max_m]
        self.lower_modes: dict = {}

    def geometry(self, width, offset, state, wavelength):
        g = self.config.geometry
        liquid = self.bank.liquid_cross if state == "cross" else self.bank.liquid_bar
        return make_geometry(
            g, w1_m=float(width), h_rib_m=g.h_rib_bounds_m[0], h_gap_m=g.h_gap_bounds_m[0],
            d_c2c_m=offset, state=state, liquid_index=float(liquid.index(wavelength)),
        )

    def check_mode(self, mode, overlap, reference, geometry, wavelength, guide):
        s, e = self.config.solver, self.config.endpoint_search
        if not all(np.isfinite(v) for v in (mode.neff.real, mode.neff.imag, mode.te_fraction, overlap)):
            raise InvalidModeError(f"{guide} mode has non-finite data.")
        if mode.te_fraction < s.minimum_te_fraction:
            raise InvalidModeError(f"{guide} TE fraction {mode.te_fraction:.4g} < {s.minimum_te_fraction}.")
        if reference is not None and overlap < s.minimum_mode_overlap:
            raise InvalidModeError(f"{guide} wavelength-tracking overlap {overlap:.4g} < {s.minimum_mode_overlap}.")
        n_silica = float(np.real(self.bank.silica.index(wavelength)))
        n_clad = max(n_silica, geometry.liquid_index) if guide == "upper" else n_silica
        if mode.neff.real <= n_clad + e.pso_guided_neff_margin:
            raise InvalidModeError(f"{guide} mode Re(neff)-n_clad={mode.neff.real-n_clad:.4g} is too small.")

    def precompute_lower(self, *, use_cache=True):
        if not use_cache:
            self.mode_profiles = []
        previous = None
        result = {}
        for index, wavelength in enumerate(self.wavelengths, 1):
            geometry = self.geometry(self.config.geometry.w1_min_m, 0.0, "cross", wavelength)
            mode, overlap = self.backend.solve_isolated_mode(
                geometry, wavelength, guide="lower", reference=previous,
                compact_x=self.config.solver.endpoint_compact_x, use_cache=use_cache,
            )
            self.check_mode(mode, overlap, previous, geometry, wavelength, "lower")
            if not use_cache:
                from .endpoint_diagnostics import profile_packet
                self.mode_profiles.append(profile_packet(self.backend, mode, geometry, wavelength, "lower", "both"))
            previous = mode
            result[wavelength] = (mode, overlap)
            if self.progress:
                self.progress.update(index, detail=f"lower guide {wavelength*1e9:.0f} nm")
        self.lower_modes = result

    def evaluate(self, widths, *, use_cache=True):
        records = []
        g = self.config.geometry
        for end, width, offset in (("input", widths[0], g.d_c2c_input_m),
                                   ("output", widths[1], g.d_c2c_output_m)):
            for state in ("cross", "bar"):
                previous = None
                for wavelength in self.wavelengths:
                    geometry = self.geometry(width, offset, state, wavelength)
                    context = f"{end} {state}, width {width*1e9:.3f} nm, lambda {wavelength*1e9:.0f} nm"
                    if self.progress:
                        self.progress.note(context, force=True)
                    try:
                        upper, overlap = self.backend.solve_isolated_mode(
                            geometry, wavelength, guide="upper", reference=previous,
                            compact_x=self.config.solver.endpoint_compact_x, use_cache=use_cache,
                        )
                        self.check_mode(upper, overlap, previous, geometry, wavelength, "upper")
                        if not use_cache:
                            from .endpoint_diagnostics import profile_packet
                            self.mode_profiles.append(profile_packet(self.backend, upper, geometry, wavelength, "upper", end))
                    except InvalidModeError as exc:
                        raise InvalidModeError(f"{context}: {exc}") from exc
                    previous = upper
                    lower, lower_overlap = self.lower_modes[wavelength]
                    beta_upper, beta_lower = 2*np.pi*upper.neff/wavelength, 2*np.pi*lower.neff/wavelength
                    signed = beta_upper - beta_lower
                    records.append({
                        "h_gap_m": geometry.h_gap_m, "h_rib_m": geometry.h_rib_m,
                        "w1_m": float(width), "end": end, "d_c2c_m": float(offset),
                        "state": state, "wavelength_m": wavelength,
                        "beta_upper_real_rad_per_m": float(beta_upper.real),
                        "beta_upper_imag_rad_per_m": float(beta_upper.imag),
                        "beta_lower_real_rad_per_m": float(beta_lower.real),
                        "beta_lower_imag_rad_per_m": float(beta_lower.imag),
                        "delta_beta_real_rad_per_m": float(signed.real),
                        "delta_beta_imag_rad_per_m": float(signed.imag),
                        "upper_mode_identity": upper.identity, "lower_mode_identity": lower.identity,
                        "upper_te_fraction": float(upper.te_fraction), "lower_te_fraction": float(lower.te_fraction),
                        "upper_tracking_overlap": float(overlap), "lower_tracking_overlap": float(lower_overlap),
                    })
                    if self.progress and not use_cache:
                        self.progress.update(2 + len(records), detail=context)
        require_endpoint_constraints(records, self.config.endpoint_search)
        return min(abs(r["delta_beta_real_rad_per_m"]) for r in records), records


def search_endpoints_python_pso(
    backend, config: WorkflowConfig, bank: MaterialBank, *, progress=None,
    validation_only=False, output_dir: str | Path | None = None,
) -> EndpointSearchResult:
    from dataclasses import replace

    config.validate()
    g, settings = config.geometry, config.endpoint_search
    if g.h_gap_bounds_m[0] != g.h_gap_bounds_m[1] or g.h_rib_bounds_m[0] != g.h_rib_bounds_m[1]:
        raise ValueError("Endpoint Python PSO requires fixed h_gap and h_rib bounds.")
    if settings.pso_lower_mode_state_independent is not True:
        raise ValueError("Endpoint Python PSO requires state-independent lower modes.")
    if validation_only:
        settings = replace(settings, pso_generation_size=3, pso_max_generations=1, pso_tolerance=0.0)
    out = Path(output_dir) if output_dir is not None else config.resolved_path(config.output.directory)
    out.mkdir(parents=True, exist_ok=True)
    # An output directory may be reused. Remove retired artifacts and previous
    # success records so a failed rerun cannot appear to have a current winner.
    for name in ("endpoint_pso_setup.lsf", "endpoint_pso.lms", "endpoint_width_preflight.json",
                 "endpoint_pso_code_verification.json", "endpoint_pso_best.json",
                 "endpoint_pso_verification.json", "endpoint_pso_checkpoint.json",
                 "endpoint_summary.json", "endpoint_detunings.csv", "endpoint_failure.json", "endpoint_adaptive_best.json"):
        (out / name).unlink(missing_ok=True)
    for name in ("endpoint_pso_particles.csv", "endpoint_pso_history.csv", "endpoint_pso_generation_detunings.csv"):
        (out / name).write_text("", encoding="utf-8")
    evaluator = EndpointEvaluator(backend, config, bank, progress=progress)
    if progress:
        progress.start_stage("Precompute state-independent lower-guide modes", total=2, unit="band-edge modes")
    evaluator.precompute_lower()
    if progress:
        progress.finish_stage()
        progress.start_stage("Endpoint detuning search", total=settings.pso_generation_size*settings.pso_max_generations, unit="PSO particles")
    particles, history, generation_rows = [], [], []
    evaluated: dict[tuple, tuple] = {}
    rejected_signs: dict[tuple, Exception] = {}

    def objective(widths):
        evaluation = len(particles) + 1
        generation = (evaluation - 1) // settings.pso_generation_size + 1
        row = {"generation": generation, "particle": (evaluation-1) % settings.pso_generation_size + 1,
               "evaluation": evaluation, "w1_input_m": float(widths[0]), "w1_output_m": float(widths[1]),
               "valid": True, "mode_checks_passed": False, "cross_state_signs_passed": None,
               "rejection_reason": "", "bottleneck_rad_per_m": None, "error": ""}
        key = tuple(float(v) for v in widths)
        try:
            if key in rejected_signs:
                raise rejected_signs[key]
            if key not in evaluated:
                evaluated[key] = evaluator.evaluate(widths)
            score, records = evaluated[key]
            constraints = require_endpoint_constraints(records, settings)
            row.update(mode_checks_passed=True, cross_state_signs_passed=True, endpoint_constraints_passed=True,
                       maximum_relative_mismatch=constraints["magnitude_balance"]["maximum_relative_mismatch"],
                       bottleneck_rad_per_m=score, **generation_detuning_fields(records))
        except (EndpointSignError, EndpointConstraintError) as exc:
            rejected_signs[key] = exc
            score = -np.inf
            row.update(valid=False, mode_checks_passed=True,
                       cross_state_signs_passed=cross_state_sign_report(exc.records)["passed"], endpoint_constraints_passed=False,
                       maximum_relative_mismatch=endpoint_constraint_report(exc.records, settings)["magnitude_balance"]["maximum_relative_mismatch"],
                       rejection_reason="cross_state_endpoint_sign" if isinstance(exc, EndpointSignError) else exc.reason, error=str(exc),
                       **generation_detuning_fields(exc.records))
        except InvalidModeError as exc:
            score = -np.inf
            row.update(valid=False, rejection_reason="invalid_mode", error=str(exc))
        particles.append(row)
        _write_csv(out / "endpoint_pso_particles.csv", particles)
        if progress:
            progress.update(evaluation, detail=f"generation {generation}, particle {row['particle']}; " +
                            (f"detuning {score:.6g} rad/m" if row["valid"] else f"rejected: {row['error']}"))
        return score

    def on_generation(state):
        widths = state["best_position"]
        row = {"generation": state["generation"], "w1_input_m": widths[0], "w1_output_m": widths[1],
               "best_delta_beta_max_rad_per_m": state["best_score"],
               "valid_particles": state["generation_valid_particles"],
               "cross_state_signs_passed": True if state["best_score"] is not None else None,
               "relative_best_change": state["relative_best_change"],
               "normalized_swarm_spread": state["normalized_swarm_spread"]}
        history.append(row)
        diagnostics = dict(row)
        if state["best_score"] is not None:
            diagnostics.update(generation_detuning_fields(evaluated[tuple(widths)][1]))
        generation_rows.append(diagnostics)
        _write_csv(out / "endpoint_pso_history.csv", history)
        _write_csv(out / "endpoint_pso_generation_detunings.csv", generation_rows)
        _write_json(out / "endpoint_pso_checkpoint.json", {
            "schema_version": 1, "status": "generation_completed", "swarm": state,
            "settings": settings.__dict__, "width_bounds_m": [g.w1_min_m, g.w1_max_m],
            "isolated_mode_domain": backend.endpoint_isolated_domain_info(),
        })
        if progress:
            progress.checkpoint("python_pso_generation", diagnostics)
            progress.note(f"PSO generation {state['generation']}: best detuning {state['best_score']} rad/m", force=True)

    # Cover increasing-width pairs in the three-particle smoke run.
    # These seeds explore both signs without running a preliminary width scan.
    # Sign feasibility is tested from FDE results, never inferred from width order.
    seeds = np.array([[0., 1.], [0., .5], [.5, 1.], [.25, .75]])[:settings.pso_generation_size]
    result = run_particle_swarm(
        objective, np.array([[g.w1_min_m, g.w1_max_m]]*2), settings,
        on_generation=on_generation, initial_positions_normalized=seeds,
    )
    if progress:
        progress.finish_stage(f"{result.evaluations} particles in {result.generations} generations")
        progress.start_stage("Verify Python PSO best candidate", total=10, unit="fresh FDE mode solves")
    # Force all ten final mode calculations to bypass SQLite. Recompute signed
    # detunings and compare each case with the optimizer's saved evaluation.
    verification = {"status": "running", "fresh_fde_solves": True,
                    "relative_tolerance": 1e-6, "absolute_tolerance_rad_per_m": 1e-3}
    try:
        evaluator.precompute_lower(use_cache=False)
        score, records = evaluator.evaluate(result.position, use_cache=False)
        verification["cross_state_sign_verification"] = require_cross_state_signs(records)
        verification["endpoint_constraint_verification"] = require_endpoint_constraints(records, settings)
        expected = evaluated[tuple(result.position)][1]
        errors = [abs(a["delta_beta_real_rad_per_m"] - b["delta_beta_real_rad_per_m"])
                  for a, b in zip(records, expected)]
        matches = all(np.isclose(a["delta_beta_real_rad_per_m"], b["delta_beta_real_rad_per_m"], rtol=1e-6, atol=1e-3)
                      for a, b in zip(records, expected))
        verification.update(status="passed" if matches else "failed", optimizer_score_rad_per_m=result.score,
                            verified_score_rad_per_m=score, maximum_case_error_rad_per_m=max(errors),
                            endpoint_records=records)
        if not matches:
            raise RuntimeError("Fresh FDE endpoint detunings disagree with Python PSO; see endpoint_pso_verification.json.")
    except Exception as exc:
        if isinstance(exc, (EndpointSignError, EndpointConstraintError)):
            verification.update(cross_state_sign_verification=cross_state_sign_report(exc.records), endpoint_records=exc.records,
                                endpoint_constraint_verification=endpoint_constraint_report(exc.records, settings))
        verification.update(status="failed", error=str(exc))
        _write_json(out / "endpoint_pso_verification.json", verification)
        raise
    _write_json(out / "endpoint_pso_verification.json", verification)
    backend._block1_mode_profiles = evaluator.mode_profiles
    if progress:
        progress.finish_stage("ten fresh FDE mode solves and all eight detunings verified")
    summary = {
        "backend": "python_pso", "generation_size": settings.pso_generation_size,
        "generations_completed": result.generations, "max_generations": settings.pso_max_generations,
        "evaluations": result.evaluations, "converged": result.converged,
        "stop_reason": "tolerance" if result.converged else "generation_limit",
        "random_seed": settings.pso_random_seed, "width_bounds_m": [g.w1_min_m, g.w1_max_m],
        "width_bounds_adjusted": False, "cutoff_preflight": False,
        "lower_mode_state_independent": True, "silicon_substrate_included": False,
        "cross_state_sign_requirement": dict(CROSS_STATE_SIGN_REQUIREMENT),
        "cross_state_endpoint_signs_verified": True,
        "endpoint_constraints_verified": True,
        "endpoint_constraint_verification": require_endpoint_constraints(records, settings),
        "settings": settings.__dict__,
    }
    _write_json(out / "endpoint_pso_best.json", dict(
        summary, w1_input_m=float(result.position[0]), w1_output_m=float(result.position[1]),
        delta_beta_max_rad_per_m=score, endpoint_records=records,
    ))
    return EndpointSearchResult(
        h_gap_m=g.h_gap_bounds_m[0], h_rib_m=g.h_rib_bounds_m[0],
        w1_input_m=float(result.position[0]), w1_output_m=float(result.position[1]),
        delta_beta_max_rad_per_m=score, wavelengths_m=np.asarray(evaluator.wavelengths),
        endpoint_records=records, candidate_records=particles, converged=result.converged,
        refinement_history=history, isolated_mode_domain=backend.endpoint_isolated_domain_info(),
        search_method="python_pso", pso_summary=summary,
    )
