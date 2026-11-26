"""Endpoint detuning search with fixed band-edge wavelength samples."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np

from .config import WorkflowConfig
from .endpoint_constraints import cross_state_sign_report, endpoint_constraint_report, require_endpoint_constraints
from .geometry import make_geometry
from .materials import MaterialBank
from .modes import ModeSnapshot


@dataclass
class EndpointSearchResult:
    h_gap_m: float
    h_rib_m: float
    w1_input_m: float
    w1_output_m: float
    delta_beta_max_rad_per_m: float
    wavelengths_m: np.ndarray
    endpoint_records: list[dict]
    candidate_records: list[dict]
    converged: bool
    refinement_history: list[dict]
    isolated_mode_domain: dict
    search_method: str = "python_grid"
    pso_summary: dict | None = None
    search_summary: dict | None = None


def _refine_axis(values: list[float], best: float, bounds: tuple[float, float]) -> list[float]:
    current = sorted(set(float(v) for v in values))
    if len(current) < 2:
        return current
    index = int(np.argmin(np.abs(np.asarray(current) - best)))
    additions = []
    if index > 0:
        additions.append((current[index - 1] + current[index]) / 2)
    if index + 1 < len(current):
        additions.append((current[index] + current[index + 1]) / 2)
    lo, hi = bounds
    return sorted(set(current + [v for v in additions if lo <= v <= hi]))


def _normalized_change(a: float, b: float, bounds: tuple[float, float]) -> float:
    span = bounds[1] - bounds[0]
    if span <= 0:
        return 0.0 if a == b else float("inf")
    return abs(a - b) / span


def search_endpoints_grid(
    backend, config: WorkflowConfig, bank: MaterialBank, *, progress=None
) -> EndpointSearchResult:
    """Maximize endpoint detuning with negative cross input, positive cross output and balanced magnitudes.

    h_gap and shared h_rib are fixed by configuration. The two upper-guide
    endpoint widths are refined on a bounded grid and held fixed during profile
    optimization. Per the configured design assumption, the endpoint metric is
    sampled only at the two wavelength-band bounds.
    """
    g, d = config.geometry, config.device
    hgap_values = list(g.h_gap_candidates_m)
    hrib_values = list(g.h_rib_candidates_m)
    w_values = list(g.w1_endpoint_candidates_m)
    for value in w_values:
        if not g.w1_min_m <= value <= g.w1_max_m:
            raise ValueError(f"Endpoint width {value} is outside the w1 bounds.")
    for value in hgap_values:
        if not g.h_gap_bounds_m[0] <= value <= g.h_gap_bounds_m[1]:
            raise ValueError(f"h_gap candidate {value} is outside its global bounds.")
    for value in hrib_values:
        if not g.h_rib_bounds_m[0] <= value <= g.h_rib_bounds_m[1]:
            raise ValueError(f"h_rib candidate {value} is outside its global bounds.")

    wavelengths = np.asarray([d.wavelength_min_m, d.wavelength_max_m], dtype=float)
    best_candidate = None
    best_endpoints = []
    candidate_records: list[dict] = []
    history: list[dict] = []
    converged = False
    candidates_completed = 0
    candidates_planned_count = 0
    candidate_work_completed = 0
    candidate_work_planned = 0

    for refinement in range(config.optimizer.max_grid_refinements + 1):
        candidate_records = []
        refinement_candidates = (
            len(hgap_values) * len(hrib_values) * len(w_values) * len(w_values)
        )
        refinement_work = refinement_candidates
        candidate_work_planned += refinement_work
        candidates_planned_count += refinement_candidates
        if progress:
            if refinement == 0:
                progress.start_stage(
                    "Endpoint detuning search",
                    total=candidate_work_planned,
                    unit="endpoint candidates",
                )
            else:
                progress.update(
                    candidate_work_completed,
                    total=candidate_work_planned,
                    detail=f"starting geometry refinement {refinement + 1}; using band bounds only",
                )
        for hgap, hrib, w_input, w_output in product(
            hgap_values, hrib_values, w_values, w_values
        ):
            bottleneck_by_lambda = np.full(len(wavelengths), np.inf)
            records: list[dict] = []
            for state in ("cross", "bar"):
                liquid = bank.liquid_cross if state == "cross" else bank.liquid_bar
                refs_by_end: dict[str, dict[str, ModeSnapshot | None]] = {
                    "input": {"upper": None, "lower": None},
                    "output": {"upper": None, "lower": None},
                }
                for iw, wavelength in enumerate(wavelengths):
                    for end, width, offset in (
                        ("input", w_input, g.d_c2c_input_m),
                        ("output", w_output, g.d_c2c_output_m),
                    ):
                        geometry = make_geometry(
                            g,
                            w1_m=width,
                            h_rib_m=hrib,
                            h_gap_m=hgap,
                            d_c2c_m=offset,
                            state=state,
                            liquid_index=float(liquid.index(wavelength)),
                        )
                        previous = refs_by_end[end]
                        upper, upper_overlap = backend.solve_isolated_mode(
                            geometry,
                            float(wavelength),
                            guide="upper",
                            reference=previous["upper"],
                            compact_x=config.solver.endpoint_compact_x,
                        )
                        lower, lower_overlap = backend.solve_isolated_mode(
                            geometry,
                            float(wavelength),
                            guide="lower",
                            reference=previous["lower"],
                            compact_x=config.solver.endpoint_compact_x,
                        )
                        refs_by_end[end]["upper"] = upper
                        refs_by_end[end]["lower"] = lower
                        scale = 2 * np.pi / wavelength
                        beta_upper = scale * upper.neff
                        beta_lower = scale * lower.neff
                        signed = beta_upper - beta_lower
                        bottleneck_by_lambda[iw] = min(
                            bottleneck_by_lambda[iw], abs(signed.real)
                        )
                        records.append(
                            {
                                "h_gap_m": hgap,
                                "h_rib_m": hrib,
                                "w1_m": width,
                                "end": end,
                                "d_c2c_m": offset,
                                "state": state,
                                "wavelength_m": float(wavelength),
                                "beta_upper_real_rad_per_m": float(beta_upper.real),
                                "beta_upper_imag_rad_per_m": float(beta_upper.imag),
                                "beta_lower_real_rad_per_m": float(beta_lower.real),
                                "beta_lower_imag_rad_per_m": float(beta_lower.imag),
                                "delta_beta_real_rad_per_m": float(signed.real),
                                "delta_beta_imag_rad_per_m": float(signed.imag),
                                "upper_mode_identity": upper.identity,
                                "lower_mode_identity": lower.identity,
                                "upper_te_fraction": upper.te_fraction,
                                "lower_te_fraction": lower.te_fraction,
                                "upper_tracking_overlap": upper_overlap,
                                "lower_tracking_overlap": lower_overlap,
                            }
                        )
            signs = cross_state_sign_report(records)
            constraints = endpoint_constraint_report(records, config.endpoint_search)
            feasible = constraints["passed"]
            bottleneck = float(np.min(bottleneck_by_lambda))
            candidate = {
                "h_gap_m": hgap,
                "h_rib_m": hrib,
                "w1_input_m": w_input,
                "w1_output_m": w_output,
                "valid": feasible,
                "cross_state_signs_passed": signs["passed"],
                "endpoint_constraints_passed": feasible,
                "maximum_relative_mismatch": constraints["magnitude_balance"]["maximum_relative_mismatch"],
                "rejection_reason": "" if feasible else "endpoint_constraints",
                "bottleneck_rad_per_m": bottleneck if feasible else None,
                "bottleneck_by_wavelength_rad_per_m": bottleneck_by_lambda.tolist(),
                "bottleneck_wavelength_m": float(wavelengths[int(np.argmin(bottleneck_by_lambda))]),
            }
            candidate_records.append(candidate)
            if feasible and (best_candidate is None or bottleneck > best_candidate["bottleneck_rad_per_m"]):
                best_candidate = candidate
                best_endpoints = records
            candidates_completed += 1
            candidate_work_completed += 1
            if progress:
                progress.update(
                    candidate_work_completed,
                    total=candidate_work_planned,
                    detail=(
                        f"{candidates_completed}/{candidates_planned_count} candidates; "
                        f"refinement {refinement + 1}, "
                        + (f"best bottleneck {best_candidate['bottleneck_rad_per_m']:.4g} rad/m"
                           if best_candidate is not None else "no sign-feasible candidate yet")
                    ),
                )

        if best_candidate is None:
            raise RuntimeError(
                "Endpoint grid search found no candidate satisfying the cross-state "
                "endpoint sign restriction (input < 0, output > 0 at both band edges), increasing width and magnitude balance. "
                "Width bounds were not changed."
            )
        history.append(
            {
                "refinement": refinement,
                "wavelength_count": len(wavelengths),
                "wavelengths_m": wavelengths.tolist(),
                "h_gap_candidate_count": len(hgap_values),
                "h_rib_candidate_count": len(hrib_values),
                "w1_endpoint_candidate_count": len(w_values),
                "wavelength_policy": "band_edges_only_by_design_assumption",
                "best": best_candidate,
            }
        )
        if refinement == config.optimizer.max_grid_refinements:
            break
        previous_best = best_candidate
        next_hgap_values = _refine_axis(
            hgap_values, previous_best["h_gap_m"], g.h_gap_bounds_m
        )
        next_hrib_values = _refine_axis(
            hrib_values, previous_best["h_rib_m"], g.h_rib_bounds_m
        )
        next_w_values = _refine_axis(
            w_values,
            previous_best["w1_input_m"],
            (g.w1_min_m, g.w1_max_m),
        )
        next_w_values = _refine_axis(
            next_w_values,
            previous_best["w1_output_m"],
            (g.w1_min_m, g.w1_max_m),
        )
        geometry_grid_changed = (
            next_hgap_values != hgap_values
            or next_hrib_values != hrib_values
            or next_w_values != w_values
        )
        if not geometry_grid_changed:
            break
        hgap_values = next_hgap_values
        hrib_values = next_hrib_values
        w_values = next_w_values
        best_candidate = None

    if progress:
        progress.finish_stage(
            f"{candidates_completed} endpoint candidates evaluated"
        )

    if best_candidate is None:
        raise RuntimeError("Endpoint search found no valid geometry candidate.")
    # Compare the last two geometry-grid refinements. The wavelength samples are
    # fixed at the band edges by design and are not refined by this search.
    if len(history) > 1:
        previous = history[-2]["best"]
        current = history[-1]["best"]
        a = previous["bottleneck_rad_per_m"]
        b = current["bottleneck_rad_per_m"]
        relative_objective_change = abs(b - a) / max(abs(b), 1e-30)
        normalized_geometry_change = max(
            _normalized_change(
                current["h_gap_m"], previous["h_gap_m"], g.h_gap_bounds_m
            ),
            _normalized_change(
                current["h_rib_m"], previous["h_rib_m"], g.h_rib_bounds_m
            ),
            _normalized_change(
                current["w1_input_m"], previous["w1_input_m"],
                (g.w1_min_m, g.w1_max_m),
            ),
            _normalized_change(
                current["w1_output_m"], previous["w1_output_m"],
                (g.w1_min_m, g.w1_max_m),
            ),
        )
        history[-1]["relative_objective_change"] = relative_objective_change
        history[-1]["normalized_geometry_change"] = normalized_geometry_change
        converged = (
            relative_objective_change <= d.grid_convergence_rtol
            and normalized_geometry_change <= d.grid_convergence_rtol
        )
    require_endpoint_constraints(best_endpoints, config.endpoint_search)
    return EndpointSearchResult(
        h_gap_m=best_candidate["h_gap_m"],
        h_rib_m=best_candidate["h_rib_m"],
        w1_input_m=best_candidate["w1_input_m"],
        w1_output_m=best_candidate["w1_output_m"],
        delta_beta_max_rad_per_m=best_candidate["bottleneck_rad_per_m"],
        wavelengths_m=wavelengths,
        endpoint_records=best_endpoints,
        candidate_records=candidate_records,
        converged=converged,
        refinement_history=history,
        isolated_mode_domain=backend.endpoint_isolated_domain_info(
            compact_x=config.solver.endpoint_compact_x
        ),
        search_method="python_grid",
    )


def search_endpoints(
    backend, config: WorkflowConfig, bank: MaterialBank, *, progress=None, output_dir=None
) -> EndpointSearchResult:
    """Dispatch to the configured Python endpoint search."""
    if config.endpoint_search.method == "adaptive_1d":
        from .adaptive_endpoint import search_endpoints_adaptive
        return search_endpoints_adaptive(backend, config, bank, progress=progress, output_dir=output_dir)
    if config.endpoint_search.method == "python_pso":
        from .python_pso import search_endpoints_python_pso

        return search_endpoints_python_pso(
            backend, config, bank, progress=progress, output_dir=output_dir
        )
    return search_endpoints_grid(backend, config, bank, progress=progress)
