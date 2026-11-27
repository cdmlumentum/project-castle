"""CSV, plot, configuration and metadata exports for completed MODE runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import WorkflowConfig, save_config
from .endpoint import EndpointSearchResult
from .endpoint_constraints import require_cross_state_signs, require_endpoint_constraints
from .geometry import solution_profiles, blackman_theta
from .materials import MaterialBank, material_report
from .modes import Coefficients
from .profiles import ProfileSolution, target_profiles


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_default(value):
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _coefficient_rows(
    solution: ProfileSolution,
    bar_coefficients: list[list[Coefficients]],
    endpoint: EndpointSearchResult,
    config: WorkflowConfig,
) -> list[dict]:
    rows = []
    w_profile, d_profile = solution_profiles(config, solution)
    for state, grid in (
        ("cross", solution.cross_coefficients),
        ("bar", bar_coefficients),
    ):
        if grid is None:
            continue
        for iw, wavelength in enumerate(solution.wavelengths_m):
            for iz, z in enumerate(solution.z_m):
                c = grid[iw][iz]
                delta_target = float(solution.delta_target_rad_per_m[iz]) if state == "cross" else float("nan")
                kappa_target = float(solution.kappa_target_rad_per_m[iz]) if state == "cross" else float("nan")
                rows.append(
                    {
                        "state": state,
                        "wavelength_m": wavelength,
                        "wavelength_nm": wavelength * 1e9,
                        "z_m": z,
                        "z_um": z * 1e6,
                        "w1_m": float(w_profile(z)),
                        "w1_um": float(w_profile(z) * 1e6),
                        "d_c2c_m": float(d_profile(z)),
                        "d_c2c_um": float(d_profile(z) * 1e6),
                        "h_gap_m": endpoint.h_gap_m,
                        "h_rib_m": endpoint.h_rib_m,
                        "delta_beta_real_rad_per_m": c.delta_beta.real,
                        "delta_beta_imag_rad_per_m": c.delta_beta.imag,
                        "delta_beta_target_rad_per_m": delta_target,
                        "delta_beta_residual_rad_per_m": (
                            c.delta_beta.real - delta_target
                            if state == "cross" else float("nan")
                        ),
                        "kappa_real_rad_per_m": c.kappa.real,
                        "kappa_imag_rad_per_m": c.kappa.imag,
                        "kappa_profile_check_rad_per_m": c.kappa_profile_check,
                        "kappa_target_rad_per_m": kappa_target,
                        "kappa_residual_real_rad_per_m": (
                            c.kappa.real - kappa_target
                            if state == "cross" else float("nan")
                        ),
                        "kappa_residual_imag_rad_per_m": (
                            c.kappa.imag if state == "cross" else float("nan")
                        ),
                        "beta_upper_real_rad_per_m": c.beta_upper.real,
                        "beta_upper_imag_rad_per_m": c.beta_upper.imag,
                        "beta_lower_real_rad_per_m": c.beta_lower.real,
                        "beta_lower_imag_rad_per_m": c.beta_lower.imag,
                        "delta_pm_real_rad_per_m": c.delta_pm.real,
                        "delta_pm_imag_rad_per_m": c.delta_pm.imag,
                        "delta_12_real_rad_per_m": c.delta_12.real,
                        "delta_12_imag_rad_per_m": c.delta_12.imag,
                        "isolated_mode_overlap": c.overlap_upper_lower,
                        "minimum_tracking_overlap": c.min_mode_overlap,
                        "upper_mode_identity": c.upper_identity,
                        "lower_mode_identity": c.lower_identity,
                        "supermode_plus_identity": c.supermode_identities[0],
                        "supermode_minus_identity": c.supermode_identities[1],
                        "extraction_flag": c.extraction_flag,
                    }
                )
    return rows


def _plot_profiles(
    out: Path,
    solution: ProfileSolution,
    bar_coefficients: list[list[Coefficients]],
    endpoint: EndpointSearchResult,
    config: WorkflowConfig,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(solution.wavelengths_m)))
    for state, grid, style in (
        ("cross", solution.cross_coefficients, "-"),
        ("bar", bar_coefficients, "--"),
    ):
        if grid is None:
            continue
        for iw, wavelength in enumerate(solution.wavelengths_m):
            delta = [v.delta_beta.real for v in grid[iw]]
            kappa = [v.kappa.real for v in grid[iw]]
            label = f"{state}, {wavelength*1e9:.0f} nm"
            axes[0, 0].plot(solution.z_m * 1e6, delta, style, color=colors[iw], label=label)
            axes[0, 1].plot(solution.z_m * 1e6, kappa, style, color=colors[iw], label=label)
    axes[0, 0].plot(
        solution.z_m * 1e6,
        solution.delta_target_rad_per_m,
        color="black",
        linewidth=2,
        label="cross target",
    )
    axes[0, 1].plot(
        solution.z_m * 1e6,
        solution.kappa_target_rad_per_m,
        color="black",
        linewidth=2,
        label="cross target",
    )
    axes[0, 0].set(title="Signed detuning", xlabel="z (µm)", ylabel="Re(Δβ) (rad/m)")
    axes[0, 1].set(title="Coupling coefficient", xlabel="z (µm)", ylabel="Re(κ) (rad/m)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 1].legend(fontsize=7, ncol=2)
    w, dc = solution_profiles(config, solution)
    dense_z = np.linspace(0, config.device.length_m, 501)
    axes[1, 0].plot(dense_z * 1e6, w(dense_z) * 1e6, color="tab:blue")
    axes[1, 0].set(title="Upper-guide width", xlabel="z (µm)", ylabel="w1 (µm)")
    axes[1, 1].plot(dense_z * 1e6, dc(dense_z) * 1e6, color="tab:orange")
    axes[1, 1].axhline(config.geometry.d_c2c_input_m * 1e6, color="gray", linewidth=0.5)
    axes[1, 1].axhline(config.geometry.d_c2c_output_m * 1e6, color="gray", linewidth=0.5)
    axes[1, 1].set(title="Lateral center offset", xlabel="z (µm)", ylabel="d_c2c (µm)")
    fig.suptitle(
        f"Local FDE coefficient profiles; minimax J={solution.objective:.4g}, "
        f"Δβmax={endpoint.delta_beta_max_rad_per_m:.4g} rad/m"
    )
    fig.savefig(out / "profiles_and_coefficients.png", dpi=180)
    plt.close(fig)


def _plot_performance(out: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for state, color in (("cross", "tab:blue"), ("bar", "tab:orange")):
        state_rows = [r for r in rows if r["state"] == state]
        for launch in ("upper", "lower"):
            selected = sorted(
                [r for r in state_rows if r["launch_layer"] == launch],
                key=lambda r: r["wavelength_nm"],
            )
            suffix = "" if launch == "upper" else " (lower launch)"
            axes[0].plot(
                [r["wavelength_nm"] for r in selected],
                [r["il_db"] for r in selected],
                marker="o",
                color=color,
                linestyle="-" if launch == "upper" else "--",
                label=state + suffix,
            )
            axes[1].plot(
                [r["wavelength_nm"] for r in selected],
                [r["xt_db"] for r in selected],
                marker="o",
                color=color,
                linestyle="-" if launch == "upper" else "--",
                label=state + suffix,
            )
    axes[0].set(title="Insertion loss", xlabel="Wavelength (nm)", ylabel="IL (dB)")
    axes[1].set(title="Crosstalk", xlabel="Wavelength (nm)", ylabel="XT (dB)")
    axes[0].legend()
    axes[1].legend()
    fig.savefig(out / "wavelength_performance.png", dpi=180)
    plt.close(fig)


def export_run(
    out_dir: str | Path,
    config: WorkflowConfig,
    bank: MaterialBank,
    endpoint: EndpointSearchResult,
    solution: ProfileSolution,
    bar_coefficients: list[list[Coefficients]],
    performance_rows: list[dict],
    *,
    backend_name: str,
    fde_run_completed: bool,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config(config, out / "configuration.json")
    _write_csv(
        out / "endpoint_search.csv",
        list(endpoint.endpoint_records[0].keys()),
        endpoint.endpoint_records,
    )
    _write_csv(
        out / "endpoint_candidates.csv",
        list(dict.fromkeys(key for row in endpoint.candidate_records for key in row)),
        endpoint.candidate_records,
    )

    w_profile, dc_profile = solution_profiles(config, solution)
    theta = blackman_theta(solution.z_m, config.device.length_m)
    _write_csv(
        out / "optimized_profiles.csv",
        [
            "z_m", "z_um", "theta_rad", "w1_m", "w1_um", "d_c2c_m",
            "d_c2c_um", "h_gap_m", "h_rib_m", "length_m",
            "delta_beta_target_rad_per_m", "kappa_target_rad_per_m",
        ],
        [
            {
                "z_m": z,
                "z_um": z * 1e6,
                "theta_rad": theta[i],
                "w1_m": w_profile(z),
                "w1_um": w_profile(z) * 1e6,
                "d_c2c_m": dc_profile(z),
                "d_c2c_um": dc_profile(z) * 1e6,
                "h_gap_m": endpoint.h_gap_m,
                "h_rib_m": endpoint.h_rib_m,
                "length_m": config.device.length_m,
                "delta_beta_target_rad_per_m": solution.delta_target_rad_per_m[i],
                "kappa_target_rad_per_m": solution.kappa_target_rad_per_m[i],
            }
            for i, z in enumerate(solution.z_m)
        ],
    )

    coefficient_rows = _coefficient_rows(solution, bar_coefficients, endpoint, config)
    _write_csv(
        out / "coefficients.csv",
        list(coefficient_rows[0].keys()),
        coefficient_rows,
    )
    _write_csv(
        out / "performance_by_launch.csv",
        list(performance_rows[0].keys()),
        performance_rows,
    )
    summary_rows = []
    for state in ("cross", "bar"):
        wavelengths = sorted(
            {r["wavelength_nm"] for r in performance_rows if r["state"] == state}
        )
        for wavelength in wavelengths:
            selected = [
                r for r in performance_rows
                if r["state"] == state and r["wavelength_nm"] == wavelength
            ]
            summary_rows.append(
                {
                    "state": state,
                    "wavelength_nm": wavelength,
                    "il_worst_launch_db": max(r["il_db"] for r in selected),
                    "xt_worst_launch_db": max(r["xt_db"] for r in selected),
                    "minimum_guided_output_power": min(
                        r["guided_output_power"] for r in selected
                    ),
                    "maximum_guided_output_power": max(
                        r["guided_output_power"] for r in selected
                    ),
                }
            )
    _write_csv(
        out / "performance_summary.csv",
        [
            "state", "wavelength_nm", "il_worst_launch_db",
            "xt_worst_launch_db", "minimum_guided_output_power",
            "maximum_guided_output_power",
        ],
        summary_rows,
    )
    _plot_profiles(out, solution, bar_coefficients, endpoint, config)
    _plot_performance(out, performance_rows)

    (out / "endpoint_refinement.json").write_text(
        json.dumps(endpoint.refinement_history, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (out / "optimizer_history.json").write_text(
        json.dumps(
            {
                "objective_definition": (
                    "J=max_{z,lambda}(abs(Re(Delta_beta_cross)-Delta_beta_target)/"
                    "Delta_beta_max, abs(kappa_cross-kappa_target)/kappa_max)"
                ),
                "objective_achieved": solution.objective,
                "grid_converged": solution.grid_converged,
                "grid_history": solution.grid_convergence_history,
                "optimizer_history": solution.optimizer_history,
                "endpoint_search_converged": endpoint.converged,
            },
            indent=2,
            default=_json_default,
        ) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "command": "vlac run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name,
        "fde_run_completed": fde_run_completed,
        "surrogate_device_results": False,
        "units": {
            "length": "m (CSV also includes selected um and nm columns)",
            "wavelength": "m (CSV also includes nm)",
            "effective_index": "dimensionless",
            "propagation_constants_and_coupling": "rad/m",
            "theta": "rad",
            "power": "input normalized to one",
            "il_xt": "dB",
        },
        "material_models": material_report(
            bank, solution.wavelengths_m
        ),
        "material_assumptions": {
            "silicon": "Green (2008) data is retained for table validation only; no silicon material or substrate region is installed in MODE",
            "sin": "Luke et al. (2015) LPCVD Sellmeier is provisional; no process-matched film evidence supplied",
            "sin_k": "zero unless an explicit extinction CSV is configured",
            "liquid": (
                f"constant n_cross={config.materials.liquid_cross_n:g} and n_bar={config.materials.liquid_bar_n:g}"
                if config.materials.liquid_dispersion_csv is None
                else f"tabulated liquid dispersion from {config.materials.liquid_dispersion_csv}"
            ),
            "roughness_scattering": "excluded; requires measured/calibrated inputs",
        },
        "geometry_assumptions": {
            "h_rib_shared_by_both_guides": config.geometry.upper_and_lower_use_same_h_rib,
            "residual_slab_lateral_extent": "both FDE x boundaries",
            "residual_slabs_centered_at_x_m": 0.0,
            "box_thickness_m": config.geometry.box_thickness_m,
            "silicon_substrate_in_fde_domain": False,
            "bottom_oxide_margin_m": config.solver.domain_margin_y_bottom_m,
            "w1_endpoint_widths_fixed_after_endpoint_search": True,
        },
        "solver_settings": config.solver.__dict__,
        "optimizer_settings": config.optimizer.__dict__,
        "propagation_settings": config.propagation.__dict__,
        "selected_global_geometry": {
            "h_gap_m": endpoint.h_gap_m,
            "h_rib_m": endpoint.h_rib_m,
            "w1_input_m": endpoint.w1_input_m,
            "w1_output_m": endpoint.w1_output_m,
            "Delta_beta_max_rad_per_m": endpoint.delta_beta_max_rad_per_m,
            "kappa_max_rad_per_m": 0.5 * endpoint.delta_beta_max_rad_per_m,
        },
        "endpoint_search_assumption": {
            "wavelength_policy": "band_edges_only_by_design_assumption",
            "wavelengths_m": endpoint.wavelengths_m.tolist(),
            "search_method": endpoint.search_method,
            "cross_state_sign_verification": require_cross_state_signs(endpoint.endpoint_records),
            "endpoint_constraint_verification": require_endpoint_constraints(endpoint.endpoint_records, config.endpoint_search),
            "h_gap_fixed_m": endpoint.h_gap_m,
            "h_rib_fixed_m": endpoint.h_rib_m,
            "search_converged": endpoint.converged,
            "detuning_method": "individual isolated upper and lower guide modes",
            "lateral_offset_invariant_for_isolated_modes": True,
            "compact_x_domain": config.solver.endpoint_compact_x,
            "lower_mode_state_independence_assumption": (
                "cross-state lower-guide modes computed once per band edge and reused in both liquid states"
                if endpoint.search_method in {"python_pso", "adaptive_1d"}
                else "lower-guide mode solved in each liquid-state sample"
            ),
            "pso_summary": endpoint.pso_summary,
            "search_summary": endpoint.search_summary or endpoint.pso_summary,
        },
        "endpoint_isolated_mode_domain": endpoint.isolated_mode_domain,
        "source_references": [
            "D'heer et al., JLT 35, 2948-2954 (2017), DOI 10.1109/JLT.2017.2705284",
            "Ishikawa, JLT 25, 1832-1840 (2007), DOI 10.1109/JLT.2007.899186",
            "Malitson, JOSA 55, 1205 (1965), DOI 10.1364/JOSA.55.001205",
            "Green, Sol. Energy Mater. Sol. Cells 92, 1305-1310 (2008), DOI 10.1016/j.solmat.2008.06.009",
            "Luke et al., Opt. Lett. 40, 4823-4826 (2015), DOI 10.1364/OL.40.004823",
        ],
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
