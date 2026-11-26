"""Independent numerical checks and solver-dependent validation hooks."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .config import WorkflowConfig
from .geometry import (
    ClampedCubicProfile,
    blackman_theta,
    make_geometry,
    rectangles_for_cross_section,
)
from .materials import build_material_bank, material_report
from .modes import BareModeReference, ModeReference, ModeSnapshot, extract_coupling
from .propagate import propagate_converged, propagate_rk4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_material_models(config: WorkflowConfig) -> dict:
    wavelengths = np.linspace(
        config.device.wavelength_min_m, config.device.wavelength_max_m, 17
    )
    bank = build_material_bank(config)
    report = material_report(bank, wavelengths)
    for name, values in report.items():
        _require(np.all(np.isfinite(values["n_at_sampled_wavelengths"])), f"{name} n is finite")
        _require(np.all(np.isfinite(values["k_at_sampled_wavelengths"])), f"{name} k is finite")
        _require(np.all(np.asarray(values["n_at_sampled_wavelengths"]) > 1), f"{name} n > 1")
        _require(np.all(np.asarray(values["k_at_sampled_wavelengths"]) >= 0), f"{name} k >= 0")
    _require(np.min(report["silicon"]["k_at_sampled_wavelengths"]) > 0, "Green Si k is present")
    return {
        "status": "passed",
        "wavelengths_nm": (wavelengths * 1e9).tolist(),
        "models": report,
        "unit_check": "all wavelength configuration fields are SI metres",
    }


def validate_coupling_extraction() -> dict:
    # Known real 2x2 coupled-mode matrix with unequal uncoupled guides.
    beta1, beta2, kappa_true = 1.02e7, 0.98e7, 1.7e4
    mean = (beta1 + beta2) / 2
    q = np.sqrt(((beta1 - beta2) / 2) ** 2 + kappa_true**2)
    extracted = extract_coupling(
        beta1, beta2, mean + q, mean - q
    )
    _require(
        abs(extracted.kappa.real - kappa_true) / kappa_true < 1e-10,
        "Dissimilar-guide kappa extraction failed.",
    )
    # Identical guides: supermode splitting is exactly 2*kappa.
    identical = extract_coupling(
        1.0e7, 1.0e7, 1.0e7 + kappa_true, 1.0e7 - kappa_true
    )
    _require(abs(identical.delta_beta) < 1e-12, "Identical-guide detuning is not zero.")
    _require(abs(identical.kappa.real - kappa_true) / kappa_true < 1e-10, "Identical-guide extraction failed.")
    # Large separation limit: as coupling tends to zero, splitting approaches |delta12|.
    tiny = 100.0
    delta12 = (beta1 - beta2) / 2
    far_q = np.sqrt(delta12**2 + tiny**2)
    far = extract_coupling(beta1, beta2, mean + far_q, mean - far_q)
    _require(abs(far.kappa.real - tiny) / tiny < 1e-6, "Near-zero-coupling limit failed.")
    return {
        "status": "passed",
        "dissimilar_guide_kappa_true_rad_per_m": kappa_true,
        "dissimilar_guide_kappa_extracted_rad_per_m": extracted.kappa.real,
        "identical_guide_kappa_extracted_rad_per_m": identical.kappa.real,
        "near_zero_coupling_extracted_rad_per_m": far.kappa.real,
        "note": "Analytic matrix checks only; not FDE results.",
    }


def validate_geometry_profiles(config: WorkflowConfig) -> dict:
    g, d = config.geometry, config.device
    z = np.linspace(0, d.length_m, 101)
    theta = blackman_theta(z, d.length_m)
    _require(abs(theta[0]) < 1e-14, "Blackman angle does not start at zero.")
    _require(abs(theta[-1] - np.pi) < 1e-14, "Blackman angle does not end at pi.")
    w_control = np.linspace(g.w1_min_m, g.w1_max_m, 6)
    d_control = np.linspace(g.d_c2c_input_m, g.d_c2c_output_m, 6)
    w_profile = ClampedCubicProfile(d.length_m, w_control)
    dc_profile = ClampedCubicProfile(d.length_m, d_control)
    _require(w_profile(0.0) == w_control[0], "w1 input endpoint is not exact.")
    _require(w_profile(d.length_m) == w_control[-1], "w1 output endpoint is not exact.")
    _require(dc_profile(0.0) == g.d_c2c_input_m, "d_c2c input endpoint is not exact.")
    _require(dc_profile(d.length_m) == g.d_c2c_output_m, "d_c2c output endpoint is not exact.")
    _require(
        np.min(w_profile(z)) >= g.w1_min_m - 1e-18
        and np.max(w_profile(z)) <= g.w1_max_m + 1e-18,
        "Spline upper-guide width escaped its bounds.",
    )
    _require(
        np.min(dc_profile(z)) >= g.d_c2c_bounds_m[0] - 1e-18
        and np.max(dc_profile(z)) <= g.d_c2c_bounds_m[1] + 1e-18,
        "Spline lateral offset escaped its bounds.",
    )
    hrib = float(np.mean(g.h_rib_bounds_m))
    hgap = float(np.mean(g.h_gap_bounds_m))
    geom = make_geometry(
        g,
        w1_m=float((g.w1_min_m + g.w1_max_m) / 2),
        h_rib_m=hrib,
        h_gap_m=hgap,
        d_c2c_m=g.d_c2c_input_m,
        state="cross",
        liquid_index=config.materials.liquid_cross_n,
    )
    rects = rectangles_for_cross_section(
        geom,
        x_span_m=30e-6,
        y_min_m=-g.box_thickness_m-config.solver.domain_margin_y_bottom_m,
        y_max_m=5e-6,
    )
    bottom_oxide = next(r for r in rects if r.name == "buried_oxide_and_gap")
    expected_bottom = -g.box_thickness_m - config.solver.domain_margin_y_bottom_m
    _require(
        bottom_oxide.material == "SiO2_Malitson_1965"
        and abs(bottom_oxide.y_center_m - bottom_oxide.y_span_m / 2 - expected_bottom) < 1e-15
        and abs(bottom_oxide.y_center_m + bottom_oxide.y_span_m / 2 - geom.upper_bottom_y_m) < 1e-15,
        "SiO2 host region does not fill the bottom-oxide domain to the upper guide.",
    )
    _require(
        not any("silicon" in rect.name.lower() or rect.material.startswith("Si_Green") for rect in rects),
        "Silicon substrate material remains in the MODE cross section.",
    )
    for slab in (r for r in rects if r.name in {"upper_slab", "lower_slab"}):
        _require(abs(slab.x_center_m) < 1e-15 and abs(slab.x_span_m-30e-6) < 1e-15,
                 f"{slab.name} does not span both FDE lateral boundaries.")
    return {
        "status": "passed",
        "blackman_theta_start_rad": float(theta[0]),
        "blackman_theta_end_rad": float(theta[-1]),
        "w1_bounds_m": [g.w1_min_m, g.w1_max_m],
        "d_c2c_bounds_m": list(g.d_c2c_bounds_m),
        "d_c2c_exact_endpoints_m": [
            float(dc_profile(0.0)), float(dc_profile(d.length_m))
        ],
        "shared_h_rib_m": hrib,
        "residual_slab_lateral_extent": "both FDE x boundaries",
        "box_thickness_m": g.box_thickness_m,
        "bottom_oxide_margin_m": config.solver.domain_margin_y_bottom_m,
        "silicon_substrate_included": False,
    }


def validate_propagation() -> dict:
    length = 1.0e-3
    z = np.array([0.0, length])
    steps = 800

    # Uniform, phase-matched coupler: kappa*L=pi/2 gives complete transfer.
    kappa = np.full(2, np.pi / (2 * length), dtype=complex)
    lossless = propagate_rk4(
        z,
        np.zeros(2, dtype=complex),
        np.zeros(2, dtype=complex),
        kappa,
        launch_layer=0,
        steps=steps,
    )
    _require(abs(lossless.output_powers[1] - 1) < 2e-8, "Uniform coupler transfer check failed.")
    _require(abs(lossless.total_guided_power - 1) < 2e-8, "Lossless power conservation failed.")

    # Decoupled lossy mode: Im(beta)=alpha/2, so guided power=e^(-alpha*L).
    alpha = 7.0
    beta = np.full(2, 0.5j * alpha, dtype=complex)
    uncoupled = propagate_rk4(
        z,
        beta,
        beta,
        np.zeros(2, dtype=complex),
        launch_layer=0,
        steps=steps,
    )
    expected = np.exp(-alpha * length)
    _require(abs(uncoupled.output_powers[0] - expected) < 2e-10, "Absorption power accounting failed.")

    # Halving-step convergence for a nonuniform analytic profile.
    z_dense = np.linspace(0, length, 11)
    b1 = np.full_like(z_dense, 1.0e5, dtype=complex)
    b2 = np.full_like(z_dense, -1.0e5, dtype=complex)
    kap = 2.4e4 * np.sin(np.pi * z_dense / length)
    settings = type("Settings", (), {
        "rk4_steps_initial": 100,
        "rk4_steps_max": 12800,
        "power_convergence_rtol": 1e-6,
    })()
    converged = propagate_converged(
        z_dense, b1, b2, kap, launch_layer=0, settings=settings
    )
    _require(converged.converged, "RK4 step-halving convergence did not meet tolerance.")
    return {
        "status": "passed",
        "uniform_coupler_cross_power": float(lossless.output_powers[1]),
        "uniform_coupler_total_power": lossless.total_guided_power,
        "lossy_decoupled_power": float(uncoupled.output_powers[0]),
        "lossy_decoupled_expected_power": float(expected),
        "nonuniform_rk4_steps": converged.steps,
        "nonuniform_rk4_relative_change": converged.relative_change,
    }


def run_analytic_validation(config: WorkflowConfig, *, progress=None) -> dict:
    checks = (
        ("material_models_and_units", "Material models and units", validate_material_models),
        ("geometry_and_spline_constraints", "Geometry and spline constraints", validate_geometry_profiles),
        ("coupling_extraction_limits", "Analytic coupling extraction", lambda _config: validate_coupling_extraction()),
        ("coupled_amplitude_and_power", "Coupled-amplitude propagation", lambda _config: validate_propagation()),
    )
    report = {}
    if progress:
        progress.start_stage("Analytic validation", total=len(checks), unit="checks")
    for index, (key, label, check) in enumerate(checks, start=1):
        if progress:
            progress.note(f"running {label}", force=True)
        report[key] = check(config)
        if progress:
            progress.update(index, detail=label)
            progress.checkpoint(key, report[key])
    report.update({
        "solver_dependent_checks": {
            "fde_mesh_and_domain_convergence": "not_run_by_this_command: use vlac validate-fde on a licensed MODE installation",
            "mode_overlap_tracking_across_z_and_wavelength": "not_run_by_this_command: use vlac validate-fde on a licensed MODE installation",
            "coupling_large_separation_from_fde": "not_run_by_this_command: use vlac validate-fde on a licensed MODE installation",
            "full_device_eme_varfdtd_fdtd_comparison": "not_run_by_this_command: provide a full-device result with compare-full-device",
            "roughness_scattering_loss": "not_run: no measured or calibrated input supplied",
        },
        "surrogate_device_results": "none",
    })
    if progress:
        progress.finish_stage("analytic checks passed")
    return report


def validate_endpoint_isolated_domain(
    backend, config: WorkflowConfig, bank, *, progress=None
) -> dict:
    """Compare compact centered endpoint modes against the original domain.

    The representative width is the midpoint of the configured upper-guide
    width range. Both liquid states, band-edge wavelengths, and endpoint
    offsets are sampled. The comparison includes the individual effective
    indices, signed complex detuning, mode overlaps, and the full-domain
    translation sensitivity of the upper isolated mode.
    """
    g, d = config.geometry, config.device
    width = 0.5 * (g.w1_min_m + g.w1_max_m)
    wavelengths = (d.wavelength_min_m, d.wavelength_max_m)
    ends = (
        ("input", g.d_c2c_input_m),
        ("output", g.d_c2c_output_m),
    )
    rows: list[dict] = []
    references: dict[tuple[str, str, str, str], ModeSnapshot | None] = {}
    if progress:
        progress.start_stage(
            "Endpoint isolated-domain comparison",
            total=2 * len(ends) * len(wavelengths),
            unit="state-end-wavelength samples",
        )

    for state in ("cross", "bar"):
        liquid = bank.liquid_cross if state == "cross" else bank.liquid_bar
        for end, offset in ends:
            for wavelength in wavelengths:
                geometry = make_geometry(
                    g,
                    w1_m=width,
                    h_rib_m=float(np.mean(g.h_rib_bounds_m)),
                    h_gap_m=float(np.mean(g.h_gap_bounds_m)),
                    d_c2c_m=offset,
                    state=state,
                    liquid_index=float(liquid.index(wavelength)),
                )
                selected: dict[str, tuple[ModeSnapshot, float, ModeSnapshot, float]] = {}
                for guide in ("upper", "lower"):
                    wide_key = (state, end, "full", guide)
                    compact_key = (state, end, "compact", guide)
                    wide, wide_overlap = backend.solve_isolated_mode(
                        geometry,
                        float(wavelength),
                        guide=guide,
                        reference=references.get(wide_key),
                        compact_x=False,
                    )
                    compact, compact_overlap = backend.solve_isolated_mode(
                        geometry,
                        float(wavelength),
                        guide=guide,
                        reference=references.get(compact_key),
                        compact_x=True,
                    )
                    references[wide_key] = wide
                    references[compact_key] = compact
                    selected[guide] = (
                        wide,
                        wide_overlap,
                        compact,
                        compact_overlap,
                    )

                factor = 2 * np.pi / wavelength
                upper_wide, upper_wide_overlap, upper_compact, upper_compact_overlap = selected[
                    "upper"
                ]
                lower_wide, lower_wide_overlap, lower_compact, lower_compact_overlap = selected[
                    "lower"
                ]
                beta_upper_wide = factor * upper_wide.neff
                beta_lower_wide = factor * lower_wide.neff
                beta_upper_compact = factor * upper_compact.neff
                beta_lower_compact = factor * lower_compact.neff
                delta_wide = beta_upper_wide - beta_lower_wide
                delta_compact = beta_upper_compact - beta_lower_compact
                rows.append(
                    {
                        "state": state,
                        "end": end,
                        "d_c2c_m": float(offset),
                        "w1_m": float(width),
                        "wavelength_m": float(wavelength),
                        "beta_upper_full_real_rad_per_m": float(beta_upper_wide.real),
                        "beta_upper_full_imag_rad_per_m": float(beta_upper_wide.imag),
                        "beta_upper_compact_real_rad_per_m": float(beta_upper_compact.real),
                        "beta_upper_compact_imag_rad_per_m": float(beta_upper_compact.imag),
                        "beta_lower_full_real_rad_per_m": float(beta_lower_wide.real),
                        "beta_lower_full_imag_rad_per_m": float(beta_lower_wide.imag),
                        "beta_lower_compact_real_rad_per_m": float(beta_lower_compact.real),
                        "beta_lower_compact_imag_rad_per_m": float(beta_lower_compact.imag),
                        "delta_beta_full_real_rad_per_m": float(delta_wide.real),
                        "delta_beta_full_imag_rad_per_m": float(delta_wide.imag),
                        "delta_beta_compact_real_rad_per_m": float(delta_compact.real),
                        "delta_beta_compact_imag_rad_per_m": float(delta_compact.imag),
                        "upper_neff_change": float(abs(upper_compact.neff - upper_wide.neff)),
                        "lower_neff_change": float(abs(lower_compact.neff - lower_wide.neff)),
                        "delta_beta_change_as_neff": float(
                            abs(delta_compact - delta_wide) / factor
                        ),
                        "upper_full_tracking_overlap": upper_wide_overlap,
                        "upper_compact_tracking_overlap": upper_compact_overlap,
                        "lower_full_tracking_overlap": lower_wide_overlap,
                        "lower_compact_tracking_overlap": lower_compact_overlap,
                        "upper_full_mode_identity": upper_wide.identity,
                        "upper_compact_mode_identity": upper_compact.identity,
                        "lower_full_mode_identity": lower_wide.identity,
                        "lower_compact_mode_identity": lower_compact.identity,
                    }
                )
                if progress:
                    progress.update(
                        len(rows),
                        detail=(
                            f"{state}, {end}, {wavelength * 1e9:.1f} nm; "
                            "full versus centered compact x-domain"
                        ),
                    )

    tolerance = config.solver.domain_convergence_neff_tolerance
    max_upper_change = max(row["upper_neff_change"] for row in rows)
    max_lower_change = max(row["lower_neff_change"] for row in rows)
    max_delta_change = max(row["delta_beta_change_as_neff"] for row in rows)
    by_state_wavelength: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        by_state_wavelength.setdefault(
            (row["state"], row["wavelength_m"]), []
        ).append(row)
    translation_changes = []
    for group in by_state_wavelength.values():
        if len(group) >= 2:
            factor = 2 * np.pi / group[0]["wavelength_m"]
            translation_changes.append(
                abs(
                    complex(
                        group[0]["beta_upper_full_real_rad_per_m"],
                        group[0]["beta_upper_full_imag_rad_per_m"],
                    )
                    - complex(
                        group[1]["beta_upper_full_real_rad_per_m"],
                        group[1]["beta_upper_full_imag_rad_per_m"],
                    )
                )
                / factor
            )
    max_translation_change = max(translation_changes, default=0.0)
    passed = (
        max_upper_change <= tolerance
        and max_lower_change <= tolerance
        and max_delta_change <= 2 * tolerance
        and max_translation_change <= tolerance
    )
    if progress:
        progress.finish_stage(
            "compact endpoint domain meets configured tolerance"
            if passed
            else "compact endpoint domain needs review; see maximum errors"
        )
    return {
        "status": "passed" if passed else "review",
        "representative_w1_m": float(width),
        "wavelengths_m": list(wavelengths),
        "sampled_states": ["cross", "bar"],
        "sampled_offsets_m": [float(value) for _, value in ends],
        "compact_domain": backend.endpoint_isolated_domain_info(compact_x=True),
        "full_domain": backend.endpoint_isolated_domain_info(compact_x=False),
        "configured_neff_tolerance": float(tolerance),
        "maximum_upper_neff_change": float(max_upper_change),
        "maximum_lower_neff_change": float(max_lower_change),
        "maximum_delta_beta_change_as_neff": float(max_delta_change),
        "maximum_full_domain_translation_change_as_neff": float(
            max_translation_change
        ),
        "rows": rows,
        "interpretation": (
            "The compact endpoint domain is acceptable only when this check passes. "
            "If it needs review, increase geometry.domain_margin_x_m or set "
            "solver.endpoint_compact_x=false."
        ),
    }


def run_fde_validation(config: WorkflowConfig, *, progress=None) -> dict:
    """Run modest MODE checks for local mesh/domain convergence and mode tracking."""
    from .solver import LumapiBackend
    backend_class = LumapiBackend
    if config.solver.engine == "native":
        from .native_backend import NativeLumapiBackend
        backend_class = NativeLumapiBackend

    bank = build_material_bank(config)
    g, d = config.geometry, config.device
    hrib = float(np.mean(g.h_rib_bounds_m))
    hgap = float(np.mean(g.h_gap_bounds_m))
    wavelength = (d.wavelength_min_m + d.wavelength_max_m) / 2
    # A cold-cache default validation performs 49 existing findmodes calls plus
    # 20 isolated-mode calls for the compact/full endpoint comparison.
    expected_solves = (
        1
        + 3 * len(config.solver.mesh_accuracy_refinements)
        + 3 * len(config.solver.domain_scale_factors)
        + 3 * 3 * 3
        + 3
        + 20
    )
    if progress:
        progress.set_solver_total(expected_solves)
        progress.start_stage("MODE API/license preflight", total=1, unit="startup checks")
    with backend_class(config, progress=progress) as backend:
        backend.preflight(wavelength)
        if progress:
            progress.update(1, detail="MODE session and small FDE license check succeeded")
            progress.finish_stage()
        mesh_geometry = make_geometry(
            g,
            w1_m=(g.w1_min_m + g.w1_max_m) / 2,
            h_rib_m=hrib,
            h_gap_m=hgap,
            d_c2c_m=0.0,
            state="cross",
            liquid_index=float(bank.liquid_cross.index(wavelength)),
        )
        mesh = backend.mesh_domain_convergence(
            mesh_geometry, wavelength, progress=progress
        )
        if progress:
            progress.checkpoint("mesh_convergence", mesh)
        domain = backend.domain_convergence(
            mesh_geometry, wavelength, progress=progress
        )
        if progress:
            progress.checkpoint("domain_convergence", domain)

        endpoint_domain = validate_endpoint_isolated_domain(
            backend, config, bank, progress=progress
        )
        if progress:
            progress.checkpoint(
                "endpoint_isolated_domain_comparison", endpoint_domain
            )

        z_values = np.linspace(0, d.length_m, 3)
        wavelengths = np.linspace(d.wavelength_min_m, d.wavelength_max_m, 3)
        prev_lambda_refs: list[ModeReference | None] = [None] * len(z_values)
        overlap_rows = []
        if progress:
            progress.start_stage(
                "Mode tracking sweep",
                total=len(z_values) * len(wavelengths),
                unit="z-wavelength samples",
            )
        for wl in wavelengths:
            current_refs = []
            previous_z_ref = None
            previous_z_kappa = None
            for iz, z in enumerate(z_values):
                u = z / d.length_m
                width = g.w1_min_m + u * (g.w1_max_m - g.w1_min_m)
                offset = g.d_c2c_input_m + u * (
                    g.d_c2c_output_m - g.d_c2c_input_m
                )
                geometry = make_geometry(
                    g,
                    w1_m=width,
                    h_rib_m=hrib,
                    h_gap_m=hgap,
                    d_c2c_m=offset,
                    state="cross",
                    liquid_index=float(bank.liquid_cross.index(wl)),
                )
                refs = [
                    ref for ref in (previous_z_ref, prev_lambda_refs[iz])
                    if ref is not None
                ]
                coeff, ref = backend.evaluate_coefficients(
                    geometry,
                    float(wl),
                    references=tuple(refs),
                    previous_kappa=previous_z_kappa,
                )
                overlap_rows.append(
                    {
                        "wavelength_nm": float(wl * 1e9),
                        "z_um": float(z * 1e6),
                        "minimum_mode_overlap": coeff.min_mode_overlap,
                        "upper_identity": coeff.upper_identity,
                        "lower_identity": coeff.lower_identity,
                        "plus_identity": coeff.supermode_identities[0],
                        "minus_identity": coeff.supermode_identities[1],
                        "extraction_flag": coeff.extraction_flag,
                    }
                )
                if progress:
                    progress.update(
                        len(overlap_rows),
                        detail=f"{wl * 1e9:.1f} nm, z={z * 1e6:.1f} um",
                    )
                current_refs.append(ref)
                previous_z_ref = ref
                previous_z_kappa = coeff.kappa
            prev_lambda_refs = current_refs
        if progress:
            progress.finish_stage()
            progress.checkpoint("mode_tracking_sweep", overlap_rows)

        if progress:
            progress.start_stage("Large-separation FDE check", total=1, unit="cross-sections")
        large_gap_geometry = make_geometry(
            g,
            w1_m=(g.w1_min_m + g.w1_max_m) / 2,
            h_rib_m=hrib,
            h_gap_m=hgap,
            d_c2c_m=g.d_c2c_bounds_m[1],
            state="cross",
            liquid_index=float(bank.liquid_cross.index(wavelength)),
        )
        large_gap_coeff, _ = backend.evaluate_coefficients(
            large_gap_geometry, wavelength
        )
        if progress:
            progress.update(1, detail="largest configured lateral separation solved")
            progress.finish_stage()
            progress.checkpoint(
                "large_separation_check",
                {
                    "d_c2c_m": g.d_c2c_bounds_m[1],
                    "kappa_real_rad_per_m": large_gap_coeff.kappa.real,
                    "kappa_imag_rad_per_m": large_gap_coeff.kappa.imag,
                },
            )
    return {
        "status": "completed",
        "mesh_accuracy_convergence": mesh,
        "domain_span_convergence": domain,
        "endpoint_isolated_domain_comparison": endpoint_domain,
        "mode_tracking_sweep": overlap_rows,
        "minimum_mode_overlap_observed": min(
            r["minimum_mode_overlap"] for r in overlap_rows
        ),
        "large_separation_fde_check": {
            "d_c2c_m": g.d_c2c_bounds_m[1],
            "kappa_real_rad_per_m": large_gap_coeff.kappa.real,
            "kappa_imag_rad_per_m": large_gap_coeff.kappa.imag,
            "overlap_upper_lower": large_gap_coeff.overlap_upper_lower,
            "extraction_flag": large_gap_coeff.extraction_flag,
            "interpretation": "reported for review; no numerical threshold is assumed",
        },
        "identical_guide_limit": (
            "covered by the independent analytic coupled-matrix check; the "
            "implemented stacked guides have different claddings"
        ),
        "full_device_comparison": (
            "not run by this command; use compare-full-device with EME/varFDTD/FDTD data"
        ),
    }


def write_validation_report(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, complex):
            return {"real": value.real, "imag": value.imag}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        raise TypeError(type(value).__name__)

    Path(path).write_text(
        json.dumps(report, indent=2, default=convert) + "\n", encoding="utf-8"
    )
