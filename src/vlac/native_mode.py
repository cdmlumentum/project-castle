"""Prepare, run and import standalone MODE blocks.

Preparing a bundle is offline. Running or importing it requires MODE/lumapi.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import savemat

from . import blocks
from .block_io import digest, load_block, source_record, validate_grid, write_artifact, write_json
from .config import config_from_dict, load_config
from .endpoint import EndpointSearchResult
from .materials import build_material_bank
from .native_backend import (NativeLumapiBackend, coefficients_from_native, column,
                             native_script, profile_request, scalar)
from .profiles import ProfileSolution


def config_payload(config):
    """Build exactly the same solver input used by the native Python backend."""
    adapter = object.__new__(NativeLumapiBackend)
    adapter.config = config
    adapter.bank = build_material_bank(config)
    adapter.mesh_accuracy = config.solver.mesh_accuracy
    adapter._build_domain()
    return adapter.native_config()


def _coefficient_packet(grid):
    values = []
    for row in grid:
        for coefficient in row:
            value = asdict(coefficient)
            value["plus_identity"], value["minus_identity"] = value.pop("supermode_identities")
            if not np.isfinite(value["kappa_profile_check"]):
                value["kappa_profile_check"] = -1.
            values.append(value)
    return {"ok": 1., "error": "", "items": values, "nz": float(len(grid[0])), "nw": float(len(grid))}


def _endpoint_packet(endpoint):
    return {"ok": 1., "error": "", "w1_input_m": endpoint.w1_input_m,
            "w1_output_m": endpoint.w1_output_m, "delta_scale": endpoint.delta_beta_max_rad_per_m,
            "h_gap_m": endpoint.h_gap_m, "h_rib_m": endpoint.h_rib_m,
            "rows": endpoint.endpoint_records, "method": endpoint.search_method,
            "search_converged": float(endpoint.converged), "fresh_verification_passed": 1.}


def prepare_bundle(config, output, *, endpoint_dir=None, profile_dir=None, width_samples=None):
    config.validate()
    g, e, o, d = config.geometry, config.endpoint_search, config.optimizer, config.device
    if g.h_gap_bounds_m[0] != g.h_gap_bounds_m[1] or g.h_rib_bounds_m[0] != g.h_rib_bounds_m[1]:
        raise ValueError("Standalone native blocks currently require fixed h_gap and h_rib")
    if not e.pso_lower_mode_state_independent:
        raise ValueError("Standalone Block 1 requires the explicit state-independent lower-mode assumption")
    if o.maximum_slope_m_per_m is not None or o.maximum_curvature_per_m is not None:
        raise ValueError("Use the native Python backend with Powell for slope/curvature constraints; the standalone pattern optimizer does not support them")
    count = e.adaptive_max_widths if width_samples is None else width_samples
    if isinstance(count, bool) or not isinstance(count, int) or count < 3:
        raise ValueError("width_samples must be an integer >= 3")
    if endpoint_dir and profile_dir:
        raise ValueError("Choose endpoint_dir or profile_dir")
    upstream = load_block(profile_dir or endpoint_dir, 2 if profile_dir else 1, config) if endpoint_dir or profile_dir else None
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = ["native_config.mat", "configuration.json", "manifest.json", "vlac_core.lsf", "vlac_blocks.lsf"]
    targets += [f"block{number}.lsf" for number in (1, 2, 3)]
    if config.solver.calibration:
        targets.append('block0_certificate.json')
    if any((output / name).exists() for name in targets):
        raise FileExistsError("Choose a new bundle directory to preserve its configuration and results")
    payload = config_payload(config)
    payload["geometry"]["h_gap_m"] = g.h_gap_bounds_m[0]
    payload["geometry"]["h_rib_m"] = g.h_rib_bounds_m[0]
    payload.update(
        endpoint_wavelengths_m=column([d.wavelength_min_m, d.wavelength_max_m]),
        endpoint_widths_m=column(np.linspace(g.w1_min_m, g.w1_max_m, count)),
        width_min_m=g.w1_min_m, width_max_m=g.w1_max_m,
        offset_min_m=g.d_c2c_bounds_m[0], offset_max_m=g.d_c2c_bounds_m[1],
        offset_input_m=g.d_c2c_input_m, offset_output_m=g.d_c2c_output_m,
        endpoint_balance_rtol=e.cross_magnitude_balance_rtol, guided_neff_margin=e.pso_guided_neff_margin,
        control_points=float(o.control_points), profile_evaluation_budget=float(o.max_function_evaluations_per_grid),
        minimum_control_step_fraction=o.minimum_control_step_fraction,
        max_grid_refinements=float(o.max_grid_refinements), grid_tolerance=d.grid_convergence_rtol,
        z_samples_initial=float(d.z_samples_initial), wavelength_samples_initial=float(d.wavelength_samples_initial),
        z_samples_max=float(d.z_samples_max), wavelength_samples_max=float(d.wavelength_samples_max),
    )
    # Store absolute source paths for the Python handoff; MODE uses embedded tables.
    saved = config.as_dict()
    saved["solver"]["engine"] = "native"
    for key in ("silicon_data_csv", "sin_extinction_csv", "liquid_dispersion_csv"):
        value = config.resolved_path(getattr(config.materials, key))
        saved["materials"][key] = str(value) if value else None
    payload["bundle_id"] = digest({"payload": payload, "core": native_script("vlac_core.lsf"), "blocks": native_script("vlac_blocks.lsf")})
    # Lists of dictionaries must be MATLAB cells, never heterogeneous structs.
    savemat(output / "native_config.mat", {"vlac_cfg": payload}, long_field_names=True)
    write_json(output / "configuration.json", saved)
    if config.solver.calibration:
        write_json(output/'block0_certificate.json', config.solver.calibration)
    for name in ("vlac_core.lsf", "vlac_blocks.lsf"):
        (output / name).write_text(native_script(name), encoding="utf-8")
    for number in (1, 2, 3):
        result_name = f"block{number}_native"
        lines = ["# Generated MODE entry point. Native algorithms are recorded in the result.",
                 f"cd({json.dumps(output.as_posix())});", "vlac_core; vlac_blocks;",
                 'matlabload("native_config.mat");', "newproject;"]
        if number > 1:
            lines.append(f'matlabload("block{number-1}_native.mat");')
        arguments = "vlac_cfg" + (f",block{number-1}_native" if number > 1 else "")
        # Safe-mode-compatible invalidation: an interrupted run cannot import old success.
        lines += [f'{result_name}=struct; {result_name}.ok=0;',
                  f'{result_name}.error="incomplete_native_attempt";',
                  f'{result_name}.bundle_id=vlac_cfg.bundle_id;',
                  f'matlabsave("{result_name}.mat",{result_name});',
                  f'save("block{number}.lms");']
        lines += [f"{result_name}=vlac_block{number}({arguments});",
                  f"{result_name}.bundle_id=vlac_cfg.bundle_id;"]
        if number == 1:
            lines += [f'if ({result_name}.ok) {{',
                      f'vlac_inspection=vlac_block1_inspection(vlac_cfg,{result_name});',
                      f'if (!vlac_inspection.ok) {{ {result_name}.ok=0; {result_name}.error="inspection_solve_failed"; }}',
                      f'else {{ {result_name}.inspection_kind="full_two_guide_input_cross_min_wavelength_v1"; }}',
                      '}']
        lines += [f'if ({result_name}.ok) {{ save("block{number}.lms"); save("block{number}_last_successful.lms");',
                  f'matlabsave("block{number}_last_successful_native.mat",{result_name}); }}',
                  f'matlabsave("{result_name}.mat",{result_name});', f"?{result_name}.ok;"]
        (output / f"block{number}.lsf").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if upstream is not None:
        endpoint, solution, _, _ = upstream
        ep = _endpoint_packet(endpoint); ep["bundle_id"] = payload["bundle_id"]
        # np.object_ ensures that records round-trip as a Lumerical cell.
        ep["rows"] = np.array(ep["rows"], dtype=object).reshape(-1, 1)
        savemat(output / "block1_native.mat", {"block1_native": ep}, long_field_names=True)
        if solution is not None:
            grid = _coefficient_packet(solution.cross_coefficients)
            grid.update(objective=solution.objective, delta_target=column(solution.delta_target_rad_per_m),
                        kappa_target=column(solution.kappa_target_rad_per_m))
            grid["items"] = np.array(grid["items"], dtype=object).reshape(-1, 1)
            profile = {"ok": 1., "endpoint": ep, "request": profile_request(endpoint, solution.w1_control_m,
                       solution.d_c2c_control_m, solution.z_m, solution.wavelengths_m, "cross"),
                       "grid": grid, "bundle_id": payload["bundle_id"], "method": "imported_python_profile",
                       "optimizer_converged": float(solution.optimizer_history[-1]["success"]),
                       "evaluations": 0., "refinements": 1., "grid_history": np.array([0.], dtype=object),
                       "grid_checks": {"passed": float(solution.grid_converged), "z_error": -1., "wavelength_error": -1.}}
            savemat(output / "block2_native.mat", {"block2_native": profile}, long_field_names=True)
    manifest = {"bundle_id": payload["bundle_id"], "standalone_algorithms": {"block1": "sampled_widths", "block2": "coordinate_pattern"},
                "block0_certificate_id": config.solver.calibration['certificate_id'] if config.solver.calibration else None,
                "scope": "Native alternatives; original adaptive_1d and Powell remain available through solver.engine=native",
                "files": {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in targets if name != "manifest.json"}}
    write_json(output / "manifest.json", manifest)
    return output


def _grid(value):
    nz, nw = int(scalar(value["nz"])), int(scalar(value["nw"]))
    return [[coefficients_from_native(value["items"][iw*nz+iz]) for iz in range(nz)] for iw in range(nw)]


def _endpoint(value, config):
    if not value.get("ok") or not value.get("fresh_verification_passed"):
        raise ValueError("Native endpoint result is incomplete or unverified")
    return EndpointSearchResult(
        float(scalar(value["h_gap_m"])), float(scalar(value["h_rib_m"])),
        float(scalar(value["w1_input_m"])), float(scalar(value["w1_output_m"])),
        float(scalar(value["delta_scale"])), np.array([config.device.wavelength_min_m, config.device.wavelength_max_m]),
        list(value["rows"]), [], bool(scalar(value.get("search_converged", 0.))), [], {},
        search_method=value["method"], search_summary={"fresh_verification_passed": True, "method": value["method"]},
    )


def export_result(value, block, config, output, *, inspection_project=None):
    if not bool(scalar(value["ok"])):
        raise ValueError("Native block did not complete: " + value.get("error", "unknown"))
    profile = value["profile"] if block == 3 else value
    endpoint_value = value if block == 1 else profile["endpoint"]
    endpoint = _endpoint(endpoint_value, config)
    source = {"backend_name": NativeLumapiBackend.backend_name, "native_bundle_id": value["bundle_id"]}
    if block == 1:
        if value.get('inspection_kind') != 'full_two_guide_input_cross_min_wavelength_v1':
            inspection_project = None
        return blocks.export_endpoint(output, config, endpoint, backend_name=NativeLumapiBackend.backend_name, source=source,
                                      mode_profiles=value.get("mode_profiles"), inspection_project=inspection_project)
    request, grid = profile["request"], profile["grid"]
    z, wavelengths = np.asarray(request["z_m"]).reshape(-1), np.asarray(request["wavelengths_m"]).reshape(-1)
    solution = ProfileSolution(
        np.asarray(request["w_control_m"]).reshape(-1), np.asarray(request["d_control_m"]).reshape(-1), z, wavelengths,
        np.asarray(grid["delta_target"]).reshape(-1), np.asarray(grid["kappa_target"]).reshape(-1), _grid(grid),
        float(scalar(grid["objective"])),
        [{"success": bool(scalar(profile["optimizer_converged"])), "method": profile["method"],
          "evaluations": int(scalar(profile["evaluations"]))}], [], bool(scalar(profile["grid_checks"]["passed"])),
    )
    # Imported Python profiles carry their objective/targets explicitly below.
    report = blocks.profile_report(solution, endpoint, config)
    report["native_optimizer_method"] = profile["method"]
    if block == 2:
        from .block0 import require_design_verification
        calibration_check = require_design_verification(config, output, endpoint, solution)
        if calibration_check is not None:
            report['block0_verification'] = calibration_check
        report['diagnostics'] = blocks.export_profiles(output, config, endpoint, solution)
        write_json(Path(output)/"block2_summary.json", report)
        write_json(Path(output)/"configuration.json", config.as_dict())
        return write_artifact(output, 2, config, endpoint, solution=solution, validation=report, source=source)
    bar = _grid(value["grid"])
    quality = validate_grid(bar, solution, config)
    passed = bool(scalar(value["grid_checks"]["passed"])) and quality["passed"]
    report = {"block": 3, "status": "passed" if passed else "review", "execution_completed": True,
              "coefficient_quality": quality, "grid_validation": value["grid_checks"],
              "upstream_validation": report, "profile_optimized_this_block": False}
    from .block0 import require_design_verification
    calibration_check = require_design_verification(config, output, endpoint, solution, bar=bar)
    if calibration_check is not None:
        report['block0_verification'] = calibration_check
    blocks.export_profiles(output, config, endpoint, solution, bar)
    write_json(Path(output)/"block3_summary.json", report)
    write_json(Path(output)/"configuration.json", config.as_dict())
    return write_artifact(output, 3, config, endpoint, solution=solution, bar=bar, validation=report, source=source)


def normalize_mat_cells(value):
    """Normalize MATLAB cell vectors while retaining numeric field array shapes."""
    if isinstance(value, dict):
        result = {key: normalize_mat_cells(item) for key,item in value.items()}
        for key in ('items', 'rows', 'grid_history'):
            if key in result and isinstance(result[key], dict): result[key] = [result[key]]
        return result
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [normalize_mat_cells(item) for item in value.flat]
    if isinstance(value, list): return [normalize_mat_cells(item) for item in value]
    return value


def _execute_bundle(bundle, block, *, import_only=False, output=None, last_successful=False):
    if last_successful and not import_only: raise ValueError('last_successful is an import option')
    suffix = '_last_successful' if last_successful else ''
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle/"manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["files"].items():
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest() != checksum:
            raise ValueError("Bundle input changed: " + name + ". Prepare a new bundle for changed inputs.")
    config = load_config(bundle/"configuration.json")
    config = replace(config, solver=replace(config.solver, hide_gui=True))
    output = Path(output) if output is not None else bundle / "python" / f"block{block}"
    output.mkdir(parents=True, exist_ok=True)
    blocks.invalidate(output, block)
    try:
        if import_only:
            from .mat_io import load_mat
            packet = load_mat(bundle/f"block{block}{suffix}_native.mat")
            value = packet[f"block{block}_native"]
            value = normalize_mat_cells(value)
        else:
            with NativeLumapiBackend(config) as backend:
                backend.session.eval((bundle/f"block{block}.lsf").read_text(encoding="utf-8"))
                value = backend.session.getv(f"block{block}_native")
        if value.get("bundle_id") != manifest["bundle_id"]:
            raise ValueError("Native result belongs to a different bundle")
        artifact = export_result(value, block, config, output, inspection_project=bundle/f"block1{suffix}.lms" if block == 1 else None)
        return output, artifact["validation"]["overall_status"]
    except Exception as exc:
        write_json(output/f"block{block}_failure.json", {"status": "failed", "error": str(exc)})
        raise


def execute_bundle(bundle, block, *, import_only=False, last_successful=False):
    from .runtime import output_attempt
    output = Path(bundle).resolve()/'python'/f'block{block}'
    with output_attempt(output, 'native import' if import_only else 'native run') as attempt:
        _, status = _execute_bundle(bundle, block, import_only=import_only, output=attempt, last_successful=last_successful)
    return output, status


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vlac-native")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="export standalone LSF blocks and material tables without launching MODE")
    prepare.add_argument("--config", required=True); prepare.add_argument("--out", required=True)
    prepare.add_argument("--width-samples", type=int)
    prepare.add_argument('--block0-dir', help='Passed Block 0 directory or certificate.json.')
    source = prepare.add_mutually_exclusive_group()
    source.add_argument("--endpoint-dir"); source.add_argument("--profile-dir")
    for name in ("run", "import"):
        command = commands.add_parser(name, help="run a native block or import a result created in MODE")
        if name == "import": command.add_argument("--last-successful", action="store_true", help="Import preserved success after a later failed standalone attempt")
        command.add_argument("--bundle", required=True); command.add_argument("--block", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            config = load_config(args.config)
            if args.block0_dir:
                from .block0 import apply_certificate
                config = apply_certificate(config, args.block0_dir)
            print(prepare_bundle(config, args.out, endpoint_dir=args.endpoint_dir,
                                 profile_dir=args.profile_dir, width_samples=args.width_samples))
            return 0
        output, status = execute_bundle(args.bundle, args.block, import_only=args.command == "import", last_successful=getattr(args,"last_successful",False))
        print(str(output) + ": " + status)
        return 0 if status == "passed" else 3
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
