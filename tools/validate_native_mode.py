"""Run licensed v23/native parity checks without installing the package."""

from dataclasses import asdict, replace
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import tempfile
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlac.block_io import write_json, load_block
from vlac.config import load_config
from vlac.geometry import ClampedCubicProfile, make_geometry
from vlac.modes import extract_coupling, field_overlap
from vlac.native_backend import NativeLumapiBackend, column, scalar, load_native_functions
from scipy.interpolate import PchipInterpolator
from vlac.solver import LumapiBackend
from vlac.native_mode import prepare_bundle, export_result, execute_bundle


def math_checks(backend):
    session = backend.session
    checks = []
    load_native_functions(session, "vlac_core.lsf")
    load_native_functions(session, "vlac_blocks.lsf")
    positions = np.linspace(0, 1.5e-3, 19)
    for count in (4, 6, 9):
        controls = np.random.default_rng(count).uniform(0.3e-6, 2e-6, count)
        session.putv("vlac_test_controls", column(controls))
        session.putv("vlac_test_positions", column(positions))
        session.eval("vlac_test_values=vlac_bspline(vlac_test_controls,vlac_test_positions,1.5e-3);")
        actual = np.asarray(session.getv("vlac_test_values")).reshape(-1)
        expected = ClampedCubicProfile(1.5e-3, controls)(positions)
        checks.append({"name": f"bspline_{count}", "max_abs_error": float(np.max(np.abs(actual-expected))),
                       "passed": bool(np.allclose(actual, expected, rtol=1e-12, atol=1e-18))})
    for index, values in enumerate(((4., 2., 5., 1., None), (4., 2., 3.2, 2.8, None),
                                   (4.+0.1j, 2.+0.05j, 5.+0.2j, 1.+0.1j, -2.))):
        upper, lower, plus, minus, previous = values
        session.putv("vlac_test_beta", np.array([upper, lower, plus, minus, previous or 0], dtype=complex))
        session.putv("vlac_test_has_previous", float(previous is not None))
        session.eval("vlac_test_coefficient=vlac_extract(vlac_test_beta(1),vlac_test_beta(2),vlac_test_beta(3),vlac_test_beta(4),vlac_test_beta(5),vlac_test_has_previous);")
        actual = session.getv("vlac_test_coefficient")
        expected = extract_coupling(upper, lower, plus, minus, previous_kappa=previous)
        checks.append({"name": f"coupling_{index}",
                       "passed": bool(np.allclose(scalar(actual["kappa"]), expected.kappa, rtol=1e-12, atol=1e-12)
                                      and actual["extraction_flag"] == expected.extraction_flag)})
    coordinates = np.array([0., .1, .4, .7, 1.])
    queries = np.linspace(0, 1, 27)
    for index, values in enumerate((coordinates**2, np.array([0., 1., -2., 0., 0.]), np.ones(5))):
        session.putv("vlac_test_x", column(coordinates)); session.putv("vlac_test_y", column(values))
        session.putv("vlac_test_queries", column(queries))
        session.eval("vlac_test_values=vlac_pchip_real(vlac_test_x,vlac_test_y,vlac_test_queries);")
        actual = np.asarray(session.getv("vlac_test_values")).reshape(-1)
        expected = PchipInterpolator(coordinates, values)(queries)
        checks.append({"name": f"pchip_{index}", "max_abs_error": float(np.max(np.abs(actual-expected))),
                       "passed": bool(np.allclose(actual, expected, rtol=1e-12, atol=1e-12))})
    coordinates, values = np.array([0., 1.]), np.array([3., -2.])
    session.putv("vlac_test_x", column(coordinates)); session.putv("vlac_test_y", column(values))
    session.putv("vlac_test_queries", column(queries))
    session.eval("vlac_test_values=vlac_pchip_real(vlac_test_x,vlac_test_y,vlac_test_queries);")
    actual = np.asarray(session.getv("vlac_test_values")).reshape(-1)
    expected = PchipInterpolator(coordinates, values)(queries)
    checks.append({"name": "pchip_two_points", "max_abs_error": float(np.max(np.abs(actual-expected))),
                   "passed": bool(np.allclose(actual, expected, rtol=1e-12, atol=1e-12))})
    return checks


@contextmanager
def mode_temporary_directory(session):
    with tempfile.TemporaryDirectory() as folder:
        try:
            yield folder
        finally:
            # MODE holds its working directory open on Windows.
            session.eval('cd('+json.dumps(ROOT.as_posix())+');')


def native_block_smoke(backend, output=None):
    """Exercise standalone control flow with explicitly synthetic optical data."""
    session = backend.session
    with mode_temporary_directory(session) as folder:
        bundle = prepare_bundle(backend.config, folder, width_samples=3)
        session.eval("matlabload(" + json.dumps((bundle/"native_config.mat").as_posix()) + ");")
        synthetic = r'''
function vlac_isolated(cfg,geometry,wavelength,guide,refs,compact) {
    result=struct; result.ok=1; result.error=""; result.overlap=1;
    mode=struct; mode.neff=1.7;
    if (guide=="upper") { mode.neff=1.7+(geometry.w1_m-1.15e-6)*1e10*wavelength/(2*pi); }
    mode.ex=matrix(16,16)+1/16; mode.ey=matrix(16,16);
    mode.te_fraction=1; mode.core_fraction=1; mode.identity="SYNTHETIC_TEST_ONLY";
    result.mode=mode; return result;
}
function vlac_coefficients(cfg,geometry,wavelength,refs,previous_kappa,has_previous) {
    result=struct; result.ok=1; result.error="";
    delta=(geometry.w1_m-1.15e-6)*1e10;
    test_coupling=(8e-6-abs(geometry.d_c2c_m))*1e8;
    splitting=sqrt((delta/2)^2+test_coupling^2);
    coefficient=vlac_extract(1e7+delta/2,1e7-delta/2,1e7+splitting,1e7-splitting,previous_kappa,has_previous);
    coefficient.kappa_profile_check=test_coupling; coefficient.overlap_upper_lower=0;
    coefficient.min_mode_overlap=1; coefficient.upper_identity="SYNTHETIC_TEST_ONLY";
    coefficient.lower_identity="SYNTHETIC_TEST_ONLY"; coefficient.plus_identity="SYNTHETIC_TEST_ONLY";
    coefficient.minus_identity="SYNTHETIC_TEST_ONLY";
    mode=struct; mode.neff=1.7; mode.ex=matrix(16,16)+1/16; mode.ey=matrix(16,16);
    reference=struct; reference.upper=mode; reference.lower=mode; reference.plus=mode; reference.minus=mode;
    result.coefficients=coefficient; result.reference=reference; return result;
}
function vlac_block1_inspection(cfg,endpoint) {
    # Synthetic execution fixture only: deliberately no FDE solve.
    result=struct; result.ok=1; return result;
}
'''
        starts = list(re.finditer(r"^function\s+(\w+)\(", synthetic, re.MULTILINE))
        for index, match in enumerate(starts):
            end = starts[index+1].start() if index+1 < len(starts) else len(synthetic)
            print("Loading synthetic fixture: " + match.group(1), flush=True)
            session.eval(synthetic[match.start():end])
        session.eval("vlac_cfg.profile_evaluation_budget=3; vlac_cfg.max_grid_refinements=0; vlac_cfg.control_points=4; vlac_cfg.z_samples_initial=3; vlac_cfg.tracking_grid_nx=16; vlac_cfg.tracking_grid_ny=16;")
        for number in (1, 2, 3):
            arguments = "vlac_cfg" + (f",vlac_smoke_{number-1}" if number > 1 else "")
            print(f"Synthetic native Block {number}", flush=True)
            session.putv("vlac_smoke_code", f"vlac_smoke_{number}=vlac_block{number}({arguments});")
            try:
                session.eval('vlac_smoke_error=""; try { eval(vlac_smoke_code); } catch(vlac_smoke_error);')
            except Exception:
                try:
                    print("Native error: " + str(session.getv("vlac_smoke_error")), flush=True)
                except Exception:
                    pass
                raise
            error = session.getv("vlac_smoke_error")
            if error:
                raise RuntimeError(f"Synthetic native Block {number}: {error}")
        checks = []
        for number in (1, 2, 3):
            saved = json.dumps((bundle/f"synthetic_block{number}.mat").as_posix())
            session.eval(f"matlabsave({saved},vlac_smoke_{number}); clear(vlac_smoke_{number}); matlabload({saved});")
            value = session.getv(f"vlac_smoke_{number}")
            if not bool(scalar(value["ok"])):
                raise AssertionError(f"Native Block {number} synthetic smoke failed: {value.get('error')}")
            if number == 2 and (bool(scalar(value["optimizer_converged"])) or int(scalar(value["evaluations"])) != 3):
                raise AssertionError("Native optimizer did not honor its smoke evaluation budget")
            target = bundle / "test_exports" / f"block{number}"
            export_result(value, number, backend.config, target)
            load_block(target, number, backend.config)
            checks.append({"name": f"native_block{number}_handoff", "passed": True,
                           "scope": "Synthetic control flow and real artifact validation; no optical performance claim"})
        # Execute the generated entry point too, preserving the synthetic functions.
        # This checks removal/replacement of stale MAT/LMS files and success gating.
        entry = (bundle/'block1.lsf').read_text().replace('vlac_core; vlac_blocks;', '')
        entry = entry.replace('matlabload("native_config.mat");',
                              'matlabload("native_config.mat"); vlac_cfg.tracking_grid_nx=16; vlac_cfg.tracking_grid_ny=16;')
        (bundle/'block1.lms').write_text('STALE SYNTHETIC TEST FILE')
        session.putv('vlac_smoke_entry', entry)
        session.eval('eval(vlac_smoke_entry);')
        value = session.getv('block1_native')
        if not bool(scalar(value['ok'])) or value.get('inspection_kind') != 'full_two_guide_input_cross_min_wavelength_v1':
            raise AssertionError('Generated Block 1 entry did not complete its synthetic inspection export')
        if output is not None:
            import shutil
            shutil.copy2(bundle/'block1_native.mat', output/'synthetic_native_import_fixture.mat')
        from unittest.mock import patch
        with patch.object(NativeLumapiBackend, '_import_lumapi', side_effect=AssertionError('Offline MAT import opened MODE')):
            imported, _ = execute_bundle(bundle, 1, import_only=True)
            load_block(imported, 1, backend.config)
        checks.append({'name':'native_mat_offline_import','passed':True,
                       'scope':'MAT written by installed MODE; synthetic optical fixture'})
        session.eval('function vlac_block1_inspection(cfg,endpoint) { result=struct; result.ok=0; return result; }')
        session.eval('eval(vlac_smoke_entry);')
        value = session.getv('block1_native')
        if bool(scalar(value['ok'])) or not (bundle/'block1.lms').is_file():
            raise AssertionError('Failed native inspection was not marked incomplete with an empty saved project')
        session.eval('load('+json.dumps((bundle/'block1.lms').as_posix())+'); vlac_empty_scene=getnamednumber("FDE");')
        if scalar(session.getv('vlac_empty_scene')) != 0:
            raise AssertionError('Incomplete LMS contains a stale solved FDE scene')
        with patch.object(NativeLumapiBackend, '_import_lumapi', side_effect=AssertionError('Recovery import opened MODE')):
            recovered, _ = execute_bundle(bundle, 1, import_only=True, last_successful=True)
            load_block(recovered, 1, backend.config)
        checks.append({'name': 'native_block1_entry_export_and_failed_rerun', 'passed': True,
                       'scope': 'Synthetic execution/serialization only; inspection solve deliberately stubbed'})
        seeded = prepare_bundle(backend.config, bundle/"seeded", profile_dir=bundle/"test_exports/block2")
        session.eval("matlabload(" + json.dumps((seeded/"native_config.mat").as_posix()) + ");")
        session.eval("matlabload(" + json.dumps((seeded/"block2_native.mat").as_posix()) + ");")
        session.eval("vlac_seeded_bar=vlac_block3(vlac_cfg,block2_native);")
        seeded_value = session.getv("vlac_seeded_bar")
        export_result(seeded_value, 3, backend.config, seeded/"test_export")
        load_block(seeded/"test_export", 3, backend.config)
        checks.append({"name": "python_profile_to_native_bar_handoff", "passed": True,
                       "scope": "Synthetic saved-profile import; no optical performance claim"})
        # Restore the real FDE functions before any physical comparison.
        load_native_functions(session, "vlac_core.lsf")
        load_native_functions(session, "vlac_blocks.lsf")
        return checks


def section(backend, config, state):
    liquid = backend.bank.liquid_cross if state == "cross" else backend.bank.liquid_bar
    geometry = make_geometry(config.geometry, w1_m=1.15e-6, h_rib_m=175e-9, h_gap_m=1e-6,
                             d_c2c_m=0., state=state, liquid_index=float(liquid.index(1.3e-6)))
    started = time.perf_counter()
    coefficients, reference = backend.evaluate_coefficients(geometry, 1.3e-6)
    return coefficients, reference, time.perf_counter()-started


def compare_sections(original, native):
    before, ref_before, seconds_before = original
    after, ref_after, seconds_after = native
    coefficients = {}
    for name in ("beta_upper", "beta_lower", "beta_plus", "beta_minus", "delta_beta", "delta_pm", "delta_12", "kappa",
                 "overlap_upper_lower", "kappa_profile_check"):
        one, two = getattr(before, name), getattr(after, name)
        coefficients[name] = {"python": one, "native": two,
                              "passed": bool(np.isclose(one, two, rtol=1e-5, atol=1e-2, equal_nan=True))}
    overlaps = {name: field_overlap(one, two) for name, one, two in (
        ("upper", ref_before.upper, ref_after.upper), ("lower", ref_before.lower, ref_after.lower),
        ("plus", ref_before.supermodes[0], ref_after.supermodes[0]),
        ("minus", ref_before.supermodes[1], ref_after.supermodes[1]))}
    return {"passed": all(value["passed"] for value in coefficients.values()) and min(overlaps.values()) > .99999
                      and before.extraction_flag == after.extraction_flag,
            "coefficients": coefficients, "mode_field_overlaps": overlaps,
            "extraction_flags": [before.extraction_flag, after.extraction_flag],
            "python_seconds": seconds_before, "native_seconds": seconds_after}


def native_cache_checks(config, output):
    """Actual MODE data: cold solve, tracking reselection, then an offline disk hit."""
    from unittest.mock import patch
    cache_path = output/'cache_check.sqlite'
    with NativeLumapiBackend(config, cache_path=cache_path) as backend:
        cold = section(backend, config, 'cross')
        before = backend.budget.solves
        liquid = float(backend.bank.liquid_cross.index(1.3e-6))
        geometry = make_geometry(config.geometry, w1_m=1.15e-6, h_rib_m=175e-9, h_gap_m=1e-6,
                                 d_c2c_m=0., state='cross', liquid_index=liquid)
        started = time.perf_counter()
        warm, reference = backend.evaluate_coefficients(geometry, 1.3e-6, references=(cold[1],), previous_kappa=cold[0].kappa)
        seconds = time.perf_counter()-started
        comparison = compare_sections(cold, (warm, reference, seconds))
        comparison['additional_warm_solves'] = backend.budget.solves-before
        comparison['cold_solves'] = before
        comparison['passed'] &= comparison['additional_warm_solves'] == 0
    with patch.object(NativeLumapiBackend, '_import_lumapi', side_effect=AssertionError('Disk hit opened MODE')):
        with NativeLumapiBackend(config, cache_path=cache_path) as backend:
            started = time.perf_counter()
            coefficients, _ = backend.evaluate_coefficients(geometry, 1.3e-6)
            comparison['offline_disk_hit_seconds'] = time.perf_counter()-started
            comparison['offline_disk_hit_passed'] = coefficients.kappa == cold[0].kappa
    comparison['passed'] &= comparison['offline_disk_hit_passed']
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--math-only", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--max-solves", type=int, default=24)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--config", default=str(ROOT / "config/python_joint.json"))
    parser.add_argument("--out", default=str(ROOT / "diagnostics/native_mode"))
    args = parser.parse_args()
    output = Path(args.out).resolve(); output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config = replace(config, solver=replace(config.solver, hide_gui=True),
                     execution=replace(config.execution, max_solves=args.max_solves, max_seconds=args.max_seconds))
    report = {"status": "failed", "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Numerical implementation parity; no mesh convergence or full-device performance claim",
              "configuration": config.as_dict()}
    started = time.perf_counter()
    try:
        if args.cache_only:
            report['cache_reuse'] = native_cache_checks(config, output)
            report['status'] = 'passed' if report['cache_reuse']['passed'] else 'failed'
            report['elapsed_seconds'] = time.perf_counter()-started
            write_json(output/'comparison.json',report)
            print('Cache check: '+report['status'],flush=True)
            return 0 if report['status']=='passed' else 1
        original = {}
        if not args.math_only:
            print("Running original v23 cross-sections (cross and bar)", flush=True)
            with LumapiBackend(config) as backend:
                for state in ("cross", "bar"):
                    print("Python baseline: " + state, flush=True)
                    original[state] = section(backend, config, state)
                backend.session.save((output / "python_section.lms").as_posix())
        print("Loading native LSF library", flush=True)
        with NativeLumapiBackend(config) as backend:
            report["math"] = math_checks(backend)
            report["native_block_smoke"] = native_block_smoke(backend, output)
            if not args.math_only:
                report["sections"] = {}
                for state in ("cross", "bar"):
                    print("Native section: " + state, flush=True)
                    report["sections"][state] = compare_sections(original[state], section(backend, config, state))
                backend.session.save((output / "native_section.lms").as_posix())
        checks = [row["passed"] for row in report["math"]]
        checks += [row["passed"] for row in report["native_block_smoke"]]
        checks += [row["passed"] for row in report.get("sections", {}).values()]
        report["status"] = "passed" if all(checks) else "failed"
    except Exception as exc:
        report["error"] = str(exc); report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    report["elapsed_seconds"] = time.perf_counter()-started
    write_json(output / "comparison.json", report)
    print("Result: " + report["status"] + "; report: " + str(output / "comparison.json"), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
