"""Command-line entry points."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .config import load_config
from .endpoint import search_endpoints
from .endpoint_constraints import CROSS_STATE_SIGN_REQUIREMENT, require_cross_state_signs, require_endpoint_constraints
from .export import export_run
from .full_device_validation import compare_full_device_csv
from .materials import build_material_bank
from .optimize import optimize_profiles
from .profiles import evaluate_design_profiles
from .propagate import propagate_state
from .progress import ProgressReporter
from .solver import LumapiBackend, LumapiUnavailable


from .validation import (
    run_analytic_validation,
    run_fde_validation,
    write_validation_report,
)
from .python_pso import search_endpoints_python_pso
from .adaptive_endpoint import search_endpoints_adaptive

from .runtime import output_attempt, resolved_cache, BudgetExceeded, RunBudget
from . import blocks
from .block_io import load_block, source_record, write_artifact, write_json


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.json"


def _backend(config, **kwargs):
    if config.solver.engine == "native":
        from .native_backend import NativeLumapiBackend
        return NativeLumapiBackend(config, **kwargs)
    return LumapiBackend(config, **kwargs)


def _output_path(config, raw: str | None = None) -> Path:
    value = raw or config.output.directory
    path = Path(value)
    return path if path.is_absolute() else (config.project_root / path).resolve()


def _apply_runtime_overrides(config, *, api_path=None, hide_gui=None, platform=None, block0_dir=None):
    updates = {}
    if api_path is not None:
        updates["api_path"] = api_path
    if hide_gui is not None:
        updates["hide_gui"] = bool(hide_gui)
    if platform is not None:
        updates["platform"] = platform
    if updates:
        config = replace(config, solver=replace(config.solver, **updates))
    if block0_dir:
        from .block0 import apply_certificate
        config = apply_certificate(config, block0_dir)
    return config


def _run_stage_plan(config) -> list[str]:
    stages = [
        "MODE API/license preflight",
        "Endpoint detuning search",
        "Cross-state profile optimization",
        "Bar-state coefficient extraction",
        "Cross-state propagation",
        "Bar-state propagation",
        "Export",
    ]
    if config.endpoint_search.method in {"python_pso", "adaptive_1d"}:
        verify = "Verify adaptive endpoint best candidate" if config.endpoint_search.method == "adaptive_1d" else "Verify Python PSO best candidate"
        stages[1:2] = ["Precompute state-independent lower-guide modes", "Endpoint detuning search", verify]
    if config.materials.sin_index_sensitivity_offsets:
        stages.append("SiN index sensitivity")
    return stages


def _write_failure_report(path: Path, command: str, exc: Exception, progress) -> None:
    payload = {
        "status": "incomplete" if isinstance(exc, BudgetExceeded) else "failed",
        "command": command,
        "failed_stage": progress.stage_name,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "completed_stages": progress.completed_stages,
        "runtime_diagnostics": progress.runtime_stats,
        "progress_log": str(progress.log_path),
        "status_file": str(progress.status_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_workflow(
    config_path: str | Path,
    out_path: str | Path | None = None,
    *,
    api_path: str | None = None,
    hide_gui: bool | None = None,
    platform: str | None = None,
    endpoint_method: str | None = None,
    endpoint_dir: str | Path | None = None,
    profile_dir: str | Path | None = None,
    bar_dir: str | Path | None = None,
    block0_dir: str | Path | None = None,
    execution=None, characterize=False,
) -> Path:
    config = _apply_runtime_overrides(load_config(config_path),api_path=api_path,hide_gui=hide_gui,platform=platform,block0_dir=block0_dir)
    if endpoint_method is not None:
        config = replace(config,endpoint_search=replace(config.endpoint_search,method=endpoint_method))
    config = _execution_overrides(config, execution, characterize)
    config.validate()
    sources = [(2,endpoint_dir),(3,profile_dir),(4,bar_dir)]
    selected = [(block,path) for block,path in sources if path is not None]
    if len(selected)>1: raise ValueError('Choose only one of endpoint_dir, profile_dir or bar_dir.')
    if not selected and config.execution.resume:
        from .runtime import resume_source
        reusable = resume_source(config, _output_path(config, str(out_path) if out_path else None))
        if reusable: selected = [reusable]
    first,source_path = selected[0] if selected else (1,None)
    endpoint,solution,bar,artifact = load_block(source_path,first-1,config) if selected else (None,None,None,None)
    output = _output_path(config,str(out_path) if out_path else None)
    output.mkdir(parents=True,exist_ok=True)
    blocks.invalidate(output,first)
    cache_path = resolved_cache(config)
    progress = ProgressReporter('vlac run',output,context={'config_path':str(Path(config_path).resolve()),
        'first_block':first,'loaded_from':str(source_path) if source_path else None})
    plan = _run_stage_plan(config)
    if first>=2: plan = ['MODE API/license preflight']+plan[plan.index('Cross-state profile optimization'):]
    if first>=3: plan = ['MODE API/license preflight']+plan[plan.index('Bar-state coefficient extraction'):]
    if first>=4: plan = plan[plan.index('Cross-state propagation'):]
    progress.set_plan(plan)
    progress.checkpoint('configuration',config.as_dict())
    try:
        bank = build_material_bank(config)
        if artifact is not None:
            # Copy the loaded stage as a portable checkpoint before computing
            # anything downstream; legacy endpoint exports are normalized here.
            artifact = write_artifact(output,first-1,config,endpoint,solution=solution,bar=bar,
                validation=artifact.get('validation'),source=source_record(artifact),
                characterization=artifact.get('payload',{}).get('characterization'))
            progress.checkpoint('loaded_upstream_block',source_record(artifact))
        budget = RunBudget(config.execution)
        manager = _backend(config,cache_path=cache_path,progress=progress) if first<=3 else nullcontext(None)
        with manager as backend:
            if backend is not None:
                backend.budget = budget
                progress.start_stage('MODE API/license preflight',total=1,unit='startup checks')
                backend.preflight(.5*(config.device.wavelength_min_m+config.device.wavelength_max_m))
                progress.update(1)
                progress.finish_stage()
            if first==1:
                endpoint = search_endpoints(backend,config,bank,progress=progress,output_dir=output)
                artifact = blocks.export_endpoint(output,config,endpoint,backend_name=backend.backend_name,backend=backend)
                progress.checkpoint('endpoint_search',artifact['validation'])
            if first<=2:
                solution,artifact = blocks.run_profile(backend,config,bank,endpoint,output,progress=progress,source=source_record(artifact))
            if first<=3:
                bar,artifact = blocks.run_bar(backend,config,bank,endpoint,solution,output,progress=progress,source=source_record(artifact))
        backend_name = source_record(artifact)['backend_name']
        blocks.run_propagation(config,bank,endpoint,solution,bar,output,progress=progress,
                               source=source_record(artifact),backend_name=backend_name,characterization=artifact.get('payload',{}).get('characterization'))
        if config.materials.sin_index_sensitivity_offsets:
            _run_index_sensitivity(config,endpoint,solution,output,cache_path,progress=progress,budget=budget)
        progress.finish()
        return output
    except Exception as exc:
        progress.fail(exc)
        _write_failure_report(output/'run_failure.json','vlac run',exc,progress)
        raise


def _run_endpoint_only(
    config_path: str | Path,
    out_path: str | Path | None = None,
    *,
    api_path: str | None = None,
    hide_gui: bool | None = None,
    platform: str | None = None,
    endpoint_method: str | None = None,
    block0_dir: str | Path | None = None,
    execution=None, characterize=False,
) -> Path:
    """Run the configured Python endpoint search and export block 1."""
    import csv

    config = _apply_runtime_overrides(
        load_config(config_path), api_path=api_path, hide_gui=hide_gui, platform=platform, block0_dir=block0_dir
    )
    if endpoint_method is not None:
        config = replace(config, endpoint_search=replace(config.endpoint_search, method=endpoint_method))
    config = _execution_overrides(config, execution, characterize)
    config.validate()
    output = Path(out_path) if out_path else config.project_root / "outputs" / "endpoint"
    blocks.invalidate(output,1)
    if not output.is_absolute():
        output = (config.project_root / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_path = resolved_cache(config)
    progress = ProgressReporter(
        "vlac endpoint", output,
        context={"config_path": str(Path(config_path).resolve()), "api_path": config.solver.api_path,
                 "hide_gui": config.solver.hide_gui, "platform": config.solver.platform,
                 "scope": "simulation block 1 only"},
    )
    stages = ["MODE API/license preflight"]
    if config.endpoint_search.method in {"python_pso", "adaptive_1d"}:
        verify = "Verify adaptive endpoint best candidate" if config.endpoint_search.method == "adaptive_1d" else "Verify Python PSO best candidate"
        stages += ["Precompute state-independent lower-guide modes", "Endpoint detuning search",verify]
    else:
        stages += ["Endpoint detuning search"]
    progress.set_plan(stages + ["Endpoint-only export"])
    progress.checkpoint("configuration", config.as_dict())
    try:
        bank = build_material_bank(config)
        progress.start_stage("MODE API/license preflight", total=1, unit="startup checks")
        with _backend(config, cache_path=cache_path, progress=progress) as backend:
            backend.preflight(0.5 * (config.device.wavelength_min_m + config.device.wavelength_max_m))
            progress.update(1, detail="MODE session started; API/license launch succeeded")
            progress.finish_stage()
            endpoint = search_endpoints(backend, config, bank, progress=progress, output_dir=output)
            block_artifact = blocks.export_endpoint(output,config,endpoint,backend_name=backend.backend_name,backend=backend)
        progress.start_stage("Endpoint-only export", total=1, unit="output bundles")
        with (output / "endpoint_detunings.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(endpoint.endpoint_records[0]))
            writer.writeheader()
            writer.writerows(endpoint.endpoint_records)
        (output / "configuration.json").write_text(
            json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        files = {"endpoint_detunings_csv": str(output / "endpoint_detunings.csv")}
        from .endpoint_diagnostics import FILES
        files.update({Path(name).stem + '_' + Path(name).suffix[1:]: str(output/name)
                      for name in FILES if (output/name).is_file()})
        if endpoint.search_method == "python_pso":
            files.update({name: str(output / filename) for name, filename in (
                ("particles_csv", "endpoint_pso_particles.csv"),
                ("pso_history_csv", "endpoint_pso_history.csv"),
                ("generation_detunings_csv", "endpoint_pso_generation_detunings.csv"),
                ("pso_verification_json", "endpoint_pso_verification.json"),
                ("pso_checkpoint_json", "endpoint_pso_checkpoint.json"),
            )})
        if endpoint.search_method == "adaptive_1d":
            files.update({name: str(output/filename) for name,filename in (
                ("width_samples_csv","endpoint_width_samples.csv"),
                ("pairs_csv","endpoint_pairs.csv"),
                ("best_json","endpoint_adaptive_best.json"),
                ("width_detunings_csv","endpoint_width_detunings.csv"),
                ("search_history_json","endpoint_adaptive_history.json"),
                ("verification_json","endpoint_adaptive_verification.json"),
                ("search_summary_json","endpoint_adaptive_summary.json"),
            )})
        summary = {
            "status": "passed", "scope": "simulation block 1 only",
            "optimization_objective": "maximize minimum absolute real detuning subject to increasing upper width, negative cross input, positive cross output and cross magnitude balance at both band edges",
            "cross_state_sign_verification": require_cross_state_signs(endpoint.endpoint_records),
            "endpoint_constraint_verification": require_endpoint_constraints(endpoint.endpoint_records, config.endpoint_search),
            "search_method": endpoint.search_method,
            "delta_beta_max_rad_per_m": endpoint.delta_beta_max_rad_per_m,
            "w1_input_m": endpoint.w1_input_m, "w1_output_m": endpoint.w1_output_m,
            "h_rib_m": endpoint.h_rib_m, "h_gap_m": endpoint.h_gap_m,
            "width_bounds_m": [config.geometry.w1_min_m, config.geometry.w1_max_m],
            "wavelengths_m": endpoint.wavelengths_m.tolist(),
            "endpoint_detunings": _endpoint_detuning_summary(endpoint.endpoint_records),
            "pso_summary": endpoint.pso_summary, "search_summary": endpoint.search_summary or endpoint.pso_summary,
            "isolated_mode_domain": endpoint.isolated_mode_domain,
            "files": files,
            "diagnostics": block_artifact['validation']['diagnostics'],
        }
        (output / "endpoint_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        metadata = {
            "command": "vlac endpoint", "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(Path(config_path).resolve()),
            "backend": "Ansys Lumerical MODE/FDE via Python API", "endpoint_search_method": endpoint.search_method,
            "config": config.as_dict(), "progress_status": str(progress.status_path),
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        progress.checkpoint("endpoint_search", summary)
        progress.update(1, detail=f"block-1 outputs saved to {output}")
        progress.finish_stage()
        progress.finish()
        return output
    except Exception as exc:
        progress.fail(exc)
        _write_failure_report(output / "endpoint_failure.json", "vlac endpoint", exc, progress)
        raise


def _endpoint_detuning_summary(records: list[dict]) -> dict:
    result = {}
    for state in ("cross", "bar"):
        for end, label in (("input", "in"), ("output", "out")):
            selected = [
                row for row in records
                if row["state"] == state and row["end"] == end
            ]
            worst = min(
                selected,
                key=lambda row: abs(row["delta_beta_real_rad_per_m"]),
            )
            result[f"delta_beta_{state}_{label}_rad_per_m"] = {
                "signed_value": worst["delta_beta_real_rad_per_m"],
                "absolute_value": abs(worst["delta_beta_real_rad_per_m"]),
                "worst_band_edge_wavelength_m": worst["wavelength_m"],
            }
    return result


def _run_index_sensitivity(config, endpoint, solution, output, cache_path, *, progress=None, budget=None):
    """Evaluate the fixed optimized layout for user-supplied absolute SiN dn offsets."""
    rows = []
    if progress:
        progress.start_stage(
            "SiN index sensitivity",
            total=len(config.materials.sin_index_sensitivity_offsets),
            unit="index offsets",
        )
    for offset in config.materials.sin_index_sensitivity_offsets:
        varied_materials = replace(
            config.materials,
            sin_absolute_index_offset=(
                config.materials.sin_absolute_index_offset + float(offset)
            ),
        )
        varied_config = replace(config, materials=varied_materials)
        varied_bank = build_material_bank(varied_config)
        with _backend(varied_config, cache_path=cache_path, progress=progress) as backend:
            if budget is not None: backend.budget = budget
            for state in ("cross", "bar"):
                if progress:
                    progress.note(f"evaluating dn={offset:+g} in {state} state", force=True)
                coefficients = evaluate_design_profiles(
                    backend,
                    varied_config,
                    varied_bank,
                    endpoint,
                    solution.w1_control_m,
                    solution.d_c2c_control_m,
                    solution.z_m,
                    solution.wavelengths_m,
                    state=state,
                )
                metrics = propagate_state(
                    solution.z_m,
                    coefficients,
                    solution.wavelengths_m,
                    state=state,
                    settings=config.propagation,
                )
                for row in metrics:
                    rows.append(
                        {
                            "sin_absolute_index_offset": offset,
                            **row,
                        }
                    )
        if progress:
            progress.update(
                len(rows) // max(1, 4 * len(solution.wavelengths_m)),
                total=len(config.materials.sin_index_sensitivity_offsets),
                detail=f"completed dn={offset:+g}",
            )
    if rows:
        import csv

        with (output / "sin_index_sensitivity.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if progress:
        progress.finish_stage(f"saved {len(rows)} sensitivity rows")


def _execution_args(args):
    return {key: getattr(args, key, None) for key in ('max_solves', 'max_seconds', 'no_resume', 'cache_dir')}


def _execution_overrides(config, execution=None, characterize=False):
    settings = dict(execution or {})
    cache_dir = settings.pop('cache_dir', None)
    no_resume = settings.pop('no_resume', False)
    updates = {k:v for k,v in settings.items() if v is not None}
    if no_resume: updates['resume'] = False
    config = replace(config, execution=replace(config.execution, **updates))
    if cache_dir:
        config = replace(config, output=replace(config.output, cache_sqlite=str(Path(cache_dir).resolve()/Path(config.output.cache_sqlite).name)))
    if characterize:
        config = replace(config, characterization=replace(config.characterization, enabled=True))
    return config


def run_workflow(config_path, out_path=None, **kwargs):
    config = load_config(config_path)
    output = _output_path(config, str(out_path) if out_path else None)
    with output_attempt(output, 'vlac run') as attempt:
        _run_workflow(config_path, attempt, **kwargs)
    return output


def run_endpoint_only(config_path, out_path=None, **kwargs):
    config = load_config(config_path)
    output = Path(out_path) if out_path else config.project_root/'outputs'/'endpoint'
    if not output.is_absolute(): output = (config.project_root/output).resolve()
    with output_attempt(output, 'vlac endpoint') as attempt:
        _run_endpoint_only(config_path, attempt, **kwargs)
    return output


def _add_solver_runtime_args(parser: argparse.ArgumentParser) -> None:
    if parser.prog.split()[-1] not in {'block0', 'verify-block0'}:
        parser.add_argument('--max-solves', type=int)
        parser.add_argument('--max-seconds', type=float)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--cache-dir')
    parser.add_argument('--characterize', action='store_true', help='Retain directly checked broadband cross/bar grids for the frozen profile.')
    parser.add_argument('--block0-dir', default=None, help='Passed Block 0 directory or certificate.json; validates and applies its solver profiles.')
    parser.add_argument(
        "--api-path",
        default=None,
        help="Override solver.api_path / LUMERICAL_PYTHON_API for this run.",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="Override the Lumerical Qt platform plugin (for example, windows).",
    )
    gui = parser.add_mutually_exclusive_group()
    gui.add_argument("--hide-gui", dest="hide_gui", action="store_true")
    gui.add_argument("--show-gui", dest="hide_gui", action="store_false")
    parser.set_defaults(hide_gui=None)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vlac")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run endpoint search, optimize, propagate, export")
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--out", default=None)
    resume = run.add_mutually_exclusive_group()
    resume.add_argument("--endpoint-dir",help="Load completed Block 1 and run Blocks 2–4.")
    resume.add_argument("--profile-dir",help="Load completed Block 2 and run Blocks 3–4.")
    resume.add_argument("--bar-dir",help="Load completed Block 3 and run Block 4.")
    run.add_argument(
        "--endpoint-method",
        choices=("adaptive_1d", "python_grid", "python_pso"),
        default=None,
        help="Override endpoint_search.method for this run.",
    )
    endpoint = subparsers.add_parser(
        "endpoint", aliases=["block1"],
        help="run endpoint width search with direct MODE FDE and delta-beta export",
    )
    endpoint.add_argument("--config", default=str(DEFAULT_CONFIG))
    endpoint.add_argument("--out", default=None)
    endpoint.add_argument("--endpoint-method", choices=("adaptive_1d", "python_pso", "python_grid"), default=None)
    _add_solver_runtime_args(endpoint)
    validate = subparsers.add_parser("validate", help="run analytic/unit validation")
    validate.add_argument("--config", default=str(DEFAULT_CONFIG))
    validate.add_argument("--out", default=None)
    validate_fde = subparsers.add_parser(
        "validate-fde",
        help="run local MODE mesh/domain, endpoint-domain, overlap and large-gap checks",
    )
    validate_fde.add_argument("--config", default=str(DEFAULT_CONFIG))
    validate_fde.add_argument("--out", default=None)
    _add_solver_runtime_args(validate_fde)
    validate_pso = subparsers.add_parser(
        "validate-pso",
        help="run a small Python PSO with MODE FDE and verify its best endpoint",
    )
    validate_pso.add_argument("--config", default=str(DEFAULT_CONFIG))
    validate_pso.add_argument("--out", default=None)
    _add_solver_runtime_args(validate_pso)
    validate_endpoint = subparsers.add_parser("validate-endpoint", help="run a small adaptive width search with ten fresh verification solves")
    validate_endpoint.add_argument("--config", default=str(DEFAULT_CONFIG))
    validate_endpoint.add_argument("--out", default=None)
    _add_solver_runtime_args(validate_endpoint)
    _add_solver_runtime_args(run)
    validate_one = subparsers.add_parser("validate-block1",help="validate/export only Block 1 (five initial widths, 13-width budget)")
    validate_one.add_argument("--config",default=str(DEFAULT_CONFIG))
    validate_one.add_argument("--out",default="outputs/block1_validation",help="Output directory.")
    _add_solver_runtime_args(validate_one)
    for number,flag,label in ((2,"--endpoint-dir","cross-state profile optimization"),
                              (3,"--profile-dir","bar-state extraction"),(4,"--bar-dir","two-mode propagation")):
        for validation in (False,True):
            name = f"validate-block{number}" if validation else f"block{number}"
            stage = subparsers.add_parser(name,help=f"{'validate/export' if validation else 'run/export'} only Block {number}: {label}")
            stage.set_defaults(block_number=number,block_validation=validation)
            stage.add_argument("--config",default=str(DEFAULT_CONFIG))
            stage.add_argument("--out",default=None,help="Output directory.")
            stage.add_argument(flag,required=True,help="Completed upstream block directory or its blockN_result.json.")
            _add_solver_runtime_args(stage)
            if number==2:
                stage.add_argument("--complete-within-seconds",type=float,default=None,
                                   help="Plan Powell work from measured costs, reserving final grid checks and exports.")
                stage.add_argument("--max-evaluations",type=int,default=None,help="Override per-grid optimizer evaluation budget.")
                stage.add_argument("--max-refinements",type=int,default=None,help="Override profile grid refinement budget.")
            if number==4:
                stage.add_argument("--rk4-steps-initial",type=int,default=None)
                stage.add_argument("--rk4-steps-max",type=int,default=None)
                stage.add_argument("--power-tolerance",type=float,default=None)
    compare = subparsers.add_parser(
        "compare-full-device",
        help="compare local-mode output CSV with EME/varFDTD/FDTD reference CSV",
    )
    compare.add_argument("local_csv")
    compare.add_argument("reference_csv")
    compare.add_argument("--out", default="full_device_comparison.json")
    calibration = subparsers.add_parser('block0', help='calibrate MODE domains and meshes with bounded, restartable native FDE tests')
    calibration.add_argument('--config', default=str(DEFAULT_CONFIG))
    calibration.add_argument('--out', required=True)
    calibration.add_argument('--method', choices=('full', 'reduced'), default='full',
                             help='Full certification, or scoped nine-setting convergence with a 200-call lifetime cap.')
    calibration.add_argument('--cases', help='Reduced method: JSON with four SI stress cases per family.')
    calibration.add_argument('--plan-only', action='store_true', help='Print the reduced plan without launching MODE or writing outputs.')
    calibration.add_argument('--trial', action='store_true', help='Two-state diagnostic only; cannot issue production profiles.')
    calibration.add_argument('--resume', action='store_true')
    calibration.add_argument('--max-solves', type=int)
    calibration.add_argument('--max-seconds', type=float)
    _add_solver_runtime_args(calibration)
    verify_zero = subparsers.add_parser('verify-block0', help='compare selected endpoints or a saved profile against the Block 0 reference')
    verify_zero.add_argument('--config', default=str(DEFAULT_CONFIG))
    verify_zero.add_argument('--out', required=True)
    verify_zero.add_argument('--max-solves', type=int)
    verify_zero.add_argument('--max-seconds', type=float)
    source_zero = verify_zero.add_mutually_exclusive_group(required=True)
    source_zero.add_argument('--endpoint-dir')
    source_zero.add_argument('--profile-dir')
    _add_solver_runtime_args(verify_zero)
    for name in ('inspect', 'report', 'plan', 'cache'):
        operation = subparsers.add_parser(name, help='Offline '+name+'; MODE is not required')
        operation.add_argument('--config', default=str(DEFAULT_CONFIG))
        if name in ('inspect', 'report'):
            operation.add_argument('--source-dir', required=True)
            operation.add_argument('--block', type=int, choices=(1,2,3,4) if name=='inspect' else (2,3,4), required=True)
        if name == 'report': operation.add_argument('--out', required=True)
        if name == 'plan': operation.add_argument('--profile-dir')
        if name == 'cache':
            operation.add_argument('--max-bytes', type=int, help='Explicitly prune least recently used entries to this payload size.')
            operation.add_argument('--vacuum', action='store_true', help='Reclaim unused SQLite disk pages after pruning.')
    characterize = subparsers.add_parser('characterize', help='Retain directly checked broadband coefficients on frozen Block 2 geometry')
    characterize.set_defaults(block_number=3, block_validation=False)
    characterize.add_argument('--config', default=str(DEFAULT_CONFIG))
    characterize.add_argument('--profile-dir', required=True)
    characterize.add_argument('--out')
    _add_solver_runtime_args(characterize)
    characterize.set_defaults(characterize=True)
    sensitivity = subparsers.add_parser('sensitivity', help='Evaluate process-supplied cases on frozen Block 2 controls')
    sensitivity.add_argument('--config', default=str(DEFAULT_CONFIG))
    sensitivity.add_argument('--profile-dir', required=True)
    sensitivity.add_argument('--cases', required=True, help='JSON with process source and named SI perturbations')
    sensitivity.add_argument('--out', required=True)
    _add_solver_runtime_args(sensitivity)
    args = parser.parse_args(argv)
    if args.command == 'sensitivity':
        from .sensitivity import load_cases, run as run_sensitivity
        try:
            config = _apply_runtime_overrides(load_config(args.config), api_path=args.api_path, hide_gui=args.hide_gui,
                                              platform=args.platform, block0_dir=args.block0_dir)
            config = _execution_overrides(config, _execution_args(args))
            result = run_sensitivity(config, args.profile_dir, load_cases(args.cases), args.out, _backend)
            print(f"Sensitivity completed: {len(result['cases'])} cases; status {result['status']}; {args.out}")
            return 0
        except Exception as exc:
            print(f'Sensitivity failed: {exc}', file=sys.stderr)
            return 2
    if args.command in {'inspect', 'report', 'plan', 'cache'}:
        from . import operations
        config = load_config(args.config)
        try:
            if args.command == 'inspect': result = operations.inspect(config, args.source_dir, args.block)
            elif args.command == 'report': result = {'output': str(operations.report(config, args.source_dir, args.block, args.out))}
            elif args.command == 'plan': result = operations.plan(config, args.profile_dir)
            else:
                from .cache import SQLiteCache
                cache = SQLiteCache(resolved_cache(config))
                if args.max_bytes is not None: cache.prune(args.max_bytes)
                if args.vacuum: cache.vacuum()
                result = cache.inspect()
            print(json.dumps(result, indent=2))
            return 0
        except Exception as exc:
            print(f'{args.command} failed: {exc}', file=sys.stderr)
            return 2
    if args.command in {'block0', 'verify-block0'}:
        from .block0 import run_calibration, verify_saved_design
        try:
            config = _apply_runtime_overrides(load_config(args.config), api_path=args.api_path,
                hide_gui=args.hide_gui, platform=args.platform, block0_dir=args.block0_dir)
            updates = {k: v for k, v in (('max_solves', args.max_solves), ('max_seconds', args.max_seconds)) if v is not None}
            if args.command == 'block0' and args.method == 'reduced' and args.max_seconds is None:
                updates['max_seconds'] = min(config.convergence.max_seconds, 7200.)
            config = replace(config, convergence=replace(config.convergence, **updates))
            config.validate()
            if args.command == 'block0':
                if args.method == 'reduced':
                    from .block0_reduced import prepare_plan, run_reduced
                    if args.trial:
                        raise ValueError('--trial and --method reduced are separate alternatives.')
                    if args.plan_only:
                        print(json.dumps(prepare_plan(config, args.cases), indent=2))
                        return 0
                    report = run_reduced(config, args.out, cases=args.cases, resume=args.resume)
                else:
                    if args.cases or args.plan_only:
                        raise ValueError('--cases and --plan-only require --method reduced.')
                    report = run_calibration(config, args.out, trial=args.trial, resume=args.resume)
            else:
                report = verify_saved_design(config, args.out, endpoint_dir=args.endpoint_dir, profile_dir=args.profile_dir)
            print(f'Block 0 {report["status"]}: {Path(args.out).resolve()}')
            return {'passed': 0, 'incomplete': 3, 'failed': 2}[report['status']]
        except Exception as exc:
            print(f'Block 0 failed: {type(exc).__name__}: {exc}', file=sys.stderr)
            return 2
    if args.command=="block1": args.command="endpoint"
    if args.command=="validate-block1":
        args.out = str(Path(args.out)/"block1_validation_report.json")
        args.command = "validate-endpoint"
    if hasattr(args,"block_number"):
        number = args.block_number
        output = Path(args.out) if args.out else None
        progress = None
        try:
            config = _apply_runtime_overrides(load_config(args.config),api_path=args.api_path,hide_gui=args.hide_gui,platform=args.platform,block0_dir=args.block0_dir)
            config = _execution_overrides(config, _execution_args(args), args.characterize)
            if number==2:
                updates = {key:value for key,value in (("max_function_evaluations_per_grid",args.max_evaluations),
                                                      ("complete_within_seconds",args.complete_within_seconds),
                                                      ("max_grid_refinements",args.max_refinements)) if value is not None}
                config = replace(config,optimizer=replace(config.optimizer,**updates))
            if number==4:
                updates = {key:value for key,value in (("rk4_steps_initial",args.rk4_steps_initial),("rk4_steps_max",args.rk4_steps_max),
                                                      ("power_convergence_rtol",args.power_tolerance)) if value is not None}
                config = replace(config,propagation=replace(config.propagation,**updates))
            config.validate()
            output = output or config.project_root/"outputs"/(args.command.replace("validate-","")+"_validation" if args.block_validation else args.command)
            progress = ProgressReporter("vlac "+args.command,output,context={"config_path":str(Path(args.config).resolve()),"block":number})
            source = getattr(args,{2:"endpoint_dir",3:"profile_dir",4:"bar_dir"}[number])
            artifact = blocks.run_single_block(number,config,output,source,backend_factory=_backend,progress=progress,
                                                validation_only=args.block_validation)
            status = artifact["validation"]["overall_status"]
            print(f"Block {number} completed; validation status: {status}; outputs: {output}")
            return 3 if args.block_validation and status!="passed" else 0
        except Exception as exc:
            if progress is not None:
                progress.fail(exc)
                _write_failure_report(output/f"block{number}_failure.json","vlac "+args.command,exc,progress)
                if args.block_validation:
                    write_json(output/f"block{number}_attempt_validation.json",{"block":number,"status":"incomplete" if isinstance(exc,BudgetExceeded) else "failed",
                        "execution_completed":False,"error_type":type(exc).__name__,"error":str(exc),
                        "failure_report":str(output/f"block{number}_failure.json")})
            print(f"Block {number} stopped: {exc}",file=sys.stderr)
            return 3 if isinstance(exc,BudgetExceeded) else 2


    if args.command == "validate":
        config = load_config(args.config)
        out = Path(args.out) if args.out else _output_path(config) / "validation_report.json"
        progress = ProgressReporter(
            "vlac validate",
            out.parent,
            context={"config_path": str(Path(args.config).resolve())},
        )
        progress.checkpoint("configuration", config.as_dict())
        progress.set_plan(["Analytic validation", "Write validation report"])
        try:
            report = run_analytic_validation(config, progress=progress)
            report["run_metadata"] = {
                "command": "vlac validate",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config_path": str(Path(args.config).resolve()),
                "config": config.as_dict(),
                "progress_status": str(progress.status_path),
            }
            progress.start_stage("Write validation report", total=1, unit="reports")
            write_validation_report(report, out)
            progress.update(1, detail=str(out))
            progress.finish_stage()
            progress.checkpoint("validation_report", {"path": str(out)})
            progress.finish()
            print(json.dumps(report, indent=2, default=str))
            print(f"\nValidation report: {out}")
            return 0
        except Exception as exc:
            progress.fail(exc)
            raise
    if args.command == "validate-fde":
        config = _apply_runtime_overrides(
            load_config(args.config),
            api_path=args.api_path,
            hide_gui=args.hide_gui,
            platform=args.platform,
            block0_dir=args.block0_dir,
        )
        config = _execution_overrides(config, _execution_args(args))
        out = Path(args.out) if args.out else _output_path(config) / "fde_validation_report.json"
        progress = ProgressReporter(
            "vlac validate-fde",
            out.parent,
            context={
                "config_path": str(Path(args.config).resolve()),
                "api_path": config.solver.api_path,
                "hide_gui": config.solver.hide_gui,
                "platform": config.solver.platform,
            },
        )
        progress.checkpoint("configuration", config.as_dict())
        progress.set_plan(
            [
                "MODE API/license preflight",
                "FDE mesh convergence",
                "FDE domain convergence",
                "Endpoint isolated-domain comparison",
                "Mode tracking sweep",
                "Large-separation FDE check",
                "Write FDE validation report",
            ]
        )
        try:
            report = run_fde_validation(config, progress=progress)
        except LumapiUnavailable as exc:
            progress.fail(exc)
            partial = out.with_name(f"{out.stem}_partial.json")
            _write_failure_report(partial, "vlac validate-fde", exc, progress)
            print(f"FDE validation not started: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            progress.fail(exc)
            partial = out.with_name(f"{out.stem}_partial.json")
            _write_failure_report(partial, "vlac validate-fde", exc, progress)
            raise
        try:
            report["run_metadata"] = {
                "command": "vlac validate-fde",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config_path": str(Path(args.config).resolve()),
                "config": config.as_dict(),
                "progress_status": str(progress.status_path),
            }
            progress.start_stage("Write FDE validation report", total=1, unit="reports")
            write_validation_report(report, out)
            progress.update(1, detail=str(out))
            progress.finish_stage()
            progress.checkpoint("fde_validation_report", {"path": str(out)})
            progress.finish()
            print(json.dumps(report, indent=2, default=str))
            print(f"\nFDE validation report: {out}")
            return 0
        except Exception as exc:
            progress.fail(exc)
            raise
    if args.command in {"validate-pso", "validate-endpoint"}:
        config = _apply_runtime_overrides(
            load_config(args.config),
            api_path=args.api_path,
            hide_gui=args.hide_gui,
            platform=args.platform,
            block0_dir=args.block0_dir,
        )
        config = _execution_overrides(config, _execution_args(args))
        adaptive = args.command == "validate-endpoint"
        config = replace(config, endpoint_search=replace(config.endpoint_search, method="adaptive_1d" if adaptive else "python_pso"))
        config.validate()
        search = search_endpoints_adaptive if adaptive else search_endpoints_python_pso
        verify_stage = "Verify adaptive endpoint best candidate" if adaptive else "Verify Python PSO best candidate"
        out = Path(args.out) if args.out else _output_path(config) / ("endpoint_validation_report.json" if adaptive else "endpoint_pso_validation_report.json")
        blocks.invalidate(out.parent,1)
        progress = ProgressReporter(
            f"vlac {args.command}",
            out.parent,
            context={
                "config_path": str(Path(args.config).resolve()),
                "api_path": config.solver.api_path,
                "hide_gui": config.solver.hide_gui,
                "platform": config.solver.platform,
                "validation_method": "adaptive_1d" if adaptive else "python_pso",
            },
        )
        progress.set_plan(["MODE API/license preflight", "Precompute state-independent lower-guide modes",
                           "Endpoint detuning search", verify_stage])
        progress.checkpoint("configuration", config.as_dict())
        report = {
            "command": f"vlac {args.command}",
            "status": "running",
            "validation_scope": ("adaptive width search with five initial samples and a 13-width budget, plus ten fresh verification modes"
                                 if adaptive else "one-generation constrained Python PSO with ten fresh final mode solves"),
            "cross_state_sign_requirement": dict(CROSS_STATE_SIGN_REQUIREMENT),
            "edge_wavelength_assumption": "only wavelength_min_m and wavelength_max_m are used by the endpoint objective",
            "lower_mode_assumption": "one cross-state lower-guide mode per band edge is reused in both liquid states",
        }
        try:
            with _backend(
                config,
                cache_path=resolved_cache(config),
                progress=progress,
            ) as backend:
                progress.start_stage("MODE API/license preflight", total=1, unit="startup checks")
                backend.preflight(0.5 * (config.device.wavelength_min_m + config.device.wavelength_max_m))
                progress.update(1)
                progress.finish_stage()
                endpoint = search(
                    backend,
                    config,
                    build_material_bank(config),
                    progress=progress,
                    validation_only=True,
                    output_dir=out.parent,
                )
                block_artifact = blocks.export_endpoint(out.parent,config,endpoint,backend_name=backend.backend_name,backend=backend)
            write_json(out.parent/"block1_validation_report.json",block_artifact["validation"])
            report.update({
                "status": "passed",
                "best_w1_input_m": endpoint.w1_input_m,
                "best_w1_output_m": endpoint.w1_output_m,
                "verified_delta_beta_max_rad_per_m": endpoint.delta_beta_max_rad_per_m,
                "cross_state_sign_verification": require_cross_state_signs(endpoint.endpoint_records),
                "endpoint_constraint_verification": require_endpoint_constraints(endpoint.endpoint_records, config.endpoint_search),
                "endpoint_case_records": len(endpoint.endpoint_records),
                "pso_summary": endpoint.pso_summary,
                "search_summary": endpoint.search_summary or endpoint.pso_summary,
            })
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            progress.finish()
            print(json.dumps(report, indent=2))
            print(f"\nEndpoint validation report: {out}")
            return 0
        except Exception as exc:
            progress.fail(exc)
            report.update({
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "progress_log": str(progress.log_path),
            })
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Endpoint validation failed: {exc}", file=sys.stderr)
            return 2
    if args.command == "endpoint":
        try:
            output = run_endpoint_only(
                args.config,
                args.out,
                api_path=args.api_path,
                hide_gui=args.hide_gui,
                platform=args.platform,
                endpoint_method=args.endpoint_method,
                block0_dir=args.block0_dir,
                execution=_execution_args(args), characterize=args.characterize,
            )
        except LumapiUnavailable as exc:
            print(f"MODE/FDE endpoint block not started: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Endpoint block stopped: {exc}", file=sys.stderr)
            return 3 if isinstance(exc,BudgetExceeded) else 2
        print(f"Endpoint block 1 completed. Outputs: {output}")
        return 0
    if args.command == "compare-full-device":
        report = compare_full_device_csv(args.local_csv, args.reference_csv)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "run":
        try:
            output = run_workflow(
                args.config,
                args.out,
                api_path=args.api_path,
                hide_gui=args.hide_gui,
                platform=args.platform,
                endpoint_method=args.endpoint_method,
                endpoint_dir=args.endpoint_dir,
                profile_dir=args.profile_dir,
                bar_dir=args.bar_dir,
                block0_dir=args.block0_dir,
                execution=_execution_args(args), characterize=args.characterize,
            )
        except Exception as exc:
            print(f"MODE/FDE run stopped: {exc}", file=sys.stderr)
            return 3 if isinstance(exc,BudgetExceeded) else 2
        print(f"MODE/FDE workflow completed. Outputs: {output}")
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    validators = {'validate-block1', 'validate-endpoint', 'validate-pso', 'validate-fde'}
    if not arguments or arguments[0] not in validators or any(v in arguments for v in ('--help','-h')):
        return _main(arguments)
    # Older validators accept a report filename; isolate their entire output directory too.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('command'); parser.add_argument('--config',default=str(DEFAULT_CONFIG)); parser.add_argument('--out')
    options, _ = parser.parse_known_args(arguments)
    config = load_config(options.config)
    names = {'validate-endpoint':'endpoint_validation_report.json','validate-pso':'endpoint_pso_validation_report.json',
             'validate-fde':'fde_validation_report.json'}
    if options.command == 'validate-block1':
        root = Path(options.out or 'outputs/block1_validation').resolve()
        filename = None
    else:
        report = Path(options.out) if options.out else _output_path(config)/names[options.command]
        root, filename = report.resolve().parent, report.name
    rewritten = []
    skip = False
    for arg in arguments:
        if skip: skip=False; continue
        if arg == '--out': skip=True; continue
        if arg.startswith('--out='): continue
        rewritten.append(arg)
    class ValidationStopped(RuntimeError): pass
    result = 2
    try:
        with output_attempt(root, 'vlac '+options.command) as attempt:
            result = _main(rewritten+['--out',str(attempt if filename is None else attempt/filename)])
            if result == 2: raise ValidationStopped('Validation failed; see attempt reports')
    except ValidationStopped:
        return result
    return result


if __name__ == "__main__":
    raise SystemExit(main())
