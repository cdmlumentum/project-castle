"""Shared width sampling with constrained endpoint-pair selection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator

from .endpoint import EndpointSearchResult
from .endpoint_constraints import (cross_state_sign_report, endpoint_constraint_report,
                                   require_endpoint_constraints)
from .modes import InvalidModeError, field_overlap
from .python_pso import EndpointEvaluator, _write_json, _write_csv


class WidthBudgetExhausted(Exception):
    """Internal guard against exceeding the shared distinct-width solve budget."""


class WidthEvaluator(EndpointEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshots = {}

    def evaluate_width(self, width):
        """Solve cross first; solve bar only if one endpoint's signs qualify."""
        rows, snapshots = [], {}
        kind = None
        for state in ("cross", "bar"):
            previous = None
            for wavelength in self.wavelengths:
                geometry = self.geometry(width, 0., state, wavelength)
                mode, overlap = self.backend.solve_isolated_mode(
                    geometry, wavelength, guide="upper", reference=previous,
                    compact_x=True, use_cache=True,
                )
                self.check_mode(mode, overlap, previous, geometry, wavelength, "upper")
                previous = mode
                snapshots[(state, wavelength)] = mode
                lower, lower_overlap = self.lower_modes[wavelength]
                upper_beta, lower_beta = 2*np.pi*mode.neff/wavelength, 2*np.pi*lower.neff/wavelength
                signed = upper_beta - lower_beta
                rows.append({
                    "h_gap_m": geometry.h_gap_m, "h_rib_m": geometry.h_rib_m,
                    "w1_m": float(width), "end": "width", "d_c2c_m": 0., "state": state,
                    "wavelength_m": wavelength,
                    "beta_upper_real_rad_per_m": float(upper_beta.real),
                    "beta_upper_imag_rad_per_m": float(upper_beta.imag),
                    "beta_lower_real_rad_per_m": float(lower_beta.real),
                    "beta_lower_imag_rad_per_m": float(lower_beta.imag),
                    "delta_beta_real_rad_per_m": float(signed.real),
                    "delta_beta_imag_rad_per_m": float(signed.imag),
                    "upper_mode_identity": mode.identity, "lower_mode_identity": lower.identity,
                    "upper_te_fraction": float(mode.te_fraction), "lower_te_fraction": float(lower.te_fraction),
                    "upper_tracking_overlap": float(overlap), "lower_tracking_overlap": float(lower_overlap),
                })
            if state == "cross":
                signs = [r["delta_beta_real_rad_per_m"] for r in rows]
                kind = "input" if all(v < 0 for v in signs) else "output" if all(v > 0 for v in signs) else None
                if kind is None:
                    self.snapshots[float(width)] = snapshots
                    return {"valid": False, "eligible_end": None, "score": None, "records": rows,
                            "error": "Cross-state detunings have mixed signs or zero across band edges."}
        self.snapshots[float(width)] = snapshots
        return {"valid": True, "eligible_end": kind, "score": min(abs(r["delta_beta_real_rad_per_m"]) for r in rows),
                "records": rows, "error": ""}

    def low_width_overlap(self, width_a, width_b):
        a, b = self.snapshots.get(width_a, {}), self.snapshots.get(width_b, {})
        return any(field_overlap(a[k], b[k]) < self.config.solver.minimum_mode_overlap for k in a.keys() & b.keys())


def _pair_records(sample_in, sample_out, config):
    records = []
    for end, sample, offset in (("input", sample_in, config.geometry.d_c2c_input_m),
                                ("output", sample_out, config.geometry.d_c2c_output_m)):
        records.extend(dict(row, end=end, d_c2c_m=offset) for row in sample["records"])
    return records


def _select_pair(samples, config):
    """Compare all directly solved pairs. Pair comparisons require no FDE calls."""
    best, nearest, rows = None, None, []
    inputs = [(w,s) for w,s in samples.items() if s["valid"] and s["eligible_end"] == "input"]
    outputs = [(w,s) for w,s in samples.items() if s["valid"] and s["eligible_end"] == "output"]
    for wi, si in inputs:
        for wo, so in outputs:
            records = _pair_records(si, so, config)
            report = endpoint_constraint_report(records, config.endpoint_search)
            mismatch = report["magnitude_balance"]["maximum_relative_mismatch"]
            score = min(si["score"], so["score"])
            row = {"w1_input_m": wi, "w1_output_m": wo, "valid": report["passed"],
                   "width_order_passed": report["width_order"]["passed"],
                   "cross_state_signs_passed": True,
                   "maximum_relative_mismatch": mismatch,
                   "magnitude_balance_passed": report["magnitude_balance"]["passed"],
                   "bottleneck_rad_per_m": score if report["passed"] else None}
            rows.append(row)
            if report["width_order"]["passed"] and (nearest is None or mismatch < nearest["maximum_relative_mismatch"]):
                nearest = row
            if report["passed"] and (best is None or (score,-mismatch) > (best["score"],-best["mismatch"])):
                best = {"widths": (wi,wo), "score": score, "mismatch": mismatch,
                        "records": records, "constraints": report}
    return best, nearest, rows


def _valid_branches(samples, evaluator):
    """Interpolate only contiguous valid, sign-consistent, tracked branches.

    No extrapolation or interpolation across an invalid sample, sign transition,
    or detected low field overlap is allowed. Values only propose new FDE widths.
    """
    branches, branch = [], []
    for width, sample in sorted(samples.items()):
        if not sample["valid"]:
            if branch: branches.append(branch)
            branch = []
            continue
        if branch and (sample["eligible_end"] != samples[branch[-1]]["eligible_end"]
                       or evaluator.low_width_overlap(branch[-1],width)):
            branches.append(branch)
            branch = []
        branch.append(width)
    if branch: branches.append(branch)
    return branches


def _surrogate_points(samples, evaluator):
    branches = _valid_branches(samples,evaluator)
    points = {end: [] for end in ("input","output")}
    for branch in branches:
        values = np.asarray([[r["delta_beta_real_rad_per_m"] for r in samples[w]["records"]] for w in branch])
        end = samples[branch[0]]["eligible_end"]
        if len(branch) == 1:
            points[end].append((branch[0],values[0]))
            continue
        # Scale x to micrometres to keep interpolation coordinates well sized.
        interpolator = PchipInterpolator(np.asarray(branch)*1e6,values,axis=0,extrapolate=False)
        grid = np.unique(np.concatenate([np.linspace(a,b,9) for a,b in zip(branch,branch[1:])]))
        for w,v in zip(grid,interpolator(grid*1e6)):
            points[end].append((float(w),v))
    return points


def _surrogate_pairs(samples, evaluator, config):
    """Rank predicted feasible pairs, then minimum-violation pairs if needed."""
    points = _surrogate_points(samples,evaluator)
    if not points["input"] or not points["output"]:
        return []
    wi = np.array([p[0] for p in points["input"]])
    wo = np.array([p[0] for p in points["output"]])
    vi = np.array([p[1] for p in points["input"]])
    vo = np.array([p[1] for p in points["output"]])
    a, b = np.abs(vi[:,:2])[:,None,:], np.abs(vo[:,:2])[None,:,:]
    mismatch = np.max(2*np.abs(a-b)/np.maximum(a+b,1e-30),axis=2)
    score = np.minimum(np.min(np.abs(vi),axis=1)[:,None],np.min(np.abs(vo),axis=1)[None,:])
    ordered = wi[:,None] < wo[None,:]
    signs = np.all(vi[:,:2]<0,axis=1)[:,None] & np.all(vo[:,:2]>0,axis=1)[None,:]
    indices = np.argwhere(ordered & signs)
    tolerance = config.endpoint_search.cross_magnitude_balance_rtol
    ranked = [(float(wi[i]),float(wo[j]),float(score[i,j]),float(mismatch[i,j])) for i,j in indices]
    # Invert the interpolated cross detunings to propose exact matches and
    # balance-boundary widths. A regular interpolation grid alone can miss a
    # narrow feasible strip, especially when a fixed endpoint is at a bound.
    ratio = (2+tolerance)/(2-tolerance)
    for branch in _valid_branches(samples,evaluator):
        if len(branch)<2: continue
        end = samples[branch[0]]["eligible_end"]
        other = "output" if end=="input" else "input"
        values = np.asarray([[r["delta_beta_real_rad_per_m"] for r in samples[w]["records"]] for w in branch])
        coords = np.asarray(branch)*1e6
        curve = PchipInterpolator(coords,values,axis=0,extrapolate=False)
        edge_curves = [PchipInterpolator(coords,values[:,k],extrapolate=False) for k in (0,1)]
        for fixed,sample in samples.items():
            if not sample["valid"] or sample["eligible_end"]!=other: continue
            fixed_values = np.array([r["delta_beta_real_rad_per_m"] for r in sample["records"]])
            roots = [coords[0],coords[-1]]
            for k,edge in enumerate(edge_curves):
                sign = -1 if end=="input" else 1
                for factor in (1/ratio,1.,ratio):
                    roots.extend(float(r) for r in edge.solve(sign*abs(fixed_values[k])*factor,extrapolate=False)
                                 if np.isfinite(r) and coords[0]<=r<=coords[-1])
            roots = sorted(set(roots))
            candidates = roots+[(a+b)/2 for a,b in zip(roots,roots[1:])]
            for coordinate in candidates:
                w = coordinate*1e-6
                vi_fixed,vo_fixed = (curve(coordinate),fixed_values) if end=="input" else (fixed_values,curve(coordinate))
                wi_fixed,wo_fixed = (w,fixed) if end=="input" else (fixed,w)
                if wi_fixed>=wo_fixed or not np.all(vi_fixed[:2]<0) or not np.all(vo_fixed[:2]>0): continue
                a,b = np.abs(vi_fixed[:2]),np.abs(vo_fixed[:2])
                error = float(np.max(2*np.abs(a-b)/(a+b)))
                value = float(min(np.min(np.abs(vi_fixed)),np.min(np.abs(vo_fixed))))
                ranked.append((float(wi_fixed),float(wo_fixed),value,error))
    ranked.sort(key=lambda p: (p[3]>tolerance, -p[2] if p[3]<=tolerance else p[3], p[3]))
    return ranked


def search_endpoints_adaptive(backend, config, bank, *, progress=None, output_dir=None, validation_only=False, checkpoint_dir=None):
    """Optimize a constrained pair from shared width solves; freshly verify it."""
    config.validate()
    g, settings = config.geometry, config.endpoint_search
    if not config.solver.endpoint_compact_x:
        raise ValueError("adaptive_1d requires endpoint_compact_x=true for the common centered width table.")
    if g.h_gap_bounds_m[0] != g.h_gap_bounds_m[1] or g.h_rib_bounds_m[0] != g.h_rib_bounds_m[1]:
        raise ValueError("adaptive_1d requires fixed h_gap and h_rib.")
    if settings.pso_lower_mode_state_independent is not True:
        raise ValueError("adaptive_1d uses the configured state-independent lower-mode assumption.")
    if validation_only:
        settings = replace(settings,adaptive_initial_widths=5,adaptive_max_widths=13)
        config = replace(config,endpoint_search=settings)
    out = Path(output_dir) if output_dir is not None else config.resolved_path(config.output.directory)
    out.mkdir(parents=True,exist_ok=True)
    for name in ("endpoint_summary.json","endpoint_detunings.csv","endpoint_failure.json",
                 "endpoint_adaptive_best.json","endpoint_adaptive_verification.json","endpoint_adaptive_summary.json",
                 "endpoint_pso_best.json"):
        (out/name).unlink(missing_ok=True)
    for name in ("endpoint_width_samples.csv","endpoint_width_detunings.csv","endpoint_pairs.csv"):
        (out/name).write_text("",encoding="utf-8")
    _write_json(out/"endpoint_adaptive_history.json",{"refinements": []})
    evaluator = WidthEvaluator(backend,config,bank,progress=progress)
    from .checkpoints import Checkpoint
    from .provenance import checkpoint_identity
    from .runtime import checkpoint_directory
    from dataclasses import asdict
    policies = asdict(settings); policies.pop('adaptive_max_widths')
    from .solver import LumapiBackend
    directory = checkpoint_dir or (checkpoint_directory(config) if isinstance(backend, LumapiBackend) else None)
    checkpoint = Checkpoint(directory, 'adaptive-widths',
                            checkpoint_identity(config, 1, dict(policies, backend=getattr(backend, "backend_name", type(backend).__qualname__)))) if directory is not None else None
    saved = checkpoint.load() if checkpoint and config.execution.resume else None
    if saved:
        evaluator.snapshots = saved['snapshots']
        evaluator.lower_modes = saved['lower_modes']
    if progress:
        progress.start_stage("Precompute state-independent lower-guide modes",total=2,unit="band-edge modes")
    if not evaluator.lower_modes: evaluator.precompute_lower()
    if progress:
        progress.finish_stage()
        progress.start_stage("Endpoint detuning search",total=settings.adaptive_max_widths,unit="distinct widths")
    samples, history = (saved["samples"], saved["history"]) if saved else ({}, [])
    tolerance = settings.adaptive_width_tolerance_m

    def save():
        if checkpoint:
            checkpoint.save({'samples': samples, 'history': history,
                             'snapshots': evaluator.snapshots, 'lower_modes': evaluator.lower_modes})
        records, summaries = [], []
        for width,sample in sorted(samples.items()):
            summaries.append({"w1_m": width,"valid": sample["valid"],"eligible_end": sample["eligible_end"],
                              "bottleneck_rad_per_m": sample["score"],"error": sample["error"]})
            records.extend(sample["records"])
        _write_csv(out/"endpoint_width_samples.csv",summaries)
        _write_csv(out/"endpoint_width_detunings.csv",records)

    def evaluate(width):
        width = float(np.clip(width,g.w1_min_m,g.w1_max_m))
        if width in samples: return
        if len(samples)>=settings.adaptive_max_widths: raise WidthBudgetExhausted()
        try:
            sample = evaluator.evaluate_width(width)
        except InvalidModeError as exc:
            sample = {"valid": False,"eligible_end": None,"score": None,"records": [],"error": str(exc)}
        samples[width] = sample
        save()
        if progress:
            progress.update(len(samples),detail=f"width {width*1e9:.3f} nm; " +
                            (f"{sample['eligible_end']} score {sample['score']:.6g} rad/m" if sample["valid"] else sample["error"]))

    def nearby(width):
        return min(abs(width-w) for w in samples)

    def localized(width):
        ordered = sorted(samples)
        i = ordered.index(width)
        return ((i==0 or width-ordered[i-1]<=2*tolerance)
                and (i==len(ordered)-1 or ordered[i+1]-width<=2*tolerance))

    for width in np.linspace(g.w1_min_m,g.w1_max_m,settings.adaptive_initial_widths):
        evaluate(width)
    boundary_budget = max(1,(settings.adaptive_max_widths-settings.adaptive_initial_widths)//4)
    for _ in range(boundary_budget):
        widths = sorted(samples)
        intervals = [(a,b) for a,b in zip(widths,widths[1:]) if b-a>tolerance
                     and (samples[a]["eligible_end"]!=samples[b]["eligible_end"]
                          or samples[a]["valid"]!=samples[b]["valid"] or evaluator.low_width_overlap(a,b))]
        if not intervals or len(samples)>=settings.adaptive_max_widths: break
        a,b = max(intervals,key=lambda ab:ab[1]-ab[0])
        evaluate((a+b)/2)

    converged, stable = False, 0
    while len(samples)<settings.adaptive_max_widths:
        best, nearest, pairs = _select_pair(samples,config)
        proposals = _surrogate_pairs(samples,evaluator,config)
        top = proposals[0] if proposals else None
        objective_stable = (best is not None and top is not None and top[3]<=settings.cross_magnitude_balance_rtol
                            and top[2]<=best["score"]*(1+settings.adaptive_objective_rtol)
                            and all(abs(a-b)<=tolerance for a,b in zip(top[:2],best["widths"])))
        stable = stable+1 if objective_stable else 0
        if stable>=2 and all(localized(w) for w in best["widths"]):
            converged = True
            break
        selected, reason = [], "interpolated_pair"
        for proposal in proposals:
            missing = [w for w in proposal[:2] if nearby(w)>tolerance/4]
            if missing:
                # Solve the least sampled member first. All subsequent decisions
                # use the updated direct table, never a surrogate-only winner.
                selected = sorted(missing,key=nearby,reverse=True)[:2]
                break
        if not selected:
            widths = sorted(samples)
            intervals = [(a,b) for a,b in zip(widths,widths[1:]) if b-a>tolerance/2]
            local = [ab for ab in intervals if best is not None and any(w in ab for w in best["widths"])]
            if local: intervals = local
            if not intervals: break
            a,b = max(intervals,key=lambda ab:ab[1]-ab[0])
            selected,reason = [(a+b)/2],"local_interval" if local else "interval_exploration"
        before = None if best is None else best["score"]
        for width in selected:
            if len(samples)>=settings.adaptive_max_widths: break
            evaluate(width)
        after,_,_ = _select_pair(samples,config)
        history.append({"widths_evaluated": len(samples),"reason": reason,"new_widths_m": selected,
                        "best_score_before_rad_per_m": before,"best_score_after_rad_per_m": None if after is None else after["score"],
                        "predicted_pair": None if top is None else {"widths_m": list(top[:2]),"score_rad_per_m": top[2],"relative_mismatch": top[3]}})
        _write_json(out/"endpoint_adaptive_history.json",{"refinements": history})
    best,nearest,pairs = _select_pair(samples,config)
    summary = {"backend": "adaptive_1d","selection_method": "constrained_pair_from_shared_width_table",
               "widths_evaluated": len(samples),"width_budget": settings.adaptive_max_widths,"converged": converged,
               "feasible": best is not None,
               "convergence_scope": "interpolation-guided local pair stability; no unsampled global-optimum guarantee",
               "stop_reason": "width_and_objective_tolerance" if converged else "width_budget_or_local_refinement_limit",
               "width_bounds_m": [g.w1_min_m,g.w1_max_m],"width_bounds_adjusted": False,"cutoff_preflight": False,
               "lower_mode_state_independent": True,"silicon_substrate_included": False,
               "slab_lateral_extent": "both FDE x boundaries",
               "input_feasible_samples": sum(s["valid"] and s["eligible_end"]=="input" for s in samples.values()),
               "output_feasible_samples": sum(s["valid"] and s["eligible_end"]=="output" for s in samples.values()),
               "feasible_sampled_pairs": sum(r["valid"] for r in pairs),"nearest_sampled_pair": nearest,
               "cross_magnitude_balance_rtol": settings.cross_magnitude_balance_rtol,"settings": settings.__dict__}
    _write_csv(out/"endpoint_pairs.csv",pairs)
    _write_json(out/"endpoint_adaptive_history.json",{"refinements": history})
    _write_json(out/"endpoint_adaptive_summary.json",summary)
    if best is None:
        missing = [end for end in ("input","output") if not any(s["valid"] and s["eligible_end"]==end for s in samples.values())]
        detail = f"no sampled sign-feasible {', '.join(missing)} width" if missing else "no sampled pair satisfying width order and cross-state magnitude balance"
        raise RuntimeError(f"Adaptive endpoint search found {detail}. Review endpoint_width_samples.csv and endpoint_pairs.csv. "
                           "Width bounds and balance tolerance were not changed; this sampled result does not prove the interval infeasible.")
    if progress:
        progress.finish_stage(f"{len(samples)} widths; constrained endpoint pair selected")
        progress.start_stage("Verify adaptive endpoint best candidate",total=10,unit="fresh FDE mode solves")
    verification = {"status": "running","fresh_fde_solves": True}
    try:
        evaluator.precompute_lower(use_cache=False)
        score,records = evaluator.evaluate(best["widths"],use_cache=False)
        constraints = require_endpoint_constraints(records,settings)
        expected = best["records"]
        errors = [abs(a["delta_beta_real_rad_per_m"]-b["delta_beta_real_rad_per_m"]) for a,b in zip(records,expected)]
        agrees = all(np.isclose(a["delta_beta_real_rad_per_m"],b["delta_beta_real_rad_per_m"],rtol=1e-6,atol=1e-3)
                     for a,b in zip(records,expected))
        verification.update(status="passed" if agrees else "failed",fresh_mode_solves=10,endpoint_records=records,
                            endpoint_constraint_verification=constraints,cross_state_sign_verification=constraints["cross_state_sign_verification"],
                            maximum_case_error_rad_per_m=max(errors),absolute_tolerance_rad_per_m=1e-3,relative_tolerance=1e-6)
        if not agrees: raise RuntimeError("Fresh endpoint FDE detunings disagree with the adaptive width table.")
    except Exception as exc:
        if hasattr(exc,"records"):
            verification.update(endpoint_records=exc.records,cross_state_sign_verification=cross_state_sign_report(exc.records),
                                endpoint_constraint_verification=endpoint_constraint_report(exc.records,settings))
        verification.update(status="failed",error=str(exc))
        _write_json(out/"endpoint_adaptive_verification.json",verification)
        raise
    _write_json(out/"endpoint_adaptive_verification.json",verification)
    backend._block1_mode_profiles = evaluator.mode_profiles
    summary.update(cross_state_endpoint_signs_verified=True,endpoint_constraints_verified=True,
                   endpoint_constraint_verification=constraints,delta_beta_max_rad_per_m=score,
                   w1_input_m=float(best["widths"][0]),w1_output_m=float(best["widths"][1]))
    _write_json(out/"endpoint_adaptive_best.json",summary)
    _write_json(out/"endpoint_adaptive_summary.json",summary)
    if progress: progress.finish_stage("ten fresh FDE solves; width order, signs and magnitude balance verified")
    return EndpointSearchResult(g.h_gap_bounds_m[0],g.h_rib_bounds_m[0],float(best["widths"][0]),float(best["widths"][1]),score,
                                np.asarray(evaluator.wavelengths),records,pairs,converged,history,
                                backend.endpoint_isolated_domain_info(),search_method="adaptive_1d",search_summary=summary)
