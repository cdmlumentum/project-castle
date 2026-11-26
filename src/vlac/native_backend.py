"""Thin adapter to the native LSF core, retaining v23's Python optimizers."""

from __future__ import annotations

from dataclasses import asdict, replace
from collections import OrderedDict
from .provenance import optical_identity
from importlib.resources import files
from pathlib import Path
import re

import numpy as np

from .block_io import digest as stable_key
from .materials import material_samples
from .modes import Coefficients, InvalidModeError, ModeReference, ModeSnapshot
from .solver import LumapiBackend


def native_script(name: str) -> str:
    if name not in {"vlac_core.lsf", "vlac_blocks.lsf", "vlac_convergence.lsf"}:
        raise ValueError("Unknown native script")
    return files("vlac").joinpath("native", name).read_text(encoding="utf-8")


def load_native_functions(session, name):
    text = native_script(name)
    starts = list(re.finditer(r"^function\s+(\w+)\(", text, flags=re.MULTILINE))
    for index, match in enumerate(starts):
        end = starts[index+1].start() if index+1 < len(starts) else len(text)
        try:
            session.putv("vlac_source", text[match.start():end])
            session.eval('vlac_script_error=""; try { eval(vlac_source); } catch(vlac_script_error);')
            error = session.getv("vlac_script_error")
            if error:
                raise RuntimeError(error)
        except Exception as exc:
            raise RuntimeError(f"Cannot load native function {match.group(1)} from {name}: {exc}") from exc


def column(values):
    return np.asarray(values).reshape(-1, 1)


def scalar(value):
    return np.asarray(value).squeeze().item()


def snapshot_from_native(value) -> ModeSnapshot:
    return ModeSnapshot(
        complex(scalar(value["neff"])), np.asarray(value["ex"], dtype=complex),
        np.asarray(value["ey"], dtype=complex), float(scalar(value["te_fraction"])),
        float(scalar(value["core_fraction"])), value["identity"],
    )


def snapshot_to_native(value: ModeSnapshot):
    """Preserve complex neff using the matrix type accepted by lumapi v221."""
    payload = asdict(value)
    payload["neff"] = np.array([[value.neff]], dtype=complex)
    return payload


def reference_to_native(value: ModeReference):
    return {"upper": snapshot_to_native(value.upper), "lower": snapshot_to_native(value.lower),
            "plus": snapshot_to_native(value.supermodes[0]), "minus": snapshot_to_native(value.supermodes[1])}


def pack_references(references):
    return {"n": float(len(references)),
            "items": [reference_to_native(ref) for ref in references] or [0.0]}


def coefficients_from_native(value):
    fields = {key: complex(scalar(value[key])) for key in (
        "beta_upper", "beta_lower", "beta_plus", "beta_minus", "delta_beta", "delta_pm", "delta_12", "kappa")}
    for key in ("kappa_profile_check", "overlap_upper_lower", "min_mode_overlap"):
        fields[key] = float(scalar(value[key]))
    if fields["kappa_profile_check"] < 0:
        fields["kappa_profile_check"] = float("nan")
    fields.update(upper_identity=value["upper_identity"], lower_identity=value["lower_identity"],
                  supermode_identities=(value["plus_identity"], value["minus_identity"]),
                  extraction_flag=value["extraction_flag"])
    result = Coefficients(**fields)
    numeric = [getattr(result, name) for name in (
        "beta_upper", "beta_lower", "beta_plus", "beta_minus", "delta_beta", "kappa")]
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("Native MODE returned non-finite coefficients")
    return result


def profile_request(endpoint, w_control, d_control, z, wavelengths, state):
    return {"w_control_m": column(w_control), "d_control_m": column(d_control),
            "z_m": column(z), "wavelengths_m": column(wavelengths), "state": state,
            "h_gap_m": float(endpoint.h_gap_m), "h_rib_m": float(endpoint.h_rib_m),
            "delta_scale": float(endpoint.delta_beta_max_rad_per_m)}


class NativeLumapiBackend(LumapiBackend):
    """Execute selectors, extraction and complete profile grids inside MODE.

    The Python adaptive endpoint search and bounded Powell algorithms remain
    unchanged. Native standalone algorithms are explicit alternatives in bundles.
    """

    backend_name = "ansys_lumerical_mode_native_lsf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.script_hash = stable_key({"core": native_script("vlac_core.lsf")})
        self._native_loaded = False

    def native_config(self, mesh_accuracy=None):
        cfg, bank = self.config, self.bank
        wavelengths = self._wavelength_samples_for_materials()
        compact = self.endpoint_isolated_domain_info(compact_x=True, mesh_accuracy=mesh_accuracy)
        nx, ny = self._mesh_cell_counts(mesh_accuracy)
        liquid_knots = wavelengths
        if cfg.materials.liquid_dispersion_csv:
            table = np.genfromtxt(cfg.resolved_path(cfg.materials.liquid_dispersion_csv), delimiter=",", names=True)
            knots = np.atleast_1d(table["wavelength_um"]) * 1e-6
            liquid_knots = np.unique(np.r_[wavelengths, knots[(knots >= wavelengths[0]) & (knots <= wavelengths[-1])]])
        payload = {
            "reuse_scene": float(cfg.solver.reuse_mode_project),
            "x_span_m": self.x_span_m, "y_min_m": self.y_min_m, "y_max_m": self.y_max_m,
            "mesh_cells_x": float(nx), "mesh_cells_y": float(ny),
            "compact_x_span_m": compact["x_span_m"], "compact_mesh_cells_x": float(compact["mesh_cells_x"]),
            "tracking_grid_nx": float(cfg.solver.tracking_grid_nx), "tracking_grid_ny": float(cfg.solver.tracking_grid_ny),
            "number_of_trial_modes": float(cfg.solver.number_of_trial_modes),
            "endpoint_lower_state_independent": float(cfg.endpoint_search.pso_lower_mode_state_independent and cfg.endpoint_search.method != 'python_grid'),
            "zero_mode_retries": float(cfg.solver.zero_mode_retries),
            "minimum_te_fraction": cfg.solver.minimum_te_fraction,
            "minimum_mode_overlap": cfg.solver.minimum_mode_overlap,
            "field_overlap_crosscheck_max_basis_overlap": cfg.solver.field_overlap_crosscheck_max_basis_overlap,
            "material_fit_max_coefficients": float(cfg.materials.material_fit_max_coefficients),
            "silica_table": material_samples(bank.silica, wavelengths).astype(complex),
            "sin_table": material_samples(bank.sin, wavelengths).astype(complex),
            "liquid_wavelengths_m": column(liquid_knots),
            "liquid_cross_n": column(bank.liquid_cross.index(liquid_knots)),
            "liquid_bar_n": column(bank.liquid_bar.index(liquid_knots)),
            "length_m": cfg.device.length_m,
            "geometry": {"w1_m": cfg.geometry.w1_min_m, "w2_m": cfg.geometry.w1_lower_m,
                         "h_wg_m": cfg.geometry.h_wg_m, "h_rib_m": cfg.geometry.h_rib_candidates_m[0],
                         "h_gap_m": cfg.geometry.h_gap_candidates_m[0], "d_c2c_m": 0.0,
                         "box_thickness_m": cfg.geometry.box_thickness_m, "state": "cross",
                         "liquid_index": cfg.materials.liquid_cross_n},
        }
        if cfg.solver.calibration:
            from .block0 import certified_profile
            payload['full_profile'] = certified_profile(cfg, 'coupled').native()
            payload['compact_profile'] = certified_profile(cfg, 'isolated').native()
            payload['full_profile'].update(mesh_cells_x=float(nx), mesh_cells_y=float(ny),
                                           x_span_m=self.x_span_m, y_min_m=self.y_min_m, y_max_m=self.y_max_m)
            payload['compact_profile'].update(mesh_cells_x=float(compact['mesh_cells_x']), mesh_cells_y=float(compact['mesh_cells_y']))
            payload['block0_certificate_id'] = cfg.solver.calibration['certificate_id']
        return payload

    def _call_native(self, function, arguments, *, mesh_accuracy=None):
        if function not in {"vlac_raw", "vlac_isolated", "vlac_coefficients", "vlac_profile",
                            "vlac_raw_batch", "vlac_coefficients_from_raw", "vlac_select_isolated"}:
            raise ValueError("Unsupported native entry point")
        session = self.session
        if not getattr(self, '_native_loaded', False):
            load_native_functions(session, "vlac_core.lsf")
            self._native_loaded = True
        cfg = self.native_config(mesh_accuracy)
        signature = stable_key(cfg)
        if getattr(self, '_uploaded_config', None) != signature:
            # A changed material fit or numerical profile invalidates the live scene.
            session.eval('newproject;')
            session.putv("vlac_cfg", cfg)
            self._uploaded_config = signature
        reserve = 0
        if function in {"vlac_raw", "vlac_isolated"}: reserve = 1
        if function == "vlac_coefficients": reserve = 3
        if function == "vlac_profile": reserve = 3*len(arguments[0]['z_m'])*len(arguments[0]['wavelengths_m'])
        if function == "vlac_raw_batch": reserve = len(arguments[0])
        reserve *= 1+self.config.solver.zero_mode_retries
        self._budget().check(reserve)
        variables = ["vlac_cfg"]
        for index, value in enumerate(arguments):
            name = f"vlac_arg_{index}"
            self.session.putv(name, value)
            variables.append(name)
        try:
            with self._timed("native_calculation"):
                self.session.eval("vlac_result=" + function + "(" + ",".join(variables) + ");")
                result = self.session.getv("vlac_result")
        except BaseException:
            # Exact attempts may be unavailable after an API/solver interruption.
            self._budget().consume(reserve)
            raise
        solves = int(scalar(result.get('solves', result.get('attempts', 0))))
        self._budget().consume(solves)
        if getattr(self, 'progress', None):
            self.progress.native_solves(solves)
        if not bool(scalar(result["ok"])):
            error = result.get("error", "native_calculation_failed")
            if error == "zero_modes":
                raise InvalidModeError("Native FDE returned zero modes after the configured clean-project retries")
            raise RuntimeError("Native MODE calculation failed: " + error)
        return result

    def _cache_identity(self, *args, **kwargs):
        return stable_key({"native_lsf": self.script_hash, "v23": super()._cache_identity(*args, **kwargs),
                           "native_config": self.native_config()})

    def _raw_request(self, geometry, wavelength, upper, lower, compact=False):
        return {'geometry': asdict(geometry), 'wavelength': float(wavelength),
                'upper': float(upper), 'lower': float(lower), 'compact': float(compact)}

    def _raw_key(self, request, mesh_accuracy=None):
        geometry = dict(request['geometry'])
        geometry.pop('state', None)  # Optical identity is the actual liquid index.
        if not request['upper']:
            geometry.pop('w1_m', None); geometry.pop('d_c2c_m', None)
        elif request['compact']:
            geometry['d_c2c_m'] = 0.0
        if not request['lower']: geometry.pop('w2_m', None)
        cfg = self.native_config(mesh_accuracy)
        for key in ('geometry','minimum_te_fraction','minimum_mode_overlap','field_overlap_crosscheck_max_basis_overlap',
                    'endpoint_lower_state_independent','length_m','liquid_cross_n','liquid_bar_n'):
            cfg.pop(key, None)
        return stable_key({'native_raw': self.script_hash, 'provenance': optical_identity(self.config),
                           'config': cfg,
                           'request': dict(request, geometry=geometry)})

    def _raw_packets(self, requests, *, mesh_accuracy=None, use_cache=True):
        """Persist each completed raw solve, independently of tracking references."""
        self._budget().check()
        if not hasattr(self, '_native_memory'): self._native_memory = OrderedDict()
        keys = [self._raw_key(r, mesh_accuracy) for r in requests]
        values, missing = {}, {}
        for key, request in zip(keys, requests):
            value = self._native_memory.get(key) if use_cache else None
            if value is None and self.cache and use_cache:
                value = self.cache.get(key, kind='native_raw_modes')
            if value is None: missing[key] = request
            else: values[key] = value
        batch_size = self.config.execution.native_batch_size
        pending = list(missing.items())
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start+batch_size]
            # Shrink a batch to the remaining solve budget, including possible retries.
            budget = self._budget()
            if budget.settings.max_solves is not None:
                available = (budget.settings.max_solves-budget.solves)//(1+self.config.solver.zero_mode_retries)
                if available < len(batch):
                    batch = batch[:max(1, available)]
            result = self._call_native('vlac_raw_batch', [[r for _, r in batch]], mesh_accuracy=mesh_accuracy)
            items = result['items']; items = [items] if isinstance(items, dict) else items
            for (key, _), value in zip(batch, items[:int(scalar(result['n']))]):
                if not bool(scalar(value['ok'])):
                    error = str(value.get('error', 'unknown'))
                    failure = InvalidModeError if error == 'zero_modes' else RuntimeError
                    raise failure('Native raw solve failed: '+error)
                values[key] = value
                if use_cache:
                    if self.cache: self.cache.put(key, value, kind='native_raw_modes')
                    self._native_memory[key] = value
                    while len(self._native_memory)>self.config.solver.raw_cache_memory_entries:
                        self._native_memory.popitem(last=False)
            if len(batch) < len(pending[start:start+batch_size]):
                from .runtime import BudgetExceeded
                raise BudgetExceeded('Remaining solve budget cannot cover the next raw batch including retries.')
        return [values[key] for key in keys]

    def _mode_candidates(self, geometry, wavelength_m, *, include_upper, include_lower,
                         mesh_accuracy=None, mesh_cells_override=None):
        if mesh_cells_override is not None:
            raise ValueError("Use solve_isolated_mode for native compact domains")
        value = self._raw_packets([self._raw_request(geometry, wavelength_m, include_upper, include_lower)],
                                  mesh_accuracy=mesh_accuracy, use_cache=False)[0]
        return [snapshot_from_native(v) for v in value['items'][:int(scalar(value['n']))]]

    def solve_isolated_mode(self, geometry, wavelength_m, *, guide, reference=None, compact_x=None,
                            mesh_accuracy=None, use_cache=True):
        if guide not in {"upper", "lower"}:
            raise ValueError("guide must be upper or lower")
        compact = self.config.solver.endpoint_compact_x if compact_x is None else bool(compact_x)
        refs = () if reference is None else (ModeReference(reference, reference, (reference, reference)),)
        key = stable_key({"isolated_native": self.script_hash, "provenance": optical_identity(self.config), "config": self.native_config(mesh_accuracy),
                          "geometry": asdict(geometry), "wavelength": wavelength_m, "guide": guide,
                          "compact": compact, "reference": reference.fingerprint() if reference else None})
        if self.cache and use_cache:
            cached = self.cache.get(key, kind="native_isolated")
            if cached is not None:
                return cached
        candidates = self._raw_packets([self._raw_request(geometry, wavelength_m, guide=='upper', guide=='lower', compact)],
                                       mesh_accuracy=mesh_accuracy, use_cache=use_cache)[0]
        value = self._call_native("vlac_select_isolated", [candidates, pack_references(refs), guide], mesh_accuracy=mesh_accuracy)
        result = snapshot_from_native(value["mode"]), float(scalar(value["overlap"]))
        if self.cache and use_cache:
            self.cache.put(key, result, kind="native_isolated")
        return result

    def evaluate_coefficients(self, geometry, wavelength_m, *, references=(), previous_kappa=None):
        key = self._cache_identity(geometry, wavelength_m, reference_fingerprints=(ref.fingerprint() for ref in references),
                                   previous_kappa=previous_kappa)
        if self.cache:
            cached = self.cache.get(key, kind="native_coefficients")
            if cached is not None:
                return cached
        raw = self._raw_packets([self._raw_request(geometry, wavelength_m, u, l) for u,l in ((1,0),(0,1),(1,1))])
        value = self._call_native("vlac_coefficients_from_raw", [float(wavelength_m), pack_references(references),
            np.array([[complex(previous_kappa or 0)]]), float(previous_kappa is not None), *raw])
        reference = value["reference"]
        result = (coefficients_from_native(value["coefficients"]),
                  ModeReference(snapshot_from_native(reference["upper"]), snapshot_from_native(reference["lower"]),
                                (snapshot_from_native(reference["plus"]), snapshot_from_native(reference["minus"]))))
        if self.cache:
            self.cache.put(key, result, kind="native_coefficients")
        return result

    def evaluate_profile_grid(self, config, bank, endpoint, w_control, d_control, z, wavelengths, *, state, progress=None):
        request = profile_request(endpoint, w_control, d_control, z, wavelengths, state)
        key = stable_key({"native_grid": self.script_hash, "provenance": optical_identity(self.config), "config": self.native_config(), "request": request})
        if self.cache:
            cached = self.cache.get(key, kind="native_profile_grid")
            if cached is not None:
                return cached
        from .profiles import _profile_coefficients
        result = _profile_coefficients(self, config, bank, endpoint, w_control, d_control, z, wavelengths,
                                       state=state, progress=progress)
        if self.cache:
            self.cache.put(key, result, kind="native_profile_grid")
        return result
