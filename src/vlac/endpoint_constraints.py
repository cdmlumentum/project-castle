"""Endpoint conditions shared by Python PSO, grid search and verification."""

from __future__ import annotations

import math


CROSS_STATE_SIGN_REQUIREMENT = {
    "state": "cross",
    "delta_beta_convention": "beta_upper - beta_lower",
    "input_real_delta_beta": "< 0",
    "output_real_delta_beta": "> 0",
    "wavelength_policy": "both configured band edges",
}


class EndpointSignError(RuntimeError):
    """A fully evaluated width pair fails the cross-state sign restriction."""

    def __init__(self, records: list[dict], report: dict):
        self.records, self.report = records, report
        failures = [
            f"{row['end']} at {row['wavelength_m']*1e9:.6g} nm: "
            f"Re(delta_beta)={row['delta_beta_real_rad_per_m']} rad/m "
            f"(required {row['required_real_delta_beta']})"
            for row in report["checks"] if not row["passed"]
        ]
        super().__init__("Cross-state endpoint sign restriction failed: " + "; ".join(failures))


def cross_state_sign_report(records: list[dict]) -> dict:
    """Require two distinct, matching band edges at each endpoint, then test signs.

    Missing/duplicate cases are programming/data errors, not particle penalties.
    Zero detuning fails the strict inequalities. Bar-state signs are unrestricted.
    """
    selected = [row for row in records if row["state"] == "cross"]
    by_end = {end: [row for row in selected if row["end"] == end]
              for end in ("input", "output")}
    wavelengths = {end: sorted(row["wavelength_m"] for row in rows)
                   for end, rows in by_end.items()}
    if (len(selected) != 4 or any(len(rows) != 2 for rows in by_end.values())
            or len(set(wavelengths["input"])) != 2
            or wavelengths["input"] != wavelengths["output"]):
        raise ValueError("Endpoint sign check requires two matching, distinct cross-state band edges at each end.")
    checks = []
    for row in selected:
        value = float(row["delta_beta_real_rad_per_m"])
        finite = math.isfinite(value)
        passed = finite and (value < 0 if row["end"] == "input" else value > 0)
        checks.append({
            "end": row["end"], "wavelength_m": float(row["wavelength_m"]),
            "delta_beta_real_rad_per_m": value if finite else None,
            "required_real_delta_beta": "< 0" if row["end"] == "input" else "> 0",
            "passed": bool(passed),
        })
    return {"passed": all(row["passed"] for row in checks),
            "requirement": dict(CROSS_STATE_SIGN_REQUIREMENT), "checks": checks}


def require_cross_state_signs(records: list[dict]) -> dict:
    report = cross_state_sign_report(records)
    if not report["passed"]:
        raise EndpointSignError(records, report)
    return report


class EndpointConstraintError(RuntimeError):
    """Signed modes are available but width order or magnitude balance fails."""

    def __init__(self, records: list[dict], report: dict):
        self.records, self.report = records, report
        self.reason = "endpoint_width_order" if not report["width_order"]["passed"] else "cross_state_magnitude_balance"
        super().__init__(f"Endpoint pair constraint failed: {self.reason}; "
                         f"maximum relative cross magnitude mismatch={report['magnitude_balance']['maximum_relative_mismatch']}; "
                         f"limit={report['magnitude_balance']['relative_tolerance']}")


def endpoint_constraint_report(records: list[dict], settings) -> dict:
    """Assess increasing width and same-wavelength cross magnitude balance.

    Detunings remain beta_upper-beta_lower. Missing/duplicate records or
    inconsistent endpoint widths are data errors rather than rejected designs.
    """
    signs = cross_state_sign_report(records)
    widths = {}
    cross = {}
    for end in ("input", "output"):
        values = {float(r["w1_m"]) for r in records if r["end"] == end}
        if len(values) != 1:
            raise ValueError("Endpoint constraint check requires one width per endpoint.")
        widths[end] = values.pop()
        cross[end] = {float(r["wavelength_m"]): float(r["delta_beta_real_rad_per_m"])
                      for r in records if r["end"] == end and r["state"] == "cross"}
    order = {"passed": all(math.isfinite(v) for v in widths.values()) and widths["input"] < widths["output"],
             "requirement": "w1_input_m < w1_output_m", "w1_input_m": widths["input"], "w1_output_m": widths["output"]}
    checks = []
    for wavelength, signed_in in sorted(cross["input"].items()):
        signed_out = cross["output"][wavelength]
        a, b = abs(signed_in), abs(signed_out)
        finite = math.isfinite(a) and math.isfinite(b) and a+b > 0
        mismatch = 2*abs(a-b)/(a+b) if finite else None
        checks.append({"wavelength_m": wavelength,
                       "input_real_delta_beta_rad_per_m": signed_in if math.isfinite(signed_in) else None,
                       "output_real_delta_beta_rad_per_m": signed_out if math.isfinite(signed_out) else None,
                       "input_magnitude_rad_per_m": a if math.isfinite(a) else None,
                       "output_magnitude_rad_per_m": b if math.isfinite(b) else None,
                       "relative_mismatch": mismatch,
                       "passed": finite and mismatch <= settings.cross_magnitude_balance_rtol})
    worst = max(c["relative_mismatch"] for c in checks) if all(c["relative_mismatch"] is not None for c in checks) else None
    balance = {"passed": all(c["passed"] for c in checks),
               "definition": "2*abs(abs(delta_in)-abs(delta_out))/(abs(delta_in)+abs(delta_out)) at each band edge",
               "relative_tolerance": settings.cross_magnitude_balance_rtol,
               "maximum_relative_mismatch": worst, "checks": checks}
    score = min(abs(float(r["delta_beta_real_rad_per_m"])) for r in records)
    target_checks = []
    if math.isfinite(score) and score > 0:
        for end in ("input", "output"):
            target = -score if end == "input" else score
            target_checks.extend({"end": end, "wavelength_m": w, "target_rad_per_m": target,
                                  "relative_target_residual": abs(v-target)/score}
                                 for w,v in sorted(cross[end].items()))
    return {"passed": signs["passed"] and order["passed"] and balance["passed"],
            "cross_state_sign_verification": signs, "width_order": order,
            "magnitude_balance": balance,
            "block2_common_target_checks": target_checks}


def require_endpoint_constraints(records: list[dict], settings) -> dict:
    require_cross_state_signs(records)
    report = endpoint_constraint_report(records, settings)
    if not report["passed"]:
        raise EndpointConstraintError(records, report)
    return report
