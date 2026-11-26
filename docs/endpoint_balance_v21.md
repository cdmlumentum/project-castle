# v21: increasing, balanced endpoints and full-width residual slabs

> v22 retains this optical model and adds independent block commands and saved-result loading. See [block_handoffs_v22.md](block_handoffs_v22.md).

## Geometry and materials

Upper-guide width bounds: 0.3–2.0 µm; bottom-guide width: 2.0 µm. Cross/bar liquid indices: 1.55/1.37. Heights, vertical gap, offsets and substrate-free stack remain as in v20.

Both 175 nm residual slab layers span the complete FDE x domain and are centered at x=0. The upper ridge moves laterally; its slab does not translate. The actual slab spans follow compact/full/enlarged domains. Isolated-mode solves omit the other guide's ridge and slab, retaining the existing bare-guide convention. The two-lower-mode cross-state reuse assumption remains explicit in adaptive and PSO searches.

Default x spans stay 10 µm compact and 26 µm coupled; y span stays 8.2 µm. Mesh cells stay 203×208 and 528×208. Lateral margin is now 4 µm. Old finite-overhang fields are removed from saved config and folded into the lateral margin on loading older config, preserving that config's domain width. Cache schemas are versioned for the changed slab geometry; no manual cache deletion is needed.

Old endpoint results, including the earlier interpolation screen, do not establish feasibility for the full-width-slab geometry. Run licensed MODE validation again.

## Constraints and objective

Detuning is Re(beta_upper-beta_lower). At each band edge, require cross input <0 and cross output >0, plus strict w_input<w_output. Let A and B denote the two positive cross magnitudes at a wavelength. Require 2|A-B|/(A+B)<=epsilon at each band edge. Default epsilon=0.10, configured by endpoint_search.cross_magnitude_balance_rtol. No averaging of mismatches across wavelengths or automatic relaxation is allowed.

Among feasible pairs maximize D=min |Re(delta_beta)| over both endpoints, cross/bar and both band edges. Bar signs remain unrestricted. This is the previous minimax magnitude objective with additional hard constraints; balance does not mathematically guarantee a larger D.

## Sampling and selection

The adaptive_1d CLI/backend name is retained for compatibility: costly evaluations still form a 1D shared width table, while selection is now jointly constrained in the two endpoint widths. Independent endpoint maximization no longer solves this problem.

Start with nine widths (five for validate-endpoint). Refine sign/validity and low-overlap transitions with part of the same budget. Compare all directly evaluated, eligible input/output pairs without MODE calls. Build shape-preserving PCHIP interpolation only in contiguous valid sign-consistent tracked branches. Rank predicted pairs by feasible objective, or by mismatch when none is predicted feasible. Invert interpolated cross detunings to propose same-magnitude and tolerance-boundary widths; test interval interiors as well. These matching proposals help discover feasible strips missed by a regular width grid. Solve new widths directly and retain all actual feasible pairs. Further subdivision explores unresolved intervals as needed.

Budgets remain 33 distinct widths (13 for validation). Local convergence requires the predicted best to agree with the direct best within objective and width tolerances on successive checks, plus nearby width samples around both winners. This is interpolation-guided local stability, not global certification. Hitting the budget can still return a verified feasible pair with converged=false. If no pair is found, report that sampled outcome without claiming the interval infeasible.

Final adaptive/PSO verification bypasses memory and disk cache: two lower plus eight upper mode solves. Recheck width order, signs, magnitude balance and signed-detuning agreement (rtol=1e-6, atol=1e-3 rad/m). Publish the best JSON only after success. Optional Python PSO and grid use the same pair constraints; PSO uses increasing-width seeds and can fail its three-particle smoke test if none is feasible. Grid uses its existing directly solved case records rather than the adaptive/PSO ten-solve verification stage.

Nominal cold-cache adaptive workload: one startup solve + two lower modes + up to four upper solves per distinct width + ten verification solves. Maxima stay 145 (endpoint) and 65 (validate-endpoint); early invalid/mixed-sign cases and cache hits reduce actual counts. Pair comparison/interpolation adds Python work, not FDE calls. No runtime improvement is claimed without measurement.

## Block 2 and outputs

Set detuning target to -D*cos(theta), so endpoints are -D and +D. Coupling remains D*sin(theta)/2. Keep endpoint widths fixed in the spline profile and evaluate both launch layers during propagation. Reversing endpoint signs does not reverse the definition of delta_beta or change the propagation equations.

The constraints match cross magnitudes at each wavelength; they do not require a common magnitude across wavelengths or across liquid states. Reports include each cross endpoint's normalized residual from the common Block 2 target. Bar-state bottlenecks and dispersion may still prevent an exact target fit.

- outputs/endpoint/endpoint_adaptive_best.json: verified selected widths, objective and constraint report.
- outputs/endpoint/endpoint_summary.json: overall endpoint result.
- endpoint_width_samples.csv / endpoint_width_detunings.csv: width table and signed cases.
- endpoint_pairs.csv: actual pair feasibility, mismatch and objective.
- endpoint_adaptive_history.json / endpoint_adaptive_summary.json: proposals, budget and local convergence.
- endpoint_adaptive_verification.json: ten fresh mode results, constraints and numerical agreement.
- outputs/run/endpoint_validation_report.json: default validate-endpoint report.

Commands:

```powershell
python -m pip install -e .
vlac validate --config config/default.json
vlac validate-endpoint --config config/default.json
vlac endpoint --config config/default.json
vlac run --config config/default.json --out outputs/full_run
```

In v21, `run` repeated endpoint search. v22 can load it with `run --endpoint-dir ...`, or load later stages with `--profile-dir` / `--bar-dir`; see the v22 guide. Full-device EME/varFDTD/FDTD is not automated.
