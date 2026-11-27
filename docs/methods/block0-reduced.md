# Reduced Block 0: mesh and lateral-domain convergence

`mode_mesh_lateral_sentinel_v1` is an explicitly named alternative to full Block 0 calibration. It tests one smaller candidate at four isolated and four coupled stress cases. Nine solver settings require **180 healthy FDE calls**; **200 is the hard lifetime limit**, including retries and conservative reservations for interrupted/failed samples, across every resume of that output directory.

This method reports convergence only at its recorded cases. It does not issue the full-envelope certificate, `solver_profiles.*`, or `calibrated_configuration.json`. A scoped pass cannot be supplied to `--block0-dir`. The full method remains the default. No real MODE convergence or speedup has been established by the offline tests for this alternative.

## Commands

Run from the repository root. Plan inspection uses no MODE license and creates no outputs:

```powershell
python tools/run_vlac.py block0 --method reduced --config config/native.json --out outputs/block0/reduced01 --plan-only
```

Run the licensed calculation only when intended:

```powershell
python tools/run_vlac.py block0 --method reduced --config config/native.json --out outputs/block0/reduced01 --max-solves 200 --max-seconds 7200 --hide-gui
python tools/run_vlac.py block0 --method reduced --config config/native.json --out outputs/block0/reduced01 --resume --max-solves 200 --max-seconds 7200 --hide-gui
```

The default reduced wall-time allowance is the smaller of the configured convergence allowance and 7,200 seconds; an explicit `--max-seconds` overrides it. It is a soft limit checked before a sample. A started sample may finish after the deadline. At a uniform 35 seconds per call, 180 calls take 105 minutes and 200 take 117 minutes before extra overhead. Reference settings can be slower, so the call cap is not a two-hour runtime guarantee.

`--max-solves` may lower the per-invocation allowance; it cannot raise the 200-call lifetime cap. On resume, completed, checksummed samples avoid FDE. A pending reservation after a crash stays charged because its actual consumption is unknown. After exhausting the lifetime allowance, the method reports incomplete; changing accuracy limits or creating a new run does not turn the prior result into convergence.

An OS file lock prevents concurrent invocations from sharing the budget journal.
It releases automatically when a process exits or crashes. Shared backend
execution-budget exhaustion also produces an incomplete result.

## Recorded stress cases

Each family contains exactly one case for each cross/bar state and minimum/maximum wavelength pair. All dimensions are SI. Gap and rib height must be fixed. Both solve families use the same physical device bounds, materials, mode eligibility, complex extraction convention and error thresholds as the full method.

The default isolated cases combine minimum/maximum widths with both states and band edges, centered for the compact solve. The coupled cases include the maximum positive offset at minimum width, a zero-offset midpoint width in the cross state, a zero-offset minimum width in the bar state, and maximum width at the negative offset limit. **These are provisional coverage cases. The midpoint is not an established phase-matching width, and a bound may not support an eligible mode.** A missing/ineligible mode remains a failure; the method does not silently substitute another width.

Where existing endpoint or modal evidence is available, choose verified weakly confined and near-phase-matched stress cases using `--cases path/to/cases.json`. This replaces the cases without adding a hidden scan. The JSON object must contain `isolated` and `coupled` lists of exactly four cases each. Every case supplies `w1_m`, `d_c2c_m`, `wavelength_m`, and `state`; optional `h_gap_m` and `h_rib_m` must match their fixed bounds. For example, an individual case has the form:

```json
{"w1_m": 1.15e-6, "d_c2c_m": 0.0, "wavelength_m": 1.26e-6, "state": "cross"}
```

This example declares coordinates only; it makes no claim about the actual mode or phase matching there. Use `--plan-only` to see the complete default case JSON and solver profiles. The exact case selection and provenance are retained in the plan and identity; changed cases prevent resume.

## Nine settings and required checks

Let `hx0` and `hy0` denote physical mesh spacings derived from the original full-domain span and cell counts. Lateral padding is added outside the complete configured ridge-position envelope, not measured from its center. The vertical domain remains fixed. The two residual slabs continue through each lateral domain and PML. Changing the span also changes the cell count, preserving the requested physical spacing.

| Setting | Lateral padding per side | hx/hx0 | hy/hy0 | PML layers (default) |
| --- | --- | --- | --- | --- |
| A: candidate | 2 um | 1.25 | 1.25 | 12 |
| B | 2 um | 1.00 | 1.25 | 12 |
| C | 2 um | 0.80 | 1.25 | 12 |
| D | 2 um | 0.80 | 1.00 | 12 |
| E | 2 um | 0.80 | 0.80 | 12 |
| F | 3 um | 0.80 | 0.80 | 12 |
| G | 4 um | 0.80 | 0.80 | 12 |
| H: joint guard | 5 um | 0.64 | 0.64 | 12 |
| I: PML guard | 5 um | 0.64 | 0.64 | 16 |

F/G padding is rounded outward to an equal-parity cell count on E's actual
uniform lattice. Their actual x spacing and x=0 mesh registration therefore
remain fixed during the lateral-domain checks. The plan records the resulting
spans and all profiles share a comparison grid covering the largest setting.

The PML settings are actually `convergence.pml_layers` and that value plus `pml_test_extra_layers`. With the shipped geometry, A has a 22 um coupled span and 6 um compact span. Nominal grids are about 358 x 167 and 98 x 167 cells, respectively. These are proposed reductions, not validated production settings.

Every case must pass A-B and B-C for x mesh, C-D and D-E for y mesh, E-F and F-G for lateral span, G-H for the joint guard, H-I for PML, **and A-I directly** to bound accumulated error. All comparisons use one common physical field grid per family. Samples are saved once and reused for comparisons; tracking/complex coupling comparison follows the existing Block 0 code.

The selected `neff_kappa_only_v1` accuracy policy uses `neff_atol=1e-4` and keeps
complex coupling absolute-plus-relative and phase-error limits, modal attenuation,
TE fraction, field/pair-subspace overlap, and comparison-grid aliasing checks.
Detuning errors are reported without an independent acceptance threshold. A failed
comparison produces failure, not an automatically relaxed tolerance or a claim
that a different candidate converged.

The reduction omits vertical-domain convergence, the full geometry and interior-wavelength envelope, an adaptive phase-matching scan, a search for the fastest settings, and independent timing replicates. It establishes neither longitudinal interpolation convergence nor physical device IL/XT. The reserve is available for configured retries; no optional hidden optical work consumes it.

## Outputs and validation

The output contains `plan.json`, `test_cases.json`, `checkpoint.json`, `pilot.json` after the first sample, checksummed numeric NPZ/JSON samples, `convergence_results.json`, `convergence_results.csv`, `reduced_report.json`, and `report.html`. The report records the exact scope, all 72 required case comparisons, omitted checks, candidate/reference profiles, total/session call accounting and accuracy limits. `production_certified` is always false.

Exit codes are 0 for a real scoped pass, 3 for budget exhaustion or synthetic-only test execution, and 2 for failed checks or execution/input errors. Synthetic fixtures can exercise every comparison without claiming convergence.

```powershell
python tools/run_tests.py test_block0_reduced
python tools/run_tests.py
```

These tests do not launch MODE. Licensed checks and any later physical validation require their own recorded evidence. The implementation reuses the native MODE sample evaluator and the full Block 0 numerical comparison gates. Primary guidance: [Ansys mesh/domain convergence](https://optics.ansys.com/hc/en-us/articles/360051107554-Tunable-CROW-Filter) and [FDE mesh, domain and PML controls](https://optics.ansys.com/hc/en-us/articles/360034916973-FDE-solver-Simulation-object).
