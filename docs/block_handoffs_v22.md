# Independent blocks and saved-result handoffs (v22)

v22 keeps the v21 optical model and numerical algorithms and adds reusable results at every block boundary. v23 adds explicit XY/search setup and zero-mode recovery/diagnostics; its saved block handoff format and intended physical model remain compatible. The default geometry remains upper width 0.3–2.0 um, bottom width 2.0 um, cross/bar liquid indices 1.55/1.37, and both residual slabs extending to the FDE x boundaries. v23 starts new raw/bare/coefficient cache namespaces for its explicit solver setup; no manual cache deletion is required.

## Start with the completed endpoint run

Install this revision from its project directory:

```powershell
python -m pip install -e .
```

Your successful v21 endpoint folder can be used directly; do not rerun the endpoint search merely to upgrade. Keep these files together:

- endpoint_summary.json
- endpoint_detunings.csv (the eight final cases)
- configuration.json
- endpoint_adaptive_verification.json, or endpoint_pso_verification.json for a PSO result

For example:

```powershell
vlac validate-block2 --config config/default.json --endpoint-dir outputs/endpoint --out outputs/block2
vlac validate-block3 --config config/default.json --profile-dir outputs/block2 --out outputs/block3
vlac validate-block4 --config config/default.json --bar-dir outputs/block3 --out outputs/block4
```

`--endpoint-dir` can point to an absolute directory in the old v21 project instead of copying it. Supply a config matching that run's geometry, materials, band and solver resolution. Endpoint widths and D are loaded from the successful result, never replaced with the width bounds or reoptimized by later blocks.

If validation returns `review`, inspect its report before considering the result numerically validated. Review exports remain usable for diagnosis or continuation; downstream reports retain the upstream validation status.

## Commands and scope

v25 adds an optional [Block 0 calibration](methods/block0-convergence.md). Supply the same passed certificate through `--block0-dir` to Blocks 1–3, or use its generated calibrated configuration. The certificate identity enters saved-result compatibility checks. Calibrated endpoints and final profiles receive fresh reference checks before publication. Existing uncalibrated handoffs retain their previous scope.

| Block | Run and export command | Validate and export command | Required input | MODE used |
|---|---|---|---|---|
| 0: MODE calibration | block0 | block0 / verify-block0 | configuration / saved design | Yes |
| 1: endpoint search | block1 (alias of endpoint) | validate-block1 | configuration | Yes |
| 2: cross-state profile optimization | block2 | validate-block2 | --endpoint-dir | Yes |
| 3: bar-state coefficient extraction | block3 | validate-block3 | --profile-dir | Yes |
| 4: two-mode propagation | block4 | validate-block4 | --bar-dir | No |

Use `--out` as a **directory** for every new block/validate-block command. Existing `validate-endpoint` and `validate-pso` retain their report-file `--out` convention. Existing `endpoint` continues to default to outputs/endpoint. New commands default to outputs/blockN or outputs/blockN_validation. Each new validation command exports all the same usable results as its run counterpart plus blockN_validation_report.json.

A complete explicit sequence is:

```powershell
vlac block1 --config config/default.json --out outputs/endpoint
vlac block2 --config config/default.json --endpoint-dir outputs/endpoint --out outputs/block2
vlac block3 --config config/default.json --profile-dir outputs/block2 --out outputs/block3
vlac block4 --config config/default.json --bar-dir outputs/block3 --out outputs/block4
```

Skip Block 1 when you already have its successful results. A source can also be the corresponding blockN_result.json path. A Block 3 folder carries the endpoint, fixed profile and cross coefficients as well as bar coefficients, so Block 4 does not need the earlier folders or an active MODE session. Custom material CSV files must still be available through the supplied config; they are checked by content hash and are not embedded.

## What validation means

- **Block 1:** `validate-block1` reuses the existing adaptive smoke test: five initial widths, a 13-width budget and ten fresh verification mode solves. Feasible, freshly verified endpoints can pass without global/search convergence. The report records the search's convergence separately.
- **Block 2:** run the configured profile optimizer at the two wavelength-band edges with fixed endpoint widths, fixed endpoint offsets and v21 targets -D*cos(theta) and D*sin(theta)/2. Validate controls, coefficient arithmetic, finite core coefficients, tracking/extraction flags and the achieved objective. Longitudinal midpoint checks assess z interpolation at the edges; interior wavelengths are not checked. A pass requires optimizer success, z-grid convergence and acceptable stored coefficient quality. No arbitrary target-fit objective threshold is imposed. A 1% grid tolerance is not a 1% target-fit requirement.
- **Block 3:** extract bar coefficients on the saved Block 2 grid for the same fixed geometry. Perform independent z and wavelength midpoint checks for the bar state, along with arithmetic, finite-value and tracking/extraction diagnostics. No endpoint search, profile optimization or bar target fitting occurs. If the bar grid needs review, inspect the errors and use a sufficiently dense saved profile/coefficient grid before relying on propagation. The denser diagnostic bar samples do not replace the saved common grid automatically.
- **Block 4:** load both coefficient grids and propagate both launch layers in both states at all saved wavelengths. New Block 2 runs save only the two band edges, so Block 4 reports those edges only. Check finite output powers, RK4 step convergence and significant total guided power above one. A pass refers to this integrator; inherited Block 2/3 review flags remain visible. It does not certify interior-wavelength behavior, mesh/domain convergence, radiation loss, full-device IL/XT or physical global optimality.

Blocks 2–4 use the configured production settings; validation is not automatically a cheap smoke test. An explicit small Block 2 test is:

```powershell
vlac validate-block2 --config config/default.json --endpoint-dir outputs/endpoint --out outputs/block2_smoke --max-evaluations 20 --max-refinements 0
```

An evaluation-limited result commonly reports `review` even when execution works. Increase the budget or grid limits as indicated by the optimizer/grid reports. Block 4 exposes `--rk4-steps-initial`, `--rk4-steps-max` and `--power-tolerance` so propagation can be reconverged without rerunning any FDE calculation.

The `validate-fde` command remains the separate licensed mesh/domain and extraction check. New profile/bar midpoint checks reuse matching raw cache entries and directly solve newly needed physical samples; they do not claim all points were recomputed with cache bypass. Block 1's ten final adaptive/PSO verification solves still bypass cache.

### Return codes

| Command outcome | Return code | Saved result |
|---|---:|---|
| validate-blockN passed | 0 | Result, summary and validation report |
| validate-block2/3/4 needs review | 3 | Usable result, summary and validation report |
| Invalid input or execution failure | 2 | Failure diagnostics; no new committed result for the failed block |
| blockN or run completes computation | 0 | Results may still have convergence/quality review flags |

Hard failure and review are separate. Regular run/block commands preserve the previous behavior of exporting usable budget-limited results. Read their summary flags before treating a design as validated. In PowerShell, `$LASTEXITCODE` reports the native command's return code.

## Continue the complete workflow from a checkpoint

```powershell
# Load Block 1; run Blocks 2, 3 and 4:
vlac run --config config/default.json --endpoint-dir outputs/endpoint --out outputs/full_run

# Load Block 2; run Blocks 3 and 4:
vlac run --config config/default.json --profile-dir outputs/block2 --out outputs/full_run

# Load Block 3; run Block 4:
vlac run --config config/default.json --bar-dir outputs/block3 --out outputs/full_run
```

These input flags are mutually exclusive. Without one, `vlac run` runs Blocks 1–4. Every completed block is exported before the next starts. If Block 3 fails, the saved Block 2 result can be resumed without repeating Blocks 1 or 2. Resuming is at block boundaries, not inside a Powell iteration or PSO generation.

`block4` and `validate-block4` never launch MODE. `run --bar-dir` likewise performs base propagation without MODE, but an explicitly configured optional SiN-index-sensitivity sweep still launches its separate FDE calculations after the base run.

## Files and compatibility

Each stage publishes blockN_result.json as an atomic, versioned JSON handoff with a checksum, saved configuration, physical/numerical signature, upstream identity, validation and the complete data needed downstream. JSON coefficients preserve real and imaginary components explicitly. No pickles or solver field snapshots are loaded from these files. Optional unavailable diagnostics and infinite/undefined IL/XT values use JSON null; readable CSV exports retain nan/inf where applicable.

- **Block 1:** existing endpoint exports plus block1_result.json and block1_summary.json.
- **Block 2:** block2_result.json, block2_summary.json, optimized_profiles.csv, cross_coefficients.csv, coefficients.csv (cross only), optimizer_history.json and profiles_and_coefficients.png. Nothing from Blocks 3/4 is computed.
- **Block 3:** block3_result.json, block3_summary.json, bar_coefficients.csv, combined coefficients.csv, inherited cross/profile exports and plots. Nothing is propagated.
- **Block 4:** block4_result.json, block4_summary.json, performance_by_launch.csv, performance_summary.csv and the existing complete configuration/metadata/plot exports.
- **All validation commands:** blockN_validation_report.json. Failures also write blockN_failure.json when the output/progress context has been initialized.

Loading checks the checksum, schema, completed status, eight endpoint cases, width/sign/balance restrictions, objective consistency, coordinate ordering, exact fixed endpoints, controls, target arrays, coefficient grid dimensions and coupling/detuning arithmetic. Geometry, physical material settings/content hashes, band edges, mode thresholds and solver resolution must match. From Block 2 onward, length and grid tolerance must match as well. Runtime API paths/GUI/thread settings, optimizer evaluation budgets and propagation integration settings can change where they do not alter the loaded upstream result.

Do not hand-edit result JSON; rerun the affected stage instead. A failed rerun clears that output folder's committed result and downstream published results while preserving completed earlier blocks. Use separate stage folders as shown above to retain previous exports independently.

Legacy v21 endpoint import additionally cross-checks the final CSV against the passed fresh-verification records and rejects finite-slab/pre-v21 geometry. Legacy exports lack material-file hashes, so an import assumes that any externally supplied material tables have not changed since that run. v22 handoffs record hashes. The default v21/v22 models need no external SiN extinction or liquid dispersion CSV.
