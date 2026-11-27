# Isolated lower-mode state-independence check

`check-lower-state` tests the named alternative `cross_reference_lower_mode_check_v1`:
reuse the cross-state isolated lower mode, including complex propagation constant
and transverse fields, in both states at the **same wavelength**. It leaves the
production method, optical caches and versioned block handoffs unchanged.

## Run within an hour

From the repository root:

```powershell
python tools/run_vlac.py check-lower-state --config config/native.json --stage screen --out diagnostics/lower_state_screen --max-solves 40 --max-seconds 3300
```

For coefficient and propagation comparisons on an existing Block 2 result:

```powershell
python tools/run_vlac.py check-lower-state --config outputs/profile/configuration.json --profile-dir outputs/profile --stage full --out diagnostics/lower_state_full --max-solves 120 --max-seconds 3300 --plan-only
python tools/run_vlac.py check-lower-state --config outputs/profile/configuration.json --profile-dir outputs/profile --stage full --out diagnostics/lower_state_full --max-solves 120 --max-seconds 3300
```

Replace the example profile paths with a completed compatible Block 2 run.
`--plan-only` validates inputs and prints samples and nominal call counts without
opening MODE or writing outputs. The supplied configuration must match the saved
handoff. A screen without a saved profile requires fixed gap and slab thickness.

The smaller workload uses **three wavelengths** (edges and center) for the lower
screen, and **the two band edges** for the coefficient/propagation comparison,
matching the existing Block 2 wavelength policy. Its five z positions are the
endpoints, one representative alignment nearest the device midpoint (or the
midpoint if no alignment exists), and the two intervening independent midpoints.
The screen uses 20 nominal FDE calls; the full comparison uses **60 total**.
All alignment roots are listed in the plan, but other roots are not certified.
This samples the frozen geometry; it does not validate the continuous band,
additional alignments or all optimizer-admissible geometries.

The CLI runs a dedicated worker with a **3300 second hard cutoff**, including
startup, leaving cleanup time within an hour. `--max-seconds` can shorten this but
cannot exceed 3300. On timeout, only the worker and its descendant processes are
terminated; existing user MODE sessions are not targeted. Completion before the
cutoff depends on actual solve/license times. A timeout is **incomplete**, never
passed. The direct `run_check` Python API instead checks time between solver calls
and propagation cases; use the CLI to interrupt blocking native calls.

Budgets are per invocation and reserve configured retries before each solve.
`--resume` reuses checksummed completed field packets in the same output directory
with the same inputs. Source handoff, configuration, code and MODE installation
identity are checked. Another invocation has a separate time budget. Existing
non-check directories are refused, protecting production results.

## Numerical comparisons

The lower-only problem disables both upper SiN objects while retaining the actual
oxide/liquid background. It uses the full production domain, not Block 1's compact
domain or its state-independent lower modes. Both states are evaluated at:

1. Production settings.
2. 1.5x finer x/y resolution, including certified local mesh overrides if present.
3. 1.5x larger domain margins at preserved spatial resolution.

All fields use one physical comparison grid. The worst observed phase/attenuation
wavelength receives a fresh pair of repeat solves. Selection uses polarization,
confinement and field-overlap tracking rather than mode numbers. Ambiguous matches
and non-finite data prevent acceptance.

In the full check, fresh upper and coupled candidates are solved for every state,
wavelength and z. Each state's own lower mode gives the reference. The alternative
substitutes the cross lower mode while retaining actual coupled eigenvalues,
upper-minus-lower detuning and the complex square-root branch convention. A
separate selection comparison detects coupled-pair changes caused by the shared
field. Both coefficient sets receive held-out midpoint checks against
PCHIP from the three base points. Both launches use the existing converged RK4
propagator. Cross-state objective values are recorded. No optimizer is rerun.

## Provisional acceptance limits

These are proposed engineering limits, not measured device tolerances:

| Quantity | Limit |
| --- | --- |
| Squared normalized transverse E-field similarity | At least 0.9999 |
| Lower-mode accumulated phase error over configured device length | At most 0.01 rad |
| Lower-mode attenuation difference over that length | At most 0.01 dB |
| Mesh/domain/repeat errors | At most one third of each preceding error budget |
| Complex detuning and coupling substitution errors | At most 0.001 of D and D/2 respectively |
| Longitudinal interpolation error for both methods | At most 0.01 of the corresponding target scale |
| Insertion-loss change | At most 0.01 dB |
| Crosstalk change above the 1e-6 output-power threshold | At most 0.1 dB |
| Absolute change in either output-port power | At most 1e-6 of launched power |

E-field similarity is not an E/H power-flux integral. Attenuation retains complex
index and the project's exp(+i beta z) convention. The absolute-power gate always
applies; the dB crosstalk gate applies only when both desired and undesired powers
in both calculations exceed 1e-6. Invalid baseline extraction, changed coupled
pair, insufficient midpoint resolution and unconverged integration prevent a pass.

## Evidence and status

- `plan.json`: scope, numerical profiles, source identity, limits and budgets.
- `samples/*.npz`: all candidate Ex/Ey fields and SI comparison-grid coordinates.
- Adjacent sample JSON: geometry, modes, complex indices, settings, actual mesh,
  solve attempts and field-file checksum.
- `report.json`: each mode/numerical comparison, baseline and substituted complex
  coefficients, objectives, both-launch propagation, timing and FDE counts.

| Status | Meaning | CLI exit |
| --- | --- | --- |
| `screen_passed` | Lower-only screen passed; downstream substitution untested | 0 |
| `passed` | All full-check sampled gates passed | 0 |
| `failed` | Acceptance gate or execution failed | 2 |
| `inconclusive` | Numerical uncertainty, invalid baseline or inadequate resolution | 3 |
| `incomplete` | Solve/time budget exhausted | 3 |

Only a nonsynthetic full pass sets `eligible_for_shared_lower_mode`. This check
does not issue a Block 0 certificate, validate a new optimum, or establish physical
device performance. A later production implementation should retain independent
state-specific verification of optimized candidates.

With raw caching, sharing between states reduces six distinct lower problems to
three for this wavelength set. It does not automatically save one third of total
runtime. Use measured calls and timing to assess the benefit.
