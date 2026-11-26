# Block 0: MODE domain and mesh calibration

Block 0 selects the fastest **tested** transverse domain and mesh that meet explicit numerical error limits. It exports two fixed solver profiles and a certificate consumed by Blocks 1–3. It changes computational padding and resolution; physical ridge widths, slabs, gap, BOX constraints and materials remain device inputs.

`src/vlac/block0.py` orchestrates the bounded search, checkpoints and reports. `src/vlac/native/vlac_convergence.lsf` calls the native MODE core for every optical sample. An isolated sample uses two FDE solves; a coupled sample uses three, plus configured zero-mode retries. No synthetic optical model is used in a licensed run.

## Validation status

This revision has offline numerical and contract tests. The user deferred the licensed Block 0 trial until further instruction. No new domain, mesh, production speedup or real-device convergence is established. Earlier v24 reports are historical implementation-parity evidence for that revision.

## Algorithm

1. Record input/source/MODE installation identities, case counts, budgets, hardware and first-sample timing.
2. Expand lateral, top and bottom reference padding independently while preserving physical dx/dy. Require two consecutive passing comparisons for each axis.
3. Refine horizontal and vertical meshes independently at fixed domain size. Require two consecutive passing comparisons for each, followed by combined mesh/domain and extra-PML-layer guards.
4. Screen lateral/top/bottom padding and horizontal/vertical spacing independently; test combined reductions and coarse backgrounds with local ridge/interface mesh overrides.
5. Cover minimum/middle/maximum widths, both liquid states, band edges plus center, and minimum/zero/maximum offsets, including rib/gap bounds if variable. Add interpolated phase-matching widths and midpoints where modal fields or TE fractions change substantially.
6. Require each candidate to pass the complete expanded envelope, including a further reference refinement/PML guard on every added case. A failed reference prevents certification. Failed candidates are rejected; a conservative passing reference remains a possible fallback.
7. Rank candidates by measured sample runtime. Run fresh timing replicates for the selected profile and original baseline. Report a slowdown when one is measured.
8. Publish profiles only for a completed, real, full calibration. Trial, exhausted-budget and synthetic runs cannot certify production settings.

This is a bounded coordinate search with joint checks, not proof of a global minimum. If an added case invalidates the reference, revise the reference plan in a new run directory; geometry bounds and tolerances are never silently relaxed.

## Accuracy limits

All applicable limits must pass against the reference.

| Quantity | Default |
| --- | --- |
| Absolute real effective-index change | `2e-5` |
| Normalized transverse E-field overlap squared | at least `0.999` |
| TE-like fraction change | at most `0.01`; both modes must also meet the existing minimum TE fraction |
| Complex detuning/coupling error | at most `2 rad/m + 0.01 * abs(reference)` |
| Detuning/coupling error times device length | at most `0.02 rad` |
| Modal attenuation change over device length | at most `0.01 dB` |
| Overlap change on a nested every-other-point grid | at most `(1 - minimum_power_overlap) / 5` |

The overlap is a transverse E-field similarity measure, not a full E/H power-flux integral. A dense fixed physical comparison grid is retained for each solve family, including certified production calculations. Fields outside their solver grid are zero, so cropping is visible. A pair-subspace singular-value comparison handles rotations of nearly degenerate modes; propagation constants and coupling still need to pass. Complex beta, upper-minus-lower detuning and the existing coupling branch convention are preserved.

The absolute coefficient floor avoids division by zero. At 1.5 mm length, the phase limit corresponds to about `13.3 rad/m`, potentially stricter than the 1% coefficient criterion. These numerical limits can be reviewed when device performance targets are defined. They do not guarantee insertion loss or crosstalk.

## Profiles and production handoffs

| Profile | Use |
| --- | --- |
| `isolated` | Guide-centered Block 1 modes, preserving its configured state-independent lower-guide assumption |
| `coupled` | Both isolated-guide and dual-guide solves used for coefficient extraction in Blocks 2–3 |

Coefficient extraction retains its full, self-consistent domain. Domains and meshes remain fixed during optimization. Slabs continue through the lateral boundaries/PML; local mesh boxes do not create physical slab edges.

The certificate contains selected/reference profiles, accuracy limits, sampled-case assessments, source identity and installed API/executable hashes. Changed geometry, material-table contents, wavelengths, length, numerical baseline/policy, endpoint assumptions, implementation or MODE installation invalidate it. Output directories, GUI visibility and invocation budgets may change. Timing checkpoints also require the original host/thread settings.

Supply `--block0-dir` or use the generated calibrated configuration. Configurations without a certificate remain usable without claiming Block 0 convergence. Block/cache identities include the calibration, preventing mixed calibrated/uncalibrated handoffs.

Before publishing calibrated endpoints or a final profile/bar result, the package runs bounded fresh reference comparisons on the selected geometry. Profiles are sampled at saved z positions and midpoints, at saved wavelengths and band center, in both states. Stored endpoint/cross/bar coefficients are also checked, catching branch-selection errors in the actual result. A failed or incomplete gate prevents publication. Block 2 still optimizes at its two band edges and retains its saved wavelength grid.

Standalone bundles embed the profiles and certificate ID in MAT data and their manifest. The LSF scripts use these settings; importing/publishing standard Python artifacts performs the selected-design gates. Direct manual LSF execution does not execute Python validation gates.

## Commands

These licensed commands were **not run during this implementation**.

```powershell
# Full calibration with explicit invocation budgets:
python tools/run_vlac.py block0 --config config/native.json --out outputs/block0/run01 --max-solves 1200 --max-seconds 21600 --hide-gui

# Diagnostic trial, when authorized:
python tools/run_vlac.py block0 --config config/native.json --out outputs/block0/trial01 --trial --max-solves 36 --max-seconds 900 --hide-gui

# Resume identical inputs with a fresh invocation budget:
python tools/run_vlac.py block0 --config config/native.json --out outputs/block0/run01 --resume --max-solves 1200 --max-seconds 21600 --hide-gui
```

A trial compares baseline, finer mesh and expanded domains in both states at central geometry/wavelength. It always reports `incomplete` for production certification. `execution_completed` distinguishes a completed diagnostic from an interrupted one.

Resume checks identities and compressed field-array checksums. Partially written samples are recalculated. Solve budgets reserve all configured retries; time limits are checked between samples and an active eigensolve can overrun the wall-time budget. Accuracy/search-plan changes require a new output directory.

```powershell
python tools/run_vlac.py block1 --config config/native.json --block0-dir outputs/block0/run01 --out outputs/calibrated/block1
python tools/run_vlac.py block2 --config config/native.json --block0-dir outputs/block0/run01 --endpoint-dir outputs/calibrated/block1 --out outputs/calibrated/block2
python tools/run_vlac.py block3 --config config/native.json --block0-dir outputs/block0/run01 --profile-dir outputs/calibrated/block2 --out outputs/calibrated/block3
python tools/run_vlac.py native prepare --config config/native.json --block0-dir outputs/block0/run01 --out outputs/calibrated/native_bundle
```

Alternatively use `calibrated_configuration.json` directly as `--config`; it includes absolute material-table paths for this machine. `verify-block0` explicitly checks a saved design using `--config`, `--block0-dir`, `--out`, and either `--endpoint-dir` or `--profile-dir`, with optional solve/time budgets.

## Artifacts

Every attempted run writes `plan.json`, `checkpoint.json`, `certificate.json`, `test_cases.json`, JSON/CSV results and `report.html`. A completed first sample adds `pilot.json`. Full runs retain their expanded per-family case sets. `samples/` contains numeric NPZ arrays and checksummed JSON metadata; no pickle is loaded.

Only passed full runs additionally publish `solver_profiles.json`, `solver_profiles.mat` and `calibrated_configuration.json`. Selected-design checks write `design_verification.json` below the block output's `block0_checks/` directory.

Exit codes: **0** passed full calibration/design verification; **3** trial scope or budget exhaustion; **2** failed accuracy/reference checks, invalid inputs or execution failure. Defaults and search factors are in the optional `convergence` config section, explicitly shown in `config/native.json`.

## Tests and sources

`python tools/run_tests.py test_block0` tests accuracy gates, trial labeling, budgets, checked resume, stale-certificate rejection and offline JSON/MAT/bundle handoffs with explicitly synthetic data. The full test suite checks previous block compatibility. Neither command launches MODE.

Primary documentation: Ansys [convergence guidance](https://optics.ansys.com/hc/en-us/articles/360051107554-Tunable-CROW-Filter), [FDE mesh and PML settings](https://optics.ansys.com/hc/en-us/articles/360034916973-FDE-solver-Simulation-object), [mesh overrides](https://optics.ansys.com/hc/en-us/articles/360034924253-addmesh-Script-command), and [mode-finding pitfalls](https://optics.ansys.com/hc/en-us/articles/360034396914-Tips-for-finding-modes-in-FDE-and-the-mode-source). Installed v221 runtime functionality awaits the deferred trial. Numerical convergence is distinct from material accuracy, fabrication robustness and full-device EME/FDTD validation.
