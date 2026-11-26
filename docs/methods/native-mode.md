# Native MODE development and execution

The Python package supports a native Lumerical script calculation core and standalone MODE entry points for Blocks 1, 2 and 3. The calculation core preserves the v23 cross-section geometry, sampled material tables, transverse-field normalization, mode selection, detuning sign and complex coupling extraction. Implementation parity is checked separately from mesh convergence and physical device performance.

## Execution choices

For certified transverse domains and meshes, see [Block 0 convergence](block0-convergence.md). `native prepare --block0-dir ...` embeds its passed profiles and certificate ID. The new calibration/native profile settings have offline contract coverage; their licensed v221 trial is deferred pending user instruction.

| Execution choice | Endpoint search | Profile optimization | FDE and coefficient processing |
| --- | --- | --- | --- |
| Original package with `config/default.json` | v23 adaptive search or configured Python alternative | SciPy bounded Powell | Original Python adapter calling MODE |
| Package with `config/native.json` | Same v23 algorithms | Same SciPy bounded Powell | Native raw batches, selectors and extraction; Python traverses tracked grids |
| Standalone MODE bundle | Explicit sampled-width pair search | Explicit bounded coordinate pattern search | Native LSF |

The native standalone algorithms are alternatives to the adaptive search and Powell optimizer. A sampled-width result is the best feasible directly sampled pair; it does not claim search convergence or a global optimum. The coordinate pattern search terminates when its normalized step becomes sufficiently small or its evaluation budget is exhausted. A budget-limited result needs review.

The native package adapter writes schema 2 and reads older compatible handoffs. It caches raw candidates, tracked coefficients and complete matching grids. Python traverses z/wavelength samples in the established tracking order; missing raw solves are sent to MODE in bounded batches. This adds raw-field transfer/storage costs while avoiding repeated eigensolves across grids. See [the v26 execution guide](workflow-runtime.md) for budgets, checkpoints and retained characterization.

The initial local comparison took approximately 75 versus 94 seconds for the Python/native cross-state sample and 98 versus 108 seconds for the bar-state sample. These are single samples under the current machine load. Keep Powell in Python and profile the repeated solve/cache boundary before making a speed-driven migration decision.

## Run this checkout

Run these commands from the repository root. The launchers select the local source instead of any other installed `vlac` package.

```powershell
python tools/run_vlac.py block1 --config config/native.json --out outputs/native_endpoint
python tools/run_vlac.py block2 --config config/native.json --endpoint-dir outputs/native_endpoint --out outputs/native_profile
python tools/run_vlac.py block3 --config config/native.json --profile-dir outputs/native_profile --out outputs/native_bar
python tools/run_vlac.py block4 --config config/native.json --bar-dir outputs/native_bar --out outputs/native_propagation
```

These are production commands and can require many FDE solves. A completed compatible endpoint or profile directory can be supplied directly, including an existing v23 result. A Block 2 smoke execution can limit work using `--max-evaluations 2 --max-refinements 0`; its status will commonly be `review`.

`solver.engine` is `python` by default and `native` in the supplied native configuration. GUI visibility, API path and thread settings retain their original configuration behavior. The provided native configuration hides newly launched MODE sessions and uses a separate cache file.

## Prepare standalone MODE blocks

Preparing a bundle does not launch MODE or use a solver license:

```powershell
python tools/run_vlac.py native prepare --config config/native.json --out outputs/native_bundle
```

The bundle contains embedded material tables in `native_config.mat`, the native function libraries, and `block1.lsf`, `block2.lsf` and `block3.lsf`. Open the scripts in MODE and run them in order. Each script saves a native MAT handoff and, after successful execution, an LMS project. The entry scripts contain the bundle's absolute working directory; regenerate the bundle after moving it.

Block 1 uses `adaptive_max_widths` evenly spaced widths unless `--width-samples` is supplied at preparation. Bounds, endpoint signs and magnitude balance are enforced. The successful pair receives ten fresh isolated-mode solves for verification. Lack of a feasible sampled pair does not prove that the continuous interval is infeasible.

Block 1 retains those ten selected modes and their SI tracking-grid coordinates in `block1_native.mat`. Before saving `block1.lms`, its entry script performs one additional full-domain solve with both guides enabled at the selected input width/offset, cross liquid state and minimum wavelength. All entry scripts first save an incomplete MAT and empty LMS using safe-mode-compatible commands; a successful result replaces them. Published Python outputs are retained in immutable attempts, so a failed rerun does not replace a prior success. MODE `native run` and `native import` publish the two-panel endpoint beta plot (PNG/PDF), field NPZ/PNG, `endpoint_diagnostics.json`, and a copy of that LMS under `python/block1`. Running the LSF directly produces the MAT/LMS files; import generates the Python plots. These outputs are inspection evidence, separate from the scalar versioned handoff.

Block 2 holds endpoint widths and offsets fixed, evaluates the clamped cubic profiles at the two band edges, fits the cross-state Blackman targets with the minimax objective, and checks coefficient interpolation at independent z midpoints. Its optimizer is named `coordinate_pattern`. This follows the current project's band-edge-only Block 2 policy. Block 3 reuses that exact profile and common grid in the bar state, with independent z and wavelength midpoint checks, no profile optimization and no bar target fit.

The standalone optimizer currently supports the default unconstrained slope and curvature settings. Preparation rejects explicit slope or curvature limits; use the native package path with the existing Powell optimizer for those limits. Standalone endpoint search requires fixed gap and rib height and the existing state-independent lower-guide assumption. Missing modes, rejected mode eligibility, API failures and failed handoffs remain distinct outcomes.

To start from an existing block result:

```powershell
python tools/run_vlac.py native prepare --config config/native.json --endpoint-dir outputs/native_endpoint --out outputs/native_from_endpoint
python tools/run_vlac.py native prepare --config config/native.json --profile-dir outputs/native_profile --out outputs/native_from_profile
```

The first command seeds the native Block 1 handoff. The second seeds both Blocks 1 and 2, allowing direct execution of `block3.lsf`.

## Run or import through Python

The same standalone script can be executed through the thin package interface:

```powershell
python tools/run_vlac.py native run --bundle outputs/native_bundle --block 1
python tools/run_vlac.py native run --bundle outputs/native_bundle --block 2
python tools/run_vlac.py native run --bundle outputs/native_bundle --block 3
```

If the result was already produced by running the script in MODE, import it with:

```powershell
python tools/run_vlac.py native import --bundle outputs/native_bundle --block 3
```

Import uses a MODE session to read the native MAT data. It validates bundle identity, the original endpoint constraints, coefficient arithmetic and the profile/grid contract before publishing the standard JSON result under `python/blockN` in the bundle. That output can feed the existing propagation stage:

```powershell
python tools/run_vlac.py block4 --config config/native.json --bar-dir outputs/native_bundle/python/block3 --out outputs/native_propagation
```

The bundle manifest records its configuration and script checksums. Prepare a new bundle when those inputs change. A changed algorithm or implementation receives a new source identity. Native result conversion returns exit code 0 for passed checks, 3 for a usable result needing review, and 2 for failure.

## Validation

```powershell
python tools/run_tests.py
python tools/validate_native_mode.py --math-only --out diagnostics/native_algorithms
python tools/validate_native_mode.py --out diagnostics/native_mode
```

The first command runs synthetic and contract tests without MODE. The second uses MODE's interpreter to compare native B-splines, PCHIP interpolation and coupling arithmetic against SciPy/Python, and runs the standalone blocks with explicitly synthetic optical fixtures to exercise their handoffs. It does not perform FDE solves.

The final command additionally evaluates actual v23 geometry in both liquid states at an upper width of 1.15 micrometres, zero lateral offset and 1300 nm, using the configured production mesh. It compares the four selected propagation constants, detuning, complex coupling, extraction flags and normalized field overlaps. The report records both elapsed times and the complete configuration. Saved Python and native LMS projects are diagnostic outputs.

Implementation parity does not establish mesh/domain convergence, interpolation convergence over a production profile, optimizer convergence, full-device insertion loss or crosstalk. Those remain the responsibility of the existing per-block and full-device validation procedures.

The endpoint beta plot joins only the calculated endpoints with straight lines over the configured length. It does not evaluate interior widths or replace Block 2. Mode-profile archives contain complex, normalized transverse Ex/Ey on the same tracking grid used for selection; individual intensity panels are scaled to their own peaks. `endpoint_diagnostics.json` records the array mapping, coordinates, mode identities and saved-project geometry. The extra project solve leaves production mesh, domain, tracking and selection conventions unchanged.

## Sources and implementation files

- `src/vlac/native/vlac_core.lsf`: geometry, FDE, interpolation, tracking, extraction, B-splines and complete profile grids.
- `src/vlac/native/vlac_blocks.lsf`: standalone searches and midpoint convergence checks.
- `src/vlac/native_backend.py`: package adapter and source-aware native caches.
- `src/vlac/native_mode.py`: offline bundle preparation, execution and validated handoff conversion.
- `docs/legacy/v23-import.json`: hashes of the original v23 source files.

In Lumerical scripts, `break` stops the whole script. The native optimizer uses bounded loop conditions to terminate iterations and preserve the result exports.

Lumerical's [script commands as API methods](https://optics.ansys.com/hc/en-us/articles/360041579954-Script-Commands-as-Methods-Python-API), [user-defined functions](https://optics.ansys.com/hc/en-us/articles/360034923573-function-Script-command) and [complex interpolation](https://optics.ansys.com/hc/en-us/articles/360034925893-interp-Script-command) provide the execution primitives. The standalone coordinate pattern optimizer is implemented in this project; it is not MODE's built-in optimization utility.

Native MAT import reads numeric MATLAB structs through SciPy (v5/v7) or h5py (v7.3) without opening MODE. A configured Block 0 gate can still require missing independent verification samples; ordinary uncalibrated import requires no license.

Standalone scripts also retain `blockN_last_successful_native.mat` and `blockN_last_successful.lms`, overwritten only after success. After a failed standalone attempt, `native import --bundle ... --block N --last-successful` restores that evidence without treating the incomplete active handoff as successful.
