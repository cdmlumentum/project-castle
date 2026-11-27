# Native MODE development and execution

For an isolated, budgeted test of sharing lower-guide modes across liquid states,
see [the sub-hour lower-state check](lower-state-check.md). It uses the native raw
solver without enabling the approximation in the main workflow.

The Python package supports a native Lumerical script calculation core and standalone MODE entry points for Blocks 1, 2 and 3. The calculation core preserves the v23 cross-section geometry, sampled material tables, transverse-field normalization, mode selection, detuning sign and complex coupling extraction. Implementation parity is checked separately from mesh convergence and physical device performance.

## Execution choices

For certified transverse domains and meshes, see [Block 0 convergence](block0-convergence.md). `native prepare --block0-dir ...` embeds its passed profiles and certificate ID. Calibration/native profile settings have offline contract coverage; use a passed full certificate from the intended installed solver before production handoff.

| Execution choice | Endpoint search | Profile optimization | FDE and coefficient processing |
| --- | --- | --- | --- |
| Default package (`config/default.json`, `config/native.json`, or no `--config`) | v23 adaptive search or configured Python alternative | Linear width then signed-offset sweeps; exact PCHIP anchors | Guarded native MODE tracking and raw FDE batches |
| Explicit `config/python_joint.json` | Same v23 algorithms | Joint bounded Powell | Original Python adapter calling MODE |
| Explicit `config/native_joint.json` | Same v23 algorithms | Joint bounded Powell | Native raw batches, selectors and extraction; Python traverses tracked grids |
| Standalone MODE bundle | Explicit sampled-width pair search | Explicit bounded coordinate pattern search | Native LSF |

The native standalone algorithms are alternatives to the adaptive search and Powell optimizer. A sampled-width result is the best feasible directly sampled pair; it does not claim search convergence or a global optimum. The coordinate pattern search terminates when its normalized step becomes sufficiently small or its evaluation budget is exhausted. A budget-limited result needs review.

The package writes block handoff schema 4 and reads compatible schemas 1 and 2. Schema 4 records the profile representation and independent anchor coordinates; experimental schema 3 is not accepted. It caches raw candidates, tracked coefficients and complete matching grids. Python traverses z/wavelength samples in the established tracking order; missing raw solves are sent to MODE in bounded batches. This adds raw-field transfer/storage costs while avoiding repeated eigensolves across grids. See [the v26 execution guide](workflow-runtime.md) for budgets, checkpoints and retained characterization.

The initial local comparison took approximately 75 versus 94 seconds for the Python/native cross-state sample and 98 versus 108 seconds for the bar-state sample. These are single samples under the current machine load. Keep Powell in Python and profile the repeated solve/cache boundary before making a speed-driven migration decision.

## Run this checkout

The default `block2` and `validate-block2` commands use `sweep_anchored_v1`:
linear isolated-width sweeps, frozen width anchors, then signed-offset sweeps
at each selected width. `config/default.json`, `config/native.json` and
`config/native_sweep.json` select this method with guarded native FDE tracking.
See [the sweep method](sweep-anchored.md) for its algorithm, budgets, runtime
estimate and validation scope. `config/native_sequential.json` retains the
[v0.26.4 sequential Powell method](../changes/v26_4.md). The standalone native
bundle implements neither sequential method.

Block 2 exports `block2_geometry.json` (fixed dimensions and exact geometry
anchors for the default method, spline controls for legacy methods), a dense
geometry CSV, and `block2_target_comparison.{json,csv,png,pdf}`. Sweep runs also
export `block2_sweep_report.json` containing the sampled optical tables.
The comparison uses Δβ = βupper − βlower, targets −D cos(θ) and (D/2) sin(θ),
and the full complex κ residual. Absolute upper/lower β values are included in
the CSV; the workflow imposes a target on their difference. The JSON reports
maximum and RMS errors separately at every saved wavelength and verifies the
reported minimax objective against the coefficients. Plot lines between FDE
samples are for display; dense analytic targets do not imply dense FDE sampling.

For an eight-hour completion plan, use `block2 --complete-within-seconds 27000
--max-seconds 28800` with the ordinary endpoint/config/output arguments. The
Powell planning path records `bounded_powell_time_planned` and measures the slowest completed
grid per-sample cost and reserves a fresh midpoint grid, another evaluation's
margin, and 15 minutes for exports/cleanup, using a 1.5 timing factor. It ends
optional Powell work and defers further refinement when that reserve is needed;
it still evaluates the selected geometry's midpoint grid and publishes a complete
Block 2 handoff with honest convergence flags. It never coarsens the optical
mesh, changes the endpoints, or loosens tolerances. `--max-seconds` remains a
last-resort execution guard, not the scheduling mechanism. Solver stalls,
license loss, or large timing changes cannot be guaranteed to fit an estimate.
The sweep method uses the same measured-cost reserve to decide whether to start another refinement; its first full sweep and direct checks are mandatory. Completed tables/rows are checkpointed. For Powell methods, each improved feasible evaluation also saves a candidate JSON in the current
attempt. Candidates are explicitly incomplete and cannot serve as block handoffs.

Run these commands from the repository root. The launchers select the local source instead of any other installed `vlac` package.

```powershell
python tools/run_vlac.py block1 --config config/native.json --out outputs/native_endpoint
python tools/run_vlac.py block2 --config config/native.json --endpoint-dir outputs/native_endpoint --out outputs/native_profile
python tools/run_vlac.py block3 --config config/native.json --profile-dir outputs/native_profile --out outputs/native_bar
python tools/run_vlac.py block4 --config config/native.json --bar-dir outputs/native_bar --out outputs/native_propagation
```

These are production commands and can require many FDE solves. A completed compatible endpoint or profile directory can be supplied directly, including an existing v23 result. A sweep smoke execution can limit work using `--sweep-width-samples 5 --sweep-offset-samples 5 --max-refinements 0 --max-solves 200 --max-seconds 3600`; its status may be `review` or fail if the hard budget is exhausted. `--max-evaluations` applies only to Powell methods. Use a separate output directory for each study.

The supplied default and native configurations both select `solver.engine=native`. The explicit `config/python_joint.json` retains the Python backend; omitted fields in saved/custom configurations retain their historical defaults. GUI visibility, API path and thread settings retain their original configuration behavior. The provided native configuration hides newly launched MODE sessions and uses a separate cache file.

## Prepare standalone MODE blocks

Preparing a bundle does not launch MODE or use a solver license. Use the explicit
joint configuration because standalone LSF does not support sequential fitting:

```powershell
python tools/run_vlac.py native prepare --config config/native_joint.json --out outputs/native_bundle
```

The bundle contains embedded material tables in `native_config.mat`, the native function libraries, and `block1.lsf`, `block2.lsf` and `block3.lsf`. Open the scripts in MODE and run them in order. Each script saves a native MAT handoff and, after successful execution, an LMS project. The entry scripts contain the bundle's absolute working directory; regenerate the bundle after moving it.

Block 1 uses `adaptive_max_widths` evenly spaced widths unless `--width-samples` is supplied at preparation. Bounds, endpoint signs and magnitude balance are enforced. The successful pair receives ten fresh isolated-mode solves for verification. Lack of a feasible sampled pair does not prove that the continuous interval is infeasible.

Block 1 retains those ten selected modes and their SI tracking-grid coordinates in `block1_native.mat`. Before saving `block1.lms`, its entry script performs one additional full-domain solve with both guides enabled at the selected input width/offset, cross liquid state and minimum wavelength. All entry scripts first save an incomplete MAT and empty LMS using safe-mode-compatible commands; a successful result replaces them. Published Python outputs are retained in immutable attempts, so a failed rerun does not replace a prior success. MODE `native run` and `native import` publish the two-panel endpoint beta plot (PNG/PDF), field NPZ/PNG, `endpoint_diagnostics.json`, and a copy of that LMS under `python/block1`. Running the LSF directly produces the MAT/LMS files; import generates the Python plots. These outputs are inspection evidence, separate from the scalar versioned handoff.

Block 2 holds endpoint widths and offsets fixed, evaluates the clamped cubic profiles at the two band edges, fits the cross-state Blackman targets with the minimax objective, and checks coefficient interpolation at independent z midpoints. Its optimizer is named `coordinate_pattern`. This follows the current project's band-edge-only Block 2 policy. Block 3 reuses that exact profile and common grid in the bar state, with independent z midpoint checks at those endpoint wavelengths only, no profile optimization and no bar target fit.

The standalone optimizer currently supports the default unconstrained slope and curvature settings. Preparation rejects explicit slope or curvature limits; use the native package path with the existing Powell optimizer for those limits. Standalone endpoint search requires fixed gap and rib height and the existing state-independent lower-guide assumption. Missing modes, rejected mode eligibility, API failures and failed handoffs remain distinct outcomes.

To start from an existing block result:

```powershell
python tools/run_vlac.py native prepare --config config/native_joint.json --endpoint-dir outputs/native_endpoint --out outputs/native_from_endpoint
python tools/run_vlac.py native prepare --config config/native_joint.json --profile-dir outputs/native_joint_profile --out outputs/native_from_profile
```

The first command seeds the native Block 1 handoff. The second requires a compatible joint-Powell profile, seeds both Blocks 1 and 2, and allows direct execution of `block3.lsf`. A sequential or sweep-anchored profile cannot be relabeled as a standalone joint result.

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
python tools/run_vlac.py block4 --config config/native_joint.json --bar-dir outputs/native_bundle/python/block3 --out outputs/native_propagation
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

For the named guarded-tracking / monotone-Powell alternative, use
`config/native_monotone.json` with the package `block2` command. See
[v0.26.2 tracking review](../changes/v26_2.md) for the endpoint preflight,
bounded recovery, candidate rejection and short-pilot requirements. These
settings are not supported by the standalone coordinate-pattern bundle.

For a capped exploratory pilot, `tools/prepare_tracking_pilot.py` now sets the
explicit `warn_exploratory` endpoint domain policy. Mode identity, detuning signs
and coefficient extraction still must pass. Strict mode remains the default.
See [v0.26.3](../changes/v26_3.md) for its budget restrictions and the independent
`tools/check_block2_coefficient_stability.py` mesh/domain sensitivity check.

The endpoint-field diagnostic runs only selected 1260 nm isolated cross-sections:

```powershell
python tools/diagnose_block2_tracking.py --block1 outputs/endpoint --block2 outputs/profile --out diagnostics/tracking
```

It reads original fields where available and labels fresh reconstructions.
Its fresh-solve budget is 60 FDE solves / 45 minutes; it does not run either
optimization block. Field alignment uses the existing transverse E-field norm,
not Ansys's electromagnetic power-overlap integral.

- `src/vlac/native/vlac_core.lsf`: geometry, FDE, interpolation, tracking, extraction, B-splines and complete profile grids.
- `src/vlac/native/vlac_blocks.lsf`: standalone searches and midpoint convergence checks.
- `src/vlac/native_backend.py`: package adapter and source-aware native caches.
- `src/vlac/native_mode.py`: offline bundle preparation, execution and validated handoff conversion.
- `docs/legacy/v23-import.json`: hashes of the original v23 source files.

In Lumerical scripts, `break` stops the whole script. The native optimizer uses bounded loop conditions to terminate iterations and preserve the result exports.

Lumerical's [script commands as API methods](https://optics.ansys.com/hc/en-us/articles/360041579954-Script-Commands-as-Methods-Python-API), [user-defined functions](https://optics.ansys.com/hc/en-us/articles/360034923573-function-Script-command) and [complex interpolation](https://optics.ansys.com/hc/en-us/articles/360034925893-interp-Script-command) provide the execution primitives. The standalone coordinate pattern optimizer is implemented in this project; it is not MODE's built-in optimization utility.

Native MAT import reads numeric MATLAB structs through SciPy (v5/v7) or h5py (v7.3) without opening MODE. A configured Block 0 gate can still require missing independent verification samples; ordinary uncalibrated import requires no license.

Standalone scripts also retain `blockN_last_successful_native.mat` and `blockN_last_successful.lms`, overwritten only after success. After a failed standalone attempt, `native import --bundle ... --block N --last-successful` restores that evidence without treating the incomplete active handoff as successful.
