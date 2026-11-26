# Vertical liquid-controlled adiabatic coupler workflow

This project orchestrates cross-section simulations in Ansys Lumerical MODE/FDE from Python. It implements endpoint detuning search, Blackman-target minimax profile optimization, separate cross/bar coefficient extraction, two-mode propagation, wavelength-dependent IL/XT export, and validation hooks.

This checkout includes local MODE integration and an optional native Lumerical script core. Start with [Native MODE development and execution](docs/methods/native-mode.md) for the implementation boundary, the runnable block scripts, and the comparison procedure. The original v23 source is recorded in [the import manifest](docs/legacy/v23-import.json).

Use `python tools/run_vlac.py ...` to run this checkout explicitly; another `vlac` package may already be installed on this computer. `python tools/run_tests.py` likewise selects this checkout. Installation with `python -m pip install -e .` makes the `vlac` and `vlac-native` entry points available in the selected Python environment.

`config/default.json` retains the original Python calculation path. `config/native.json` selects native LSF mode selection, coefficient extraction with reusable raw eigenproblem samples, while retaining the adaptive endpoint search, SciPy Powell optimizer and saved block format. Original reference files are in [docs/references](docs/references/README.md).

## Reusable execution and broadband characterization (v26)

See [v26 changes](CHANGES_v26.md) and [the execution guide](docs/methods/workflow-runtime.md) for durable attempts, shared caches, checkpoints, solve/time budgets, offline inspection/reporting and frozen-profile characterization. The native configuration now retains directly checked cross/bar broadband grids for propagation. The original Python configuration keeps edge-grid behavior unless `--characterize` is supplied. Neither path changes the physical model or reduced-mode formulation.

## Block 0: domain and mesh calibration (v25)

Block 0 searches transverse padding, mesh spacing and local refinements, validates modes and extracted coefficients against a converged reference, and supplies certified fixed solver settings to Blocks 1–3. It includes solve/time budgets, checked resume and HTML/CSV/JSON reports. Start with [the Block 0 guide](docs/methods/block0-convergence.md) and [v25 changes](CHANGES_v25.md).

`python tools/run_vlac.py block0 --help` lists the command options. A passed full calibration can be supplied as `--block0-dir` to the production blocks or standalone bundle preparation. A trial cannot issue production profiles. Licensed calibration results remain in the chosen local output directory; offline tests do not establish MODE convergence or a speedup.

## Zero-mode failure recovery (v23)

v23 explicitly sets the XY-plane FDE solver and near-max-index search, never caches an empty mode list, and retries a zero-mode solve once in a clean project with identical physical settings. It also distinguishes no eigenmodes from unusable transferred fields and writes `fde_mode_diagnostics.json` in the command output directory. The default `solver.zero_mode_retries=1` is configurable from 0 to 2. Old FDE cache namespaces are bypassed; saved compatible block results remain loadable. See [CHANGES_v23.md](CHANGES_v23.md) for scope, troubleshooting and official command references. A healthy solve has no additional eigensolve; zero-mode retries add actual findmodes calls recorded in progress.

## Run each block independently (v22)

Saved results can now be loaded between all four blocks. Your successful v21 endpoint folder can be used directly with matching configuration:

~~~powershell
vlac validate-block2 --config config/default.json --endpoint-dir outputs/endpoint --out outputs/block2
vlac validate-block3 --config config/default.json --profile-dir outputs/block2 --out outputs/block3
vlac validate-block4 --config config/default.json --bar-dir outputs/block3 --out outputs/block4
~~~

These commands run, validate and export only the named block. Block 4 needs no MODE session. `block1` aliases the existing `endpoint` command; `validate-block1` runs its small adaptive validation and exports a reusable endpoint result. `block2`, `block3` and `block4` provide the same per-block computations/exports without a nonzero exit for review-only results.

All new block commands take an output **directory**. Validation returns 0 for passed checks, 3 for a usable result needing optimizer/grid/integration review, and 2 for failure. An optimization budget limit can produce review even when execution succeeds. No target-fit objective threshold or full-device performance guarantee is implied by a numerical pass.

The full workflow can skip completed blocks:

~~~powershell
vlac run --config config/default.json --endpoint-dir outputs/endpoint --out outputs/full_run
vlac run --config config/default.json --profile-dir outputs/block2 --out outputs/full_run
vlac run --config config/default.json --bar-dir outputs/block3 --out outputs/full_run
~~~

Choose one input flag per invocation. Without one, `run` resumes compatible completed blocks by default; use `--no-resume` for a fresh search. Each completed block writes `blockN_result.json` inside its attempt before the next starts. Successful invocation publication uses `latest.json`; interrupted attempts remain available through `last_attempt.json`. Later blocks carry the required upstream data and validate configuration/integrity before reuse. A complete v21 endpoint folder needs its summary, configuration, final detuning CSV and passed adaptive/PSO verification report. See [docs/block_handoffs_v22.md](docs/block_handoffs_v22.md) for all commands, compatibility checks, validation scopes, budget overrides and failure recovery.

## Model conventions

- The propagation axis is z; each local FDE cross-section is the x-y plane. The lower-guide center is x=0; the upper-guide center is x=d_c2c. Positive d_c2c moves the upper guide toward +x.
- The lower guide is guide 2 and the upper liquid-clad guide is guide 1. Signed detuning is Delta_beta = beta_upper - beta_lower, in rad/m.
- beta_plus and beta_minus are the two coupled-section modes, ordered by descending real propagation constant. Local coupling is extracted with:

  delta_pm = (beta_plus-beta_minus)/2  
  delta_12 = (beta_upper-beta_lower)/2  
  kappa^2 = delta_pm^2-delta_12^2

  The positive square-root gauge is used initially and followed continuously along z. Complex propagation constants and complex kappa are retained when passive-material FDE returns them. The profile objective compares signed real phase detuning to its target and complex kappa to the real target.
- Isolated upper and lower modes are solved separately in their local claddings, followed by the dual-guide solve. Endpoint detuning uses those individual modes. For that endpoint-only calculation, each isolated guide is recentered at x=0 and the x-domain is compacted; the vertical stack and y-domain are unchanged. Both residual slabs extend to the FDE x boundaries. With the default 4 µm lateral ridge-to-boundary margin, the endpoint x-span is 10.0 µm, compared with 26.0 µm for the full cross-section domain. The compact mesh cell count is scaled with the x-span to preserve the nominal transverse resolution. The coupled-mode coefficient calculation still uses the full domain and field coordinates. This is the self-consistent dissimilar-guide extraction; supermode splitting alone is not treated as 2*kappa.
- FDE field profiles are interpolated onto one fixed grid. Normalized electric-field overlap, core localization, and TE-like fraction select and track modes across samples. Effective-index sorting is only used to label final plus/minus eigenvalues.
- The Blackman angle uses radians:

  theta = pi*u - 0.5952*sin(2*pi*u) + 0.0476*sin(4*pi*u), with u=z/L.

  Cross-state targets are Delta_beta_target=-Delta_beta_max*cos(theta) and kappa_target=0.5*Delta_beta_max*sin(theta).
- Endpoint search evaluates input/output × cross/bar at the two wavelength-band bounds only, under the explicit assumption that the worst endpoint detuning occurs at one of those bounds. It holds h_gap=1.0 µm and shared h_rib=175 nm fixed, and refines only the two upper-guide endpoint widths. Those widths are then fixed in the spline optimization.
- Longitudinal w1 and d_c2c are open-clamped cubic B-splines whose bounded coefficients are the control values. Offset endpoints are exactly -8 µm and +8 µm. Optional slope and curvature limits are null by default; no process-specific limits are invented.

## Geometry assumptions that need confirmation

The implementation uses the same residual slab thickness for both guides. The default geometry fixes h_rib=175 nm and h_gap=1.0 µm. Each SiN body is a ridge of height h_wg-h_rib above a residual slab. Both slabs are centered at x=0 and span the full current FDE x domain, including its PML boundaries. The upper ridge alone moves with lateral offset; its slab edges remain at the domain edges. The slab span follows compact/full domains and domain-convergence enlargements automatically. Zero h_rib omits the slabs. Isolated solves retain only the selected guide's ridge and slab, preserving the existing isolated-mode definition.

`geometry.domain_margin_x_m=4e-6` preserves the previous 10 µm compact and 26 µm coupled x spans and their 203/528 by 208 mesh cells. Legacy finite-overhang config fields are migrated into this margin to preserve their configured domain widths; slabs always reach the lateral boundaries. v21 cache schemas prevent reuse of results from the finite-slab geometry. Repeat licensed mode, mesh and domain validation for this changed cross-section.

The lower guide and vertical gap are embedded in SiO2. The FDE cross section is intentionally substrate-free: no silicon region is drawn or installed in MODE. SiO2 fills the lower half-space from the upper stack to the bottom PML; the modeled lower oxide depth is the configured 3 µm BOX plus a 0.5 µm bottom margin. Above the upper-guide bottom, the ambient is liquid; the upper ridge and slab override the liquid. This represents the lower guide as oxide-clad and the upper guide as liquid-clad. Removing the silicon substrate is an explicit simulation assumption and should be revisited if substrate loading is part of the intended physical stack.

## Materials

- **SiO2:** Malitson fused-silica Sellmeier model, valid from 0.21 to 3.71 µm. Transparent-band extinction is zero.
- **Si:** The Green 2008 intrinsic-silicon data table is retained for material-data validation only. No silicon material is installed in MODE and no silicon substrate region is included in the FDE domain.
- **Si3N4:** Luke et al. 2015 Sellmeier equation is the provisional nominal LPCVD model. Index depends on deposition, stoichiometry, stress, and process; it is not represented as process-matched. materials.sin_absolute_index_offset shifts real index by an absolute dn. For sensitivity, add measured or credible dn offsets to materials.sin_index_sensitivity_offsets. No uncertainty magnitude is assumed by default.
- **SiN absorption:** k=0 unless an explicit materials.sin_extinction_csv is supplied with wavelength_um,k columns. Roughness/scattering loss is excluded without measured or calibrated input.
- **Liquid:** constant real indices 1.55 (cross) and 1.37 (bar). For later dispersion, provide a CSV with wavelength_um,n columns and set liquid_dispersion_csv.

The Green data transcription and attribution are in data/green_si_2008_1260_1340.csv and data/README.md. Material curves are range-checked; there is no extrapolation beyond stated ranges.

## Installation and use

Install the dependencies and project:

~~~bash
python -m pip install -e .
~~~

Run independent numerical checks:

~~~bash
vlac validate --config config/default.json
~~~

The default solver settings target the Windows Lumerical v221 installation used for development: `C:/Program Files/Lumerical/v221/api/python`, visible MODE GUI, and the `windows` Qt platform plugin. The default FDE domain is 8.2 µm tall, from 3.5 µm below the lower guide to 4.7 µm above it. `solver.mesh_cells_y=208` keeps the vertical grid spacing near the v10 value while reducing the grid count with the removed substrate. `api_path` may be a folder containing `lumapi.py` or the file itself. This path is installation-specific; other releases or Ansys-managed installs may use a different folder. `solver.endpoint_compact_x` defaults to `true`; set it to `false` to use the full x-domain for endpoint isolated-mode solves.

Runtime settings can be overridden without editing the configuration. Precedence is command-line option, environment variable, then JSON configuration:

~~~powershell
$env:LUMERICAL_PYTHON_API = "C:/Program Files/Lumerical/v221/api/python"
$env:VLAC_HIDE_GUI = "false"  # true/false, 1/0, yes/no, or on/off
$env:VLAC_PLATFORM = "windows"
vlac run --config config/default.json
~~~

Or pass one-run command-line overrides:

~~~powershell
vlac run --config config/default.json --api-path "C:/Program Files/Lumerical/v221/api/python" --show-gui
vlac validate-fde --config config/default.json --hide-gui
~~~

When `--hide-gui` or `VLAC_HIDE_GUI=true` is selected, the workflow lets Lumerical choose its offscreen Qt platform automatically. Before expensive stages begin, the workflow verifies that `lumapi.py` is found, starts MODE, and runs a tiny empty-domain FDE solve to check the solver/license path.

All commands print timestamped stage progress to stderr. Python PSO reports each particle, its eight endpoint cases as they run, and the generation's best widths and signed detunings. The existing solver counters record actual `findmodes` calls; cache hits avoid those calls. The progress log records cache lookups, hits, misses and writes. `validate-fde` uses a nominal upper workload of 69 calls with defaults; raw-cache reuse can lower the actual count. Run ETAs are approximate, particularly on the first run.

Each command appends events to `outputs/run/progress.jsonl` (or the selected output directory) and updates `run_status.json` with the configuration, current/completed stages, checkpoints, and failure details. Successful runs also update `runtime_history.json` so later runs can estimate total time from prior stage timings. If FDE validation fails, `fde_validation_partial.json` points to the saved checkpoints. A failed regular run writes `run_failure.json`. The SQLite cache remains available across a rerun, so completed FDE coefficient results can be reused when their cache keys match.

The MODE run requires Ansys-provided `lumapi.py` and a working MODE license. Run:

~~~bash
vlac run --config config/default.json
~~~

The run performs endpoint search, cross-state spline optimization, separate bar-state coefficient extraction, propagation for both input layers in both states, and CSV/plot/configuration/metadata export. SQLite caches raw isolated endpoint mode candidates by guide-specific geometry, wavelength, liquid state, material models, and domain/mesh; branch selection is then applied using field overlap. Bare-mode and coupled-coefficient results remain cached by geometry, wavelength, state, material offset, solver grid, and mode-reference fingerprints. The cache reuses matching FDE samples on reruns. v22 separately supports block-level continuation through saved blockN_result.json handoffs and the input flags above.

optimizer.max_function_evaluations_per_grid, initial/refined sample counts, and candidate arrays are workload controls, not physical constraints. Increase them until the endpoint minimum, minimax objective, and control profiles stop changing. Bounded Powell is deterministic and derivative-free but finds a local solution; compare restarts and control-point counts before treating it as globally optimal.

The current default upper-guide width interval is **0.3–2.0 µm**, and the lower guide is fixed at **2.0 µm**. The full-domain x mesh uses 528 cells and the compact endpoint mesh uses 203 cells, retaining approximately the previous lateral resolution.

The default endpoint backend is **adaptive_1d**. Python builds a shared one-dimensional table of isolated-mode results versus upper-guide width and selects a constrained **pair** of endpoints. MODE performs the FDE solves; the optimizer runs in Python.

Run the endpoint block only:

~~~powershell
vlac endpoint --config config/default.json
~~~

The optimizer starts with nine widths and a budget of 33 distinct widths. Cross-state cases at 1260 and 1340 nm are solved first at each width. Negative detuning at both edges qualifies a width for input; positive at both qualifies it for output. Only qualifying widths receive the bar-state solves. Opposite signs alone are insufficient: input width must be strictly smaller than output width, and cross magnitudes must match within the configured relative tolerance at **each** band edge:

`2*abs(abs(delta_in)-abs(delta_out))/(abs(delta_in)+abs(delta_out)) <= endpoint_search.cross_magnitude_balance_rtol`

The default tolerance is 0.10 (10% of the mean magnitude). The objective remains the minimum absolute real detuning over both endpoints, both liquid states and both band edges. Bar signs are unrestricted. Magnitude balance and width order are hard constraints, with no automatic relaxation. Only the band edges are tested here; this does not certify behavior at interior wavelengths.

Every directly solved input/output pair is compared without additional FDE solves. Pair constraints invalidate the previous decomposition into independent endpoint maxima. The search refines sign/validity boundaries and builds PCHIP interpolants only within contiguous valid, sign-consistent branches with acceptable field overlap. It proposes promising pairs and roots of the interpolated magnitude-matching/balance-boundary conditions, then solves the proposed widths directly. It does not extrapolate across invalid or untracked intervals. If necessary it subdivides remaining intervals within the same budget. No separate cutoff scan or bound expansion is performed.

Only directly solved, feasible pairs can win. Interpolation guides sampling and local stopping; it does not certify an unsampled global optimum. Feasibility and convergence are separate report fields. Failure distinguishes a missing sign-eligible width from a table containing both signs but no pair meeting the joint constraints. Bounds and tolerance remain unchanged.

Final adaptive/PSO verification forces ten fresh mode solves (two lower and eight upper), bypassing both cache tiers. It repeats all mode, sign, width-order and balance checks and compares every signed detuning with the saved evaluation. A failed verification writes diagnostics and does not publish a best-result JSON. The report also lists each endpoint's residual from Block 2's common -D/+D target, since wavelength dispersion and the bar-state bottleneck can still prevent an exact target match.

The selected widths remain fixed in Block 2. Its signed target is now `-D*cos(theta)`, progressing from negative input to positive output, while coupling remains `0.5*D*sin(theta)`. Endpoint width order does not impose monotonicity on every interior spline value.
Run the new licensed smoke validation:

~~~powershell
vlac validate-endpoint --config config/default.json
~~~

It uses five initial widths and a 13-width budget, then ten fresh verification solves. A passed report establishes execution and numerical agreement for a feasible sampled pair, including width order, signs, magnitude balance, reusable MODE layout and batched transfer paths. It does not establish global optimality, mesh/domain convergence, full-device performance or a measured speedup.

The full workflow uses the same default:

~~~powershell
vlac run --config config/default.json
~~~

Profile optimization uses bounded Powell with all controls normalized to [0,1], and evaluates only the two wavelength-band edges. After each optimized grid, direct FDE checks insert longitudinal midpoints at those two wavelengths and compare complex detuning/coupling with interpolation on the original z grid, normalized by the target coefficient scales. Only z is refined. If its midpoint check cannot fit within the maximum sample count, convergence remains unverified. Optimizer success and z-grid convergence are reported separately; neither checks interior wavelengths. The legacy `device.wavelength_samples_initial` setting no longer changes Block 2's wavelength sampling.

New Block 2 results carry the two edge wavelengths into Blocks 3 and 4, so their saved coefficients and propagated performance also cover only those edges. Block 3 still performs its existing wavelength-midpoint diagnostic, which does not add interior points to the saved grid. A narrow-band or full-band performance claim requires separate interior-wavelength evaluation.

Raw eigenmode candidates are cached separately from reference-dependent mode selection for endpoint, bare-mode and coupled-mode solves. The raw key represents the active geometry, materials, actual domain/mesh, trial modes and field grid; it excludes tracking references and the previous kappa. The lower-guide raw key omits the absent upper-guide width/offset. Selected-mode and final coefficient caches retain their tracking/gauge and selection-threshold context.

A persistent MODE layout contains both guide bodies, enabling only the bodies required for each isolated/coupled solve. Geometry, mesh, wavelength and inclusion flags are updated in layout mode. Static materials and each distinct liquid index are installed once per project. Mode data are gathered into a structure/cell array and transferred to Python in a batch. Runtime status records material setup, geometry setup, eigensolve, transfer, field processing, and disk-cache timings.

For debugging, `solver.reuse_mode_project=false` rebuilds a project per solve and `solver.batch_mode_data=false` uses individual getdata calls. `VLAC_DUMP_FDE_SCRIPT` records the current layout-update script; when reuse is enabled it must be replayed against the corresponding live layout. Disabling project reuse produces a complete build script (material installation is still a separate step).

Python PSO and grid search remain explicit comparisons through `--endpoint-method python_pso` or `python_grid` on `endpoint` and `run`. `vlac validate-pso` explicitly tests the Python swarm. Existing JSON that explicitly selects PSO keeps that selection; use the provided v22 configuration or `--endpoint-method adaptive_1d` to activate the new backend. Legacy native-PSO settings migrate to Python PSO with a warning; the retired cutoff options are ignored.

See [docs/endpoint_balance_v21.md](docs/endpoint_balance_v21.md) for current endpoint constraints, geometry, settings and cost accounting. [docs/efficiency_v18.md](docs/efficiency_v18.md) records the earlier efficiency changes; its independent endpoint decomposition is superseded in v21. Optional Python PSO behavior remains documented in [docs/python_pso_endpoint_plan.md](docs/python_pso_endpoint_plan.md).

## Generated files

v22 adds `blockN_result.json` and `blockN_summary.json` after every completed stage, plus `blockN_validation_report.json` for validation commands. Block 2 exports cross/profile data immediately; Block 3 exports fixed-profile bar coefficients; Block 4 exports the complete propagation/performance bundle. See the v22 guide above for per-stage file lists.

For `vlac endpoint`, outputs default to `outputs/endpoint` (`--out` overrides). The verified adaptive best is `endpoint_adaptive_best.json`. Block 1 produces `endpoint_summary.json`, `endpoint_detunings.csv`, the PSO files below, `configuration.json`, `metadata.json`, and progress/status files. Later profile and propagation outputs are produced by `vlac run`.

Successful Block 1 runs also save these inspection outputs (including `block1`, `validate-block1`, and the Block 1 stage of `run`):

- `endpoint_betas_linear.png` and `.pdf`: Re(beta) versus position over `device.length_m`, with one panel per band edge, cross/bar upper-guide curves, the lower-guide curve, and signed endpoint detunings. Dots are calculated endpoints; connecting lines are linear illustrations, not simulated interior beta values or the optimized Block 2 profile.
- `endpoint_mode_profiles.npz`: selected isolated-mode complex normalized Ex/Ey arrays and x/y coordinates in metres. The adaptive and PSO paths retain the ten fresh verification modes (eight upper, two lower reused in both states), without extra isolated solves. These are fields on the tracking grid; raw Ez/H fields are not included.
- `endpoint_mode_profiles.png`: a visual overview of the transverse mode intensities, each scaled to its own peak.
- `endpoint_inspection.lms`: one solved full-domain, two-guide input cross-section in the cross state at the minimum wavelength. Opening it in MODE restores the geometry, materials, FDE settings and available solved mode data. This is one transverse section, not a longitudinal device model. Creating it adds one coupled FDE solve call with the configured production settings, independent of `output.save_solver_projects`.
- `endpoint_diagnostics.json`: mode array keys, identities, effective indices, geometry, coordinate frames, inspection-project selection and validation scope. Load the NPZ with `numpy.load(path, allow_pickle=False)` and use each mode's `array_prefix` to find `_ex`, `_ey`, `_x_m`, and `_y_m`.

The original `python_grid` alternative re-solves the 16 selected guide/state/endpoint/edge cases for fields because it does not retain fresh final profiles. Field exports are checked against the saved endpoint propagation constants. Reusing only a scalar JSON handoff can recreate the graph but cannot recover fields or a MODE project; diagnostics explicitly mark these as unavailable. Solver export failures prevent publication of a successful Block 1 handoff. Rerunning Block 1 removes prior inspection outputs before starting.

- `endpoint_width_samples.csv`: each distinct width, endpoint eligibility, score and mode/sign rejection
- `endpoint_width_detunings.csv`: available cross/bar signed detunings for sampled widths
- `endpoint_pairs.csv`: directly solved candidate pairs, width-order and balance checks, mismatch and feasible objective
- `endpoint_adaptive_history.json`: pair proposals, added widths, best score and local stopping information
- `endpoint_adaptive_summary.json`, `endpoint_adaptive_best.json`, `endpoint_adaptive_verification.json`: budget, selected pair and fresh-solve verification

The following PSO files are generated only when the optional `python_pso` backend is selected:

- `endpoint_pso_particles.csv`: each candidate's widths, validity, mode/sign status, objective, signed detunings, and rejection reason
- `endpoint_pso_history.csv`: best-so-far widths/objective and swarm spread after each generation
- `endpoint_pso_generation_detunings.csv`: all eight signed detunings and the worst edge for cross-in, cross-out, bar-in, and bar-out
- `endpoint_pso_checkpoint.json`: current generation, normalized swarm state, RNG state, bounds, settings and domain; for inspection, not automatic resume
- `endpoint_pso_best.json`, `endpoint_pso_verification.json`: winner and independent fresh-solve checks
- `endpoint_search.csv`, `endpoint_candidates.csv`, `endpoint_refinement.json`: additional endpoint exports from `vlac run`
- optimized_profiles.csv: w1(z), d_c2c(z), constant h_gap/h_rib, angle and targets
- coefficients.csv: signed detuning, complex propagation constants/coupling, target/residual columns, mode identities and overlap diagnostics for both states and sampled wavelengths
- performance_by_launch.csv: IL, XT, output guided power and integration diagnostics per state/wavelength/input guide
- performance_summary.csv: worst-launch IL/XT and guided-power range
- profiles_and_coefficients.png, wavelength_performance.png
- configuration.json, metadata.json, optimizer_history.json
- fde_validation_report.json with compact endpoint-domain comparison when `vlac validate-fde` is run
- sin_index_sensitivity.csv when the user configures index offsets

The design brief specifies no longitudinal coefficient target for the bar state, so its target/residual cells are blank rather than invented.

XT uses propagation.xt_power_floor as a floor on P_undesired/P_desired; default 1e-15 reports a floor of -150 dB. IL is not floored; zero desired output is infinite IL. Guided output is the sum of the two modeled output-mode powers. Loss from this two-mode basis can include absorption and radiation/basis leakage; the local model does not separate these mechanisms by itself.

## Validation coverage

vlac validate checks material units/ranges, Green-table absorption handling, analytical dissimilar/identical/near-zero coupling extraction limits, a uniform phase-matched coupler, RK4 step-halving, lossless power conservation, and exponential absorption accounting.

Run `vlac validate-fde` on a licensed MODE installation for mesh-cell and domain-span convergence, a compact-versus-full endpoint-domain comparison, a small z/wavelength mode-overlap sweep, and a large-separation coupling check. The endpoint comparison samples a representative upper-guide width, both liquid states, both wavelength bounds, and both lateral endpoint offsets. It reports changes in each isolated-mode effective index and signed detuning, plus mode-tracking overlaps and full-domain translation sensitivity. A `review` status means the compact endpoint results exceed the configured `solver.domain_convergence_neff_tolerance`; increase `geometry.domain_margin_x_m` or set `solver.endpoint_compact_x=false` before relying on those endpoint results. The endpoint check is representative; for a critical design, repeat the comparison at all refined endpoint widths. The FDE uses configurable `solver.mesh_cells_x` and `solver.mesh_cells_y`. `mesh_accuracy_refinements` contains refinement levels; each level step doubles both counts relative to `solver.mesh_accuracy`. Review the reported cell counts, changes, and low-overlap flags rather than assuming a pass from a hard-coded physical tolerance. The workflow records mode-overlap quality in coefficients.csv. `compare-full-device` can compare local-model CSV against user-generated EME/varFDTD/FDTD output using matching state/wavelength/launch rows; it does not launch a full-device solver. No full-device comparison was possible here.

For Windows MODE debugging, set `VLAC_PAUSE_ON_MODE_ERROR=1` in the shell to pause before the Python context manager closes MODE after an exception. Optionally set `VLAC_DUMP_FDE_SCRIPT` to an `.lsf` path to save the exact geometry/FDE script sent to `session.eval`. Inspect the MODE Script Prompt while paused. The saved script is useful for inspection and command-by-command replay in the same live session, where the user-defined materials have already been installed.

The two-mode model does not capture transition radiation or higher-order modes absent from the FDE basis. Calibrate it against EME/varFDTD/FDTD for selected candidates before treating IL/XT as full-device predictions. Intrinsic absorption is included through complex effective indices where supported. Sidewall roughness, scattering, fabrication variation and liquid-index uncertainty require additional inputs.

## References

- H. D'heer et al., “Nonvolatile Liquid Controlled Adiabatic Silicon Photonics Switch,” *J. Lightwave Technol.* 35, 2948–2954 (2017), [doi:10.1109/JLT.2017.2705284](https://doi.org/10.1109/JLT.2017.2705284).
- H. Ishikawa, “Fully Adiabatic Design of Waveguide Branches,” *J. Lightwave Technol.* 25, 1832–1840 (2007), [doi:10.1109/JLT.2007.899186](https://doi.org/10.1109/JLT.2007.899186).
- I. H. Malitson, “Interspecimen Comparison of the Refractive Index of Fused Silica,” *JOSA* 55, 1205–1209 (1965), [doi:10.1364/JOSA.55.001205](https://doi.org/10.1364/JOSA.55.001205).
- M. A. Green, “Self-consistent optical parameters of intrinsic silicon at 300 K including temperature coefficients,” *Sol. Energy Mater. Sol. Cells* 92, 1305–1310 (2008), [doi:10.1016/j.solmat.2008.06.009](https://doi.org/10.1016/j.solmat.2008.06.009).
- K. Luke et al., “Broadband mid-infrared frequency comb generation in a Si3N4 microresonator,” *Opt. Lett.* 40, 4823–4826 (2015), [doi:10.1364/OL.40.004823](https://doi.org/10.1364/OL.40.004823).
- Ansys Lumerical, [Python API overview](https://optics.ansys.com/hc/en-us/articles/360037824513-Python-API-overview), [Python API reference](https://optics.ansys.com/hc/en-us/articles/38660003331859-Lumerical-Python-API-Reference), and [MODE/FDE overview](https://optics.ansys.com/hc/en-us/articles/360034917233-MODE-Finite-Difference-Eigenmode-FDE-solver-introduction).
