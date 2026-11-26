# v18: Adaptive endpoint search and reusable FDE work

> Historical v18 design. v21 replaces independent endpoint maximization with balanced pair selection and finite slabs with boundary-spanning slabs. See [endpoint_balance_v21.md](endpoint_balance_v21.md) for current behavior.

## Commands

```powershell
python -m pip install -e .
vlac validate-endpoint --config config/default.json
vlac validate-fde --config config/default.json
vlac endpoint --config config/default.json
vlac run --config config/default.json
```

`validate-endpoint` is the adaptive backend smoke test; `validate-pso` remains the optional Python PSO smoke test. No native Optimization task, nested sweep or analysis group is used. Python owns the searches and mode selection. Small MODE scripts build/update the layout, run FDE, and gather raw mode data.

## 1. Independent endpoint optimizations

For the current centered isolated-guide calculation, define Q(w) as the minimum absolute real signed detuning across cross/bar and both band edges. A width is input eligible only if cross-state detuning is positive at both edges, or output eligible if negative at both. It must also meet the existing mode validity checks in both states. These endpoint feasibility sets are independent, so maximizing each endpoint's Q independently maximizes the minimum of the two scores. Shared geometric parameters must stay fixed and compact centering must remain enabled for this backend. Any future constraint jointly involving both endpoint widths would require revisiting the decomposition.

All modes at a sampled width are cached/reused by the shared width table. Cross-state cases run first; mixed/zero signs cannot qualify for either end, so bar-state cases are skipped. Invalid modes are recorded and do not become successful candidates. The lower modes are computed once per band edge and reused across liquid states under the same state-independence assumption as v17.

## 2. Sampling and bounded Brent

| Endpoint setting | Default | Meaning |
|---|---:|---|
| `method` | `adaptive_1d` | default Python endpoint backend |
| `adaptive_initial_widths` | 9 | initial evenly spaced widths within existing bounds |
| `adaptive_max_widths` | 33 | maximum distinct widths, including scalar refinement |
| `adaptive_width_tolerance_m` | 5e-9 | scalar width tolerance, a numerical setting rather than a fabrication limit |
| `adaptive_objective_rtol` | 0.01 | relative score-change condition for reported local convergence |

The algorithm uses part of the remaining budget to subdivide sign/validity transitions and neighboring widths whose selected fields have low overlap. If one endpoint has no eligible width, largest-interval subdivision continues within the same search budget. It does not adjust the allowed interval or run a separate cutoff scan.

It identifies sampled local maxima in feasible runs, including boundaries, and alternates input/output bounded Brent refinements. The remaining width budget is shared among identified peaks. Invalid points inside a scalar bracket receive a penalty. Every sampled valid boundary and interior value remains eligible to win, independent of the scalar optimizer's return location. The final pair is assembled from the two best sampled endpoint widths.

`converged=true` requires completed identified-peak refinements and successful, localized scalar results near the actual winners, with score changes below tolerance. It concerns sampled local branches; a finite sample set does not certify absence of narrow unsampled feasible regions or better global maxima. Hitting budgets can still produce a valid, verified pair with `converged=false`. If either endpoint lacks a valid sampled width, the search reports failure with diagnostics and leaves the bounds unchanged.

The default nominal endpoint ceiling is `1 + 2 + 4*33 + 10 = 145` findmodes calls: one startup solve, two lower precomputations, four upper cases per distinct width, and ten final verification solves. Invalid cases and caches reduce the actual count. The previous 8-particle/15-generation PSO ceiling was 973 calls under the same accounting. These are workload ceilings, not measured speedups or equal-accuracy benchmark results.

Final verification bypasses memory and SQLite for two lower plus eight upper mode solves, rechecks the four strict cross-state signs, and compares each signed detuning against the width table with rtol=1e-6 and atol=1e-3 rad/m. Fresh solves use the current MODE layout; disabling project reuse can provide a full-layout-rebuild diagnostic comparison.

## 3. Raw FDE cache

The expensive raw-candidate layer is now shared by isolated endpoint modes, bare modes during profile optimization, and coupled supermodes. Keys include enabled physical rectangles, materials/liquid index, wavelength, domain, actual cell counts, trial-mode count, material fit settings/band and interpolation grid. They exclude previous references and previous kappa. A lower-only key cannot depend on the absent upper SiN geometry, although its background liquid, interfaces, domain and mesh still matter.

Raw mode snapshots are retained in SQLite and a bounded memory LRU (`solver.raw_cache_memory_entries`, default 16; zero disables the memory tier). Selection and overlap tracking are reapplied to raw candidates using current references. Derived bare-mode and coefficient keys retain the applicable reference/gauge/selection settings. This avoids incorrectly reusing a previously selected branch while still avoiding repeated eigensolves. v18 uses a new cache schema key; older stored entries remain on disk but cannot alias the new raw/derived data.

## 4. MODE layout, materials and transfers

`solver.reuse_mode_project=true` initializes a substrate-free layout once and updates it in layout mode. Both guide bodies are present; their `enabled` flags choose upper-only, lower-only or coupled calculations. Slab topology changes rebuild layout objects without discarding installed materials. A changed material signature or disabled reuse starts a new project. Domains, mesh and active wavelength are reset from configuration for each call. The FDE object retains its default name.

Static silica/SiN materials and each distinct liquid index are installed once per project. The liquid representation retains v17's per-wavelength scalar-index treatment. `solver.batch_mode_data=true` gathers effective index, coordinates, Ex and Ey into a cell/struct payload and uses one bulk Python transfer. An empty mode set is represented without a zero-sized cell allocation. Setting the flag false restores individual getdata calls.

`run_status.json` and the progress completion event include timing totals for `geometry_setup`, `material_setup`, `eigensolve`, `field_transfer`, `field_processing`, `cache_lookup`, and `cache_write`. Actual findmodes counters remain separate from cache hit counts. These counters are the basis for measuring speed improvements on the licensed installation.

For diagnosis set reuse and batch flags independently. API/script errors propagate with their original cause; the workflow does not silently substitute a solver or change physical assumptions.

## 5. Normalized profile controls and independent grid checks

Powell receives only unit coordinates. Physical interior width/offset controls are reconstructed within their respective bounds, and endpoint controls are fixed exactly. The existing `minimum_control_step_fraction` therefore applies to dimensionless controls. A small per-grid memo avoids recalculating a recent final objective/control combination.

After optimizing a grid, the workflow obtains direct FDE coefficients on a longitudinal midpoint grid at unchanged wavelengths, and on a wavelength midpoint grid at unchanged z. The raw cache reuses old cross-sections while mode tracking is reapplied. New complex detuning/kappa values are compared with PCHIP interpolation from the original axis, normalized by Delta_beta_max and kappa_max respectively. Low mode-overlap flags force review/refinement.

Only axes failing `device.grid_convergence_rtol` are refined for another optimization stage. Passing both checks stops immediately, rather than setting a flag and continuing to double both grids. A maximum sample count that prevents a midpoint check leaves convergence unverified; the refinement budget can also stop with `grid_converged=false`. Solver success and sampling convergence remain separate fields. These checks establish convergence of local coefficients for the reported profile, not a full-device accuracy guarantee.

All optimization and grid-validation coefficients here are obtained directly from MODE or matching raw FDE caches.

## Validation and limits

The release includes numerical/integration tests using synthetic modes and recording session stubs. They test the shared-table decomposition, known scalar maxima, constrained verification, cache reuse/invalidation, template/material reuse, batch/scalar transfer equivalence, normalized optimization, early stopping and independent axis refinement. They do not execute the proprietary MODE parser or establish device performance. Run the licensed commands above on the configured Windows installation before relying on a design result.

For the current 2 um lower guide and 0.5–2.0 um upper-width range, the derived domains are 10.0 × 8.2 um compact and 26.0 × 8.2 um coupled. They use nominal meshes of 203 × 208 and 528 × 208, respectively. Mesh/domain convergence and later EME/varFDTD/FDTD comparison remain separate validation tasks.
