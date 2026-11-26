# v18: Computational-efficiency revision

## Endpoint search

The provided default configuration now selects `adaptive_1d`. A shared width table supports independent input/output scalar optimizations, retaining the existing absolute-detuning objective, cross-state sign restriction, fixed geometry, band-edge policy and lower-mode assumption. The decomposition is exact under those independent endpoint constraints; finite-width sampling is not a global-optimum proof.

The search starts with nine widths, prunes bar solves for widths whose cross-state signs cannot qualify for either end, subdivides observed sign/validity or tracking transitions, and uses bounded Brent refinement around sampled local maxima. Largest-interval subdivision continues if a sign branch was not initially found. The hard budget is 33 distinct widths by default, with configurable 5 nm numerical width tolerance and 1% relative local score condition. No width preflight or automatic bounds expansion was added.

Ten fresh mode solves bypass all raw caches to verify the selected pair. `validate-endpoint` tests the adaptive method with five initial widths and a 13-width budget. Optional `validate-pso`, `python_pso`, and `python_grid` remain available. Explicit backend choices in old configurations are respected; use v18's default file or `--endpoint-method adaptive_1d` to select the new method.

## FDE reuse

Raw eigenmode candidate caching now applies to profile-stage isolated and coupled calculations. Actual enabled geometry, materials, wavelength, domain/mesh and field grid define raw keys. Tracking/gauge references are applied after raw lookup. The lower-only raw key excludes absent upper SiN geometry. Derived selected-mode/coefficient caches retain reference and selection settings.

An in-memory LRU complements SQLite. Material-fit settings and fit band enter cache identities. v18 keys prevent older cache schemas from aliasing the new data; old entries are not deleted.

The MODE project and rectangle/FDE objects are reused, with geometry, mesh, wavelength, active liquid and enabled guide bodies updated in layout mode. Static materials and distinct liquid-index materials are installed once per project. Cell/struct mode-data collection reduces Python transfers. Project reuse and batch retrieval can be disabled independently for diagnosis.

Status and completion records include geometry/material setup, eigensolve, field transfer/processing and cache timing totals, plus actual solve and cache-hit counts.

## Profile optimization

All interior spline controls are optimized in [0,1] and mapped back to physical bounds. Endpoint controls remain exact. The fractional step tolerance now operates on unit coordinates.

A small recent-objective memo avoids a repeated final evaluation. After each optimized grid, direct FDE midpoint checks independently assess z and wavelength interpolation errors for complex detuning and coupling. Only failing axes are refined; passing both checks immediately stops the loop. An unchecked axis at its maximum size or an exhausted refinement budget is reported without claiming convergence. Optimizer success remains separate from coefficient-grid convergence.

The simulation domains, materials and two-mode propagation model are retained. All optimization/grid-check coefficients are calculated directly with MODE or read from matching FDE caches.

## Validation performed

- 38 numerical/integration tests passed, including known endpoint extrema, hard budgets, sign rejection, fresh verification, raw-cache reuse/invalidation, reusable layouts/materials, bulk/scalar transfer equivalence, normalized optimization, early stopping and independent-axis refinement.
- Existing material, geometry, coupling and propagation checks passed.
- Editable installation of version 0.18.0, CLI help, Python syntax and retired-native-command source audit passed.
- Synthetic modes and session stubs were used for software integration tests. No licensed MODE parser/eigensolver run or measured wall-time speedup was available in this environment.

See `validation/release_validation.json`, `validation/test_results.txt`, `validation/analytic_validation.json`, and `docs/efficiency_v18.md`.

```powershell
python -m pip install -e .
vlac validate-endpoint --config config/default.json
vlac validate-fde --config config/default.json
vlac endpoint --config config/default.json
```
