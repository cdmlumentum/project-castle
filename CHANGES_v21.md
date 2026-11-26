# v21 changes

- Upper-guide width bounds changed to 0.3–2.0 µm; grid seed widths 0.3/1.15/2.0 µm. Bottom guide stays 2.0 µm and cross/bar liquid indices stay 1.55/1.37.
- Both residual SiN slab layers span the current FDE x domain, including at nonzero upper-ridge offsets. Compact/full domains and nominal mesh settings are preserved. Old finite-overhang fields migrate into the lateral margin; new cache schemas isolate the changed geometry.
- Require upper input width < output width, cross input detuning <0 and cross output >0 at both wavelength bounds, using the existing beta_upper-beta_lower convention.
- Add a hard same-wavelength cross-magnitude balance constraint: 2|A-B|/(A+B)<=epsilon at both band edges. Default epsilon=10%, configurable by endpoint_search.cross_magnitude_balance_rtol.
- Replace independent endpoint maxima with constrained pair selection from the shared 1D width table. PCHIP interpolation, inverse magnitude matching/tolerance boundaries and interval subdivision guide direct MODE width solves within the existing budgets.
- Retain the worst-case cross/bar objective, mode guards, raw-cache reuse, Python-only optimization and ten fresh adaptive/PSO verification solves. Apply the new pair constraints consistently to optional PSO/grid and CLI/full-run exports.
- Reverse Block 2 target to -D*cos(theta); keep kappa target D*sin(theta)/2. Record balance, width order, per-edge signed values and residuals from the common target. Add endpoint_pairs.csv and clearer pair-feasibility failures.
- Add regression tests for narrow feasible strips, independent per-edge balance, explicit width order, fresh verification rejection, legacy geometry migration and full-width slabs across offsets/domain changes.

Tests use synthetic modes and recording API stubs. No licensed MODE or physical device result is included. Prior finite-slab endpoint widths are not validated for the new geometry.
