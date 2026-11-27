# Consistent isolated-guide background (v0.27.1)

`solver.isolated_reference=retained_slabs_v1` keeps both continuous residual SiN
slabs in isolated upper-guide, isolated lower-guide and full-coupler solves.
Only the other ridge is disabled. With zero slab thickness this reduces to the
ordinary strip-guide reference. Full-coupler geometry, material functions,
domains, mesh settings, mode selection and complex coupling arithmetic do not
change. Python and native MODE/FDE paths use the same component-enable rule.

The default/native/native_sweep configurations select this named model
alternative alongside the completed `sweep_anchored_v1` Block 2 method. Configs
that omit the field keep `removed_guide_v23`: both the other ridge and its slab
are removed. Explicit joint/sequential configurations retain that historical
reference. The v23 ZIP and import manifest remain unchanged.

## Why this correction is needed

Splitting-based extraction subtracts the square of half the isolated-guide
detuning from the square of half the coupled-supermode splitting. Comparing
different slab backgrounds inserts a systematic propagation-constant difference
into that small residual. At large lateral ridge separation the slabs still
span the full domain, so the background perturbation does not disappear.

The November 27 diagnostic used the fixed original endpoint widths at 1260 and
1340 nm. In the cross-state output, the historical references produced coupling
magnitudes about 945 and 1537 rad/m, predominantly imaginary. Keeping both slabs
reduced the magnitudes below 0.02 rad/m. Spurious positive input coupling also
disappeared. This is evidence for reference-background inconsistency, not a
mesh-error bound. No negative square is clipped and no imaginary term is removed.

Local evidence is retained in the original workspace:
`outputs/block2_checkout/diagnostics/block2_negative_kappa_validation_20261127_run01/`.
Its 56 FDE solves completed in 32.55 minutes. The subsequent shared-grid study
`outputs/block2_checkout/diagnostics/mesh_study_20261127_run02/` completed 120 solves:
its horizontal ladder passed scoped comparisons but its vertical ladder failed.
Domain/PML and broader geometry/state checks were not reached. Those results do
not constitute a Block 0 certificate or physical-device validation.

## Lower-guide state dependence and execution cost

The retained upper slab can convey liquid-state sensitivity to the lower mode.
The corrected reference therefore requires
`endpoint_search.pso_lower_mode_state_independent=false`; a contradictory config
is rejected. Adaptive and PSO endpoint searches precompute four lower modes:
one per state and band edge, reused across widths and endpoints. They freshly
verify four lower and eight upper modes, retain all twelve field packets, and
report the actual state dependence. Legacy reference searches retain their
two-lower-mode reuse and ten-mode verification.

This adds two lower-mode solves per precomputation/fresh verification relative to
the legacy assumption, not one per sampled width. Block 2's cross-state width
sweep continues to reuse one lower mode per wavelength; coefficient extraction
uses the selected reference in both states. No runtime speedup is claimed.

The compact endpoint solve centers the remaining ridge while both slabs remain
full width. Full-domain profile solves retain their actual ridge placement.
The existing compact/full endpoint checks remain mandatory. Standalone LSF
bundles assume a shared lower endpoint mode across liquid states and do not
implement this alternative; preparation rejects it explicitly. Use package
commands with the native backend. Legacy bundles remain supported.

## Saved data and verification

The reference name is included in the corrected physics signature, native
payload, coefficient/selected-mode caches and checkpoint/calibration identities.
The existing versioned handoff format is sufficient: old references are not
relabeled. A historical Block 1/2/3 artifact is incompatible with the corrected
reference and fails before optimization. Its fixed geometry must be separately
reverified under the new reference before a new handoff is published; this code
change neither reoptimizes endpoints nor manufactures updated endpoint results.
Missing reference fields retain legacy loading semantics.

Run from this checkout:

```powershell
python tools/run_tests.py
python tools/validate_native_mode.py --slab-reference-only --config config/native_sweep.json --endpoint-dir <original-block1> --out diagnostics/slab_reference_<new-id> --max-solves 12 --max-seconds 1200
```

The second command verifies native/Python component-enable parity and evaluates
the unchanged saved output geometry in the cross state at both band edges, using
both implementations. It validates the old endpoint under its saved legacy
configuration solely as geometry provenance. It does not reuse its coefficients,
publish an endpoint, run optimization or perform a mesh sweep. It caps each
backend at six solves (twelve total), disables diagnostic retries, and shares a
1200-second between-batch time budget. A stalled solver call can exceed that
between-batch guard. The report uses existing coefficient/field/phase criteria
and tests that the residual coupling is small on that phase scale. Passing is
implementation parity at two sections; convergence, basis adequacy and physical
device performance are separate questions.

Sources: REF-002 equations (5)-(7) and REF-004 in the source register. The common
slab reference is an explicit project correction supported by the diagnostic;
it is not claimed as a new derivation or validation of orthogonal coupled-mode
theory. REF-016 records the numerical evidence and its limits.
