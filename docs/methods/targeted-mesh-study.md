# Endpoint shared-grid mesh study

`endpoint_shared_grid_mesh_study_v2_neff_kappa_only` is a diagnostic alternative, separate from
full Block 0 certification and production Block 2. It retains both continuous
slabs in single-ridge references, and keeps mesh boxes at both ridge locations
and across the intervening optical region in all three coefficient solves.
Production background definitions, tracking and complex coupling branch
conventions are unchanged. No negative kappa squared is clipped.

The user stopped the initial v1 study and selected `neff_kappa_only_v1` on
November 27, 2026. Effective-index acceptance is now `1e-4`; detuning differences
and their length-times-error estimates remain diagnostic outputs only. Complex
coupling absolute/relative and phase limits, loss, fields, polarization and mode
validity retain their previous settings. Existing v1 results keep their original
limits and are not relabeled as a completed v2 run. A fresh output directory and
explicit fresh solve/time budgets are required for a rerun.

The first background comparison records both original and matched-slab
definitions on common grids. Their difference is a model difference, not mesh
uncertainty. Localization, TE fractions, field identity and basis overlap are
retained; E-field metrics do not establish an electromagnetic power-normalized
orthogonal basis or validate the two-mode model.

The study freezes the initial monotone spline geometry based on existing Block 1
endpoints. Probes are input and zero-offset center, at 1260 and 1340 nm in the
cross state. The center is not asserted to be the phase-matching width. Local x
spacing is refined with y fixed, then y with x at its finest passing reference
(or finest attempted value if x fails). The provisional ladder is 40, 30, 20,
15, 10, 7.5 nm. Two consecutive passing comparisons are required per axis.
Failure of either axis prevents domain work and production optimization.

After independent passes, a combined candidate/reference comparison covers the
input, zero-offset center, +3 um shoulder and output at both endpoint wavelengths
in both states. Lateral, top and bottom expansion each receive two checks with
fixed background spacing/registration. Actual coordinates in the optical region
must match; requested counts alone do not establish a fixed lattice. An extra
PML-layer check and combined mesh/domain/PML guard complete the scoped study.
All selected Block 0 metrics must pass: neff, complex coupling and its accumulated
phase, fields, loss, polarization and grid aliasing. Detuning does not gate a pass.

Removed-ridge dimensions remain in raw cache keys. Samples include actual grids
for all three solves, source identities, exact geometry, coefficients, selected
fields and checksums. Per-sample JSON checkpoints and compressed NPZ fields
survive failed/incomplete gates. Existing output directories are rejected rather
than resetting budgets.

```powershell
python tools/run_vlac.py mesh-study --config config/native_tabulated.json --endpoint-dir ../block1_v0250_fixed_20261126_run01 --out diagnostics/mesh_study_run01 --max-solves 300 --max-seconds 17700
```

`tools/supervise_mesh_study.py` adds a separate session directory, idle-sleep
prevention and a five-hour hard ceiling on its own child tree. The ordinary
budget ends five minutes earlier for cleanup; timeout is an emergency guard.
Fine-grid runtime is measured. Budget exhaustion is incomplete, never convergence.
Reports: `mesh_study.json`, `comparisons.csv`, `mesh_convergence.png`,
`mesh_convergence.pdf`, `report.md`.

A scoped numerical pass does not issue a Block 0 certificate, publish solver
settings, refresh Block 1 or automatically launch Block 2. The changed reference
basis and affected endpoint handoffs need separate verification before a pilot.
Physical device validation and fabrication accuracy remain separate.

References: REF-008/009/010 in the source register and the full
[fixed-lattice calibration method](block0-convergence.md).
