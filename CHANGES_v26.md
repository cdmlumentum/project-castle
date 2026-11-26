# v26: reusable workflow execution and frozen-profile characterization

- Preserve successful artifacts with immutable attempts and atomic publication; resume full runs from compatible completed stages.
- Add schema 2 provenance, payload/frozen-geometry identities and overall validation status while retaining older handoff readers.
- Cache native raw eigenproblems separately from tracked results, persist bounded batches, defer MODE startup until needed and reuse scene objects/material fits.
- Replace pickle cache loading with checksummed numeric storage, configurable byte limits, inspection and explicit pruning.
- Add adaptive width and Powell warm-start checkpoints, shared solve/time budgets and reusable independent Block 0 verification samples.
- Add retained cross/bar broadband characterization of frozen controls, joint z/wavelength midpoint checks and automatic use of that dataset in Block 4.
- Share interpolation and RK4 work between launches; retain separate convergence decisions.
- Add offline plan/inspect/report operations and process-supplied deterministic sensitivity cases.
- Reconcile native safe-mode handling and add offline MAT imports. A failed standalone attempt saves an incomplete MAT and empty LMS; prior published Python artifacts remain intact.
- Fix Block 0 execution on MODE v221 by detecting optional PML autoscaling support; retain actual mesh/PML evidence and the original accuracy guards.
- Preserve the uniform mesh lattice during reference-domain expansions, including cell parity and roundoff-safe counts, under the explicitly versioned `mode_domain_mesh_calibration_v2_fixed_lattice` method.
- Expose native solver error details and conservatively account for failed Block 0 calls in the solve budget.

The physical stack/material alternatives, reduced-model basis audit and EME comparisons are excluded. Existing SI units, fixed endpoints, detuning sign, selection/tracking rules and complex coupling convention are preserved. No new model validation or global-optimality claim is made.

See [workflow execution and reuse](docs/methods/workflow-runtime.md) for commands, migration, budgets and validation scope.

[Validation record](docs/validation/workflow-v26.md) includes licensed MODE parity, measured raw reuse and offline import evidence.
