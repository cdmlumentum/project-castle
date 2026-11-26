# v26 validation record

Checks ran on 2026-11-26 against the local Windows Lumerical MODE v221 installation. They cover numerical implementation parity, API integration and reuse. They do not establish mesh/domain convergence, device performance, EME equivalence, or physical validity of the unchanged material/basis assumptions.

## Licensed checks

[Native/Python comparison](../../diagnostics/workflow_v26/native_parity.json): both cross and bar sections passed coefficient tolerances (`rtol=1e-5`, `atol=1e-2`) and normalized field overlap greater than 0.99999. The reported minimum overlap was 1.0 for each state. Ten native mathematical checks and five synthetic standalone/handoff checks also passed. The physical comparison uses the configured 1.3 micrometre cross-section, 1.15 micrometre upper width, 175 nm rib, 1 micrometre gap and zero lateral offset; detailed settings are in the report.

| Section | Python seconds | Native seconds |
| --- | ---: | ---: |
| cross | 98.46 | 96.19 |
| bar | 96.83 | 106.32 |

These are single cold-section measurements under concurrent machine load, not a controlled whole-workflow speedup benchmark. Mesh resolution and tolerances were not reduced.

[Raw reuse check](../../diagnostics/workflow_v26/raw_cache_reuse.json): three actual MODE solves took 119.06 seconds. Re-selecting with tracking references and the previous complex branch took 0.67 seconds and **zero additional eigensolves**, with matching coefficients/fields. Reopening the backend and reading the saved coefficient result took 0.044 seconds while API import was explicitly forbidden. This demonstrates reuse for the tested sample, not a general throughput guarantee.

[Native script/MAT import](../../diagnostics/workflow_v26/native_mat_import.json): ten math checks and six synthetic control-flow/handoff checks passed. The generated safe-mode-compatible scripts replace interrupted MAT/LMS handoffs with incomplete evidence. A MAT file written by MODE was imported through the offline v7.3 HDF5 reader with API import forbidden. Synthetic optical values are confined to this explicitly labeled control-flow test; they are not device results.

## Offline checks and remaining scope

The complete test suite exercises versioned handoffs, configuration guards, zero-mode recovery, cache integrity, failure publication, resumable searches, frozen-profile characterization, process sensitivity, propagation parity and offline MATLAB import. See [the captured test output](../../diagnostics/workflow_v26/offline_tests.txt) for the final count/result. Native raw-batch failures retain completed samples, and unusable fields remain execution failures rather than being labeled geometrically infeasible.

Production optimization, broadband convergence on a real optimized candidate, process-informed tolerance values, and full Block 0 calibration were not run as part of this bounded revision validation. Characterization and sensitivity are implemented and covered by synthetic regression fixtures. Physical-model alternatives, the reduced-basis audit and EME comparisons remain excluded as requested.
