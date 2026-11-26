# v16: Python-controlled endpoint PSO

- Python now owns swarm initialization, normalized updates, deterministic RNG seeding, mode validity, objective values and stopping. Defaults remain 8 particles and 15 generations. NumPy is already a dependency.
- MODE runs each cross-section directly through the existing Python adapter. Removed the native Optimization/nested-sweep builder and its analysis groups, including the temporary-object lookup and unsupported `getanalysis` failure paths.
- Removed the upper-width cutoff preflight, width bisection and automatic expansion. Candidate validity checks stay inside each objective evaluation; configured width bounds are preserved.
- Signed eight-case detunings and four endpoint/state summaries are available during optimization. Particle CSV updates after every candidate; history, generation detunings and swarm checkpoint update after every generation.
- Final verification performs ten fresh FDE mode solves with SQLite bypassed, then compares each signed detuning with the winning evaluation. A disagreement fails the run and does not write a best-candidate success file.
- `endpoint` and `run` use the same configurable dispatcher and default to `python_pso`. Both accept `--endpoint-method python_grid`. `validate-pso` now exercises the Python backend with three particles and one generation.
- Endpoint-only exports and full-run metadata use `pso_summary`. Old JSON backend/reset settings migrate with a warning; removed cutoff settings are ignored. The old native backend CLI choice is retired.
- Retained and checked the substrate-free geometry already present in v15: silica to the bottom PML, no silicon object/material, 8.2 um y-span and 208 y mesh cells. The endpoint x-span is 9.2 um with 187 cells; coupled solves retain 25.2 um with 512 cells. Domain sizing now includes the wider of the upper and lower guides even if upper bounds are narrowed.
- Versioned raw-mode and coupled-coefficient cache identities prevent reuse of older silicon-inclusive results. Previous native setup/project/preflight files and endpoint success records are removed when starting a new PSO block in a reused output folder.
- Updated documentation and added 16 numerical/integration tests. Tests cover a known PSO optimum, reproducibility, bounds, invalid modes, unmasked runtime errors, the eight-case objective, lower-mode reuse, fresh verification, legacy migration, direct FDE script generation, CLI execution and full-workflow exports.

## Validation performed

All 16 tests passed. Existing material, geometry, coupling and propagation validation passed. Editable installation and installed CLI help passed. See `validation/release_validation.json` and `validation/analytic_validation.json`.

MODE was not installed or licensed in this environment. Synthetic test modes establish software behavior only; no physical endpoint widths, detunings or performance are claimed. Run the licensed smoke and convergence commands below on the Windows MODE machine before the endpoint design run.

```powershell
python -m pip install -e .
vlac validate-pso --config config/default.json
vlac validate-fde --config config/default.json
vlac endpoint --config config/default.json
```
