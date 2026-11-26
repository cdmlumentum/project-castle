# Workflow execution, reuse and characterization (v26)

The computation remains an explicit dependency chain: Block 0 certifies transverse settings; Block 1 selects endpoints; Block 2 optimizes cross-state controls at the band edges; Block 3 extracts the bar state and optionally characterizes the frozen profile in both states across the band; Block 4 propagates the retained coefficients. Python owns search, scheduling, durable storage and publication. Native LSF owns eigensolves, candidate processing, mode selection and coupling extraction. This revision does not change the physical stack, material models or reduced-model basis, and does not add EME comparisons.

## Typical commands

Run from this checkout with the supplied launchers. The native configuration enables retained broadband characterization; the Python default keeps the prior edge-grid Block 3 behavior unless `--characterize` is supplied.

```powershell
python tools/run_vlac.py plan --config config/native.json
python tools/run_vlac.py block1 --config config/native.json --out outputs/endpoint --max-solves 500 --max-seconds 7200
python tools/run_vlac.py block2 --config config/native.json --endpoint-dir outputs/endpoint --out outputs/profile --max-solves 3000 --max-seconds 14400
python tools/run_vlac.py characterize --config config/native.json --profile-dir outputs/profile --out outputs/characterization --max-solves 3000 --max-seconds 14400
python tools/run_vlac.py block4 --config config/native.json --bar-dir outputs/characterization --out outputs/performance
python tools/run_vlac.py inspect --config config/native.json --source-dir outputs/characterization --block 3
python tools/run_vlac.py report --config config/native.json --source-dir outputs/characterization --block 3 --out outputs/regenerated
```

`run` uses the same block functions. With resume enabled, it finds the furthest compatible completed block in the output directory's attempts, including a completed block from an interrupted workflow. Explicit `--endpoint-dir`, `--profile-dir` and `--bar-dir` still take precedence. A changed optimizer or sampling configuration prevents automatic completed-block reuse. Saved compatible artifacts may still be supplied explicitly. `--no-resume` restarts searches and characterization; physical caches remain enabled. Final endpoint verification always performs ten fresh isolated solves.

## Outputs and status

Each invocation writes into `.attempts/<id>` beneath its output directory. `latest.json` is replaced only when that invocation completes. Root-level files are convenience copies; package readers resolve the pointer. `last_attempt.json` identifies a failed or budget-limited attempt without replacing the previous successful result. Completed blocks inside that attempt remain reusable by directory or explicit `blockN_result.json` path. Keep the whole directory when moving a run, or copy the self-contained result JSON for a portable handoff.

Schema 2 adds implementation/installation provenance, a timestamp-independent payload content ID, and a separate frozen-geometry ID. Readers retain schema 1 and legacy endpoint import support. The complete artifact checksum, coefficient shapes, endpoint constraints, target values, complex encoding and configuration compatibility remain checked. A characterization dataset has its own checksum and must match the frozen controls.

`validation.status` describes this block; `validation.overall_status` also includes upstream review. Validation commands return 3 when the overall result needs review. A complete artifact can legitimately need review because optimizer, interpolation or propagation limits were reached. A solve/time stop is incomplete, not numerical convergence. Neither status is physical device validation. Block 4 reports propagation step convergence separately from coefficient sampling and upstream optimizer convergence.

## Cost and budgets

Both shipped configurations specify 12,000 eigensolve attempts and 14,400 seconds per invocation. Override them using `--max-solves` and `--max-seconds`, or set explicit JSON values. Time limits are soft checks between batches: a running MODE eigensolve is allowed to finish. Native raw batches reserve all allowed zero-mode retries before dispatch; a smaller remaining budget can stop before using every last solve. The default batch size is four raw requests, with individual coefficient evaluations normally requesting three.

`plan --profile-dir outputs/profile` estimates the independent Block 0 design check as well as the optimization grid. For example, 17 saved z points produce 33 verification points; two edges plus the center, both states, isolated/coupled solves and selected/reference meshes give 1,980 cold solves before retries. `convergence.max_solves` remains the separate Block 0 gate budget; raise that explicitly when needed. Reusable samples make resumed checks cheaper. The plan is an estimate, not an admission guarantee or a convergence result.

The standalone sampled-width and coordinate-pattern LSF alternatives retain their own width/evaluation/refinement limits. The execution solve/time budget applies to Python `blockN`, `run`, characterization and sensitivity commands using the adapters. `native run` and scripts run directly in MODE use the standalone search limits; they do not provide a wall-clock watchdog or the shared SQLite sample cache. Keep standalone runs in separate bundles and select explicit search budgets when preparing their configuration.

## Cache and checkpoint behavior

`--cache-dir` chooses a shared local cache directory. `VLAC_CACHE_DIR` is an environment-wide override and takes precedence. A local SSD outside a synchronized folder is preferable for large caches. Cache hits that already contain a complete requested result do not import `lumapi` or acquire a license; deferred preflight runs on the first actual MODE use.

The SQLite v2 namespace contains checksummed compressed numeric arrays and JSON metadata, loaded with `allow_pickle=False`. Legacy pickle rows are ignored. Raw native candidates are independent of tracking history; absent guide widths/offsets do not distinguish identical isolated problems. Compact and full-domain problems remain distinct. Tracked coefficients also include reference fingerprints and the previous complex-coupling branch. Whole-grid caching remains available. Repeated samples can therefore avoid eigensolves while selecting modes in the requested z/wavelength order. Failed/empty native results never become cache successes.

Optical keys include implementation hashes, the MODE executable/API identity, material contents, geometry and numerical settings. Runtime GUI and output paths do not define the physics. Native configuration uploads are skipped when unchanged; scene objects and invariant material fits are reused, with clean projects on changed configurations or zero-mode retries. Local mesh overrides are rebuilt explicitly. Raw field transfers and compression have a cost; a cold single-section run is not expected to prove a workflow speedup.

```powershell
python tools/run_vlac.py cache --config config/native.json
python tools/run_vlac.py cache --config config/native.json --max-bytes 1000000000 --vacuum
```

The automatic limit is 2 GB of compressed payload, with least-recently-used eviction. SQLite may retain free pages; explicit vacuum reclaims them. Checkpoints and archived attempts are separate evidence and are not evicted by cache pruning. Inspect their size and archive completed runs as needed.

Adaptive search saves its width table, mode references and lower-mode evidence after each completed width. Powell saves its best feasible controls and completed grid history; it resumes as a **warm restart**, not exact restoration of SciPy's directions/line search. Broadband refinement saves each completed grid. Checkpoint replacement is atomic and includes a checksum; corrupt checkpoints are rejected. Increase a solve/time budget and rerun to continue. Changing the numerical/physical identity starts a separate checkpoint.

Block 0 design checks reuse checksummed independent selected/reference samples keyed by the certificate, design, coordinates, implementation and verification criteria. Block 3 can reuse Block 2's reference solves while checking its newly saved bar coefficients. Saved-coefficient comparisons are rerun; ordinary optimization-cache entries never substitute for independent verification. Set `execution.reuse_verification=false` for a separate fresh evidence directory.

## Frozen-profile characterization

`characterize` runs Block 3 on saved Block 2 controls. It seeds the wavelength grid with the configured broadband samples and saved edges, then directly calculates cross/bar coefficients on nested z/wavelength grids including joint midpoints. It compares complex detuning, coupling and common attenuation against tensor PCHIP interpolation (the attenuation term uses the log-power estimate `2*length*error`), checks coefficient quality and retains the checked dense grid. The original optimization solution and its edge-grid evidence remain in the artifact. Block 3 also exports `characterization_coefficients.csv` and `characterization_summary.json`. Axis or refinement limits produce review, never a fabricated pass. This establishes sampled coefficient interpolation accuracy at the configured transverse discretization, not an error bound over all unsampled points.

Block 4 automatically uses the retained characterization dataset when present. Both launch columns share interpolated RK4 matrices and common attenuation; each launch retains its own convergence stopping step. Fixed endpoint widths, profile controls, upper-minus-lower detuning and the complex coupling branch are unchanged.

## Process sensitivity

`sensitivity --profile-dir ... --cases process-cases.json --out outputs/sensitivity` evaluates explicitly supplied deterministic perturbations without re-optimizing. The JSON needs a nonempty `source` describing the process/metrology evidence and a `cases` list with unique `name` values. Allowed offsets are `upper_width_bias_m`, `lower_width_bias_m`, `thickness_bias_m`, `gap_bias_m`, `rib_bias_m`, `alignment_bias_m`, `sin_index_offset` and `liquid_index_offset`. Lengths are SI; index shifts are dimensionless. No default fabrication tolerance or yield distribution is invented.

Each case retains both states, both launches, coefficient quality, input identity and a resumable result. The initial implementation uses the saved optimization sampling and reports review because it does not establish case-specific interpolation convergence or yield. A supplied liquid dispersion table cannot silently be replaced by a constant offset. Perturbations cannot inherit a Block 0 certificate for a different model/envelope; supply a matching `block0_dir` in each perturbed case where certification is required. This capability does not implement the excluded physical-model alternatives, basis audit or EME comparisons.

## Validation scope

Run `python tools/run_tests.py` for offline contracts, synthetic numerical fixtures and failure recovery. Run `python tools/validate_native_mode.py --max-solves 24 --max-seconds 900 --out diagnostics/native_v26` for bounded licensed numerical parity. The limits apply per backend in that comparison. Reports distinguish synthetic control-flow checks, actual MODE samples, numerical parity and physical validation. See [the validation record](../validation/workflow-v26.md) for measured results and remaining limits.

Standalone scripts also retain `blockN_last_successful_native.mat` and `blockN_last_successful.lms`, overwritten only after success. After a failed standalone attempt, `native import --bundle ... --block N --last-successful` restores that evidence without treating the incomplete active handoff as successful.
