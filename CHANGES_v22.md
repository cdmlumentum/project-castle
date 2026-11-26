# v22 changes

- Add portable, checksummed block1_result.json through block4_result.json, with upstream endpoint/profile/coefficient data, saved configuration and validation.
- Load complete successful v21 endpoint directories directly, checking the passed verification report and final eight CSV cases. No endpoint rerun is required for an otherwise compatible upgrade.
- Add block1 alias, block2, block3 and block4 run/export commands and validate-block1 through validate-block4 commands. Existing endpoint/validate-endpoint/validate-pso remain available.
- Add run --endpoint-dir, --profile-dir and --bar-dir to continue from a completed block. Export checkpoints before later blocks execute, so later failures retain completed earlier stages.
- Export cross/profile data after Block 2, fixed-geometry bar data after Block 3, and propagation/performance after Block 4. Block 4 commands require no MODE runtime or license.
- Separate execution, optimizer convergence, coefficient interpolation convergence and propagation convergence. Validation exits 0 on pass, 3 for usable results needing convergence/quality review, and 2 on failure.
- Add independent bar-state z/wavelength midpoint checks. Retain existing cross-state checks and endpoint fresh verification. No full-device solver is added.
- Check physical/numerical configuration compatibility, external material-file content hashes, exact endpoint controls, grids, targets, coefficient arithmetic and artifact integrity before reusing results. Preserve complex coefficients explicitly in JSON.
- Add explicit optimizer budget overrides for Block 2 and integration overrides for Block 4, failure reports, upstream validation provenance and stale downstream-result invalidation.
- Optical dimensions/materials, endpoint objective/constraints and v21 FDE cache schemas remain unchanged.

See docs/block_handoffs_v22.md for commands, output formats, compatibility rules, validation scope and return codes. Tests use synthetic modes and recording backends; no licensed MODE result is bundled.
