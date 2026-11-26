# Changes in v25 / 0.25.0

## Native reference serialization fix

- Encode mode-reference effective indices as 1 x 1 complex NumPy arrays for the installed v221 `lumapi` interface. Preserve their real and imaginary components, field arrays, and tracking metadata.
- Cover reference round trips and the two-wavelength isolated-mode API boundary with synthetic regression tests. This fixes the `Unsupported data type` failure encountered by Block 1 when sending its first wavelength-tracking reference.
- Geometry, materials, solver resolution, selection criteria, and optimization budgets are unchanged.

## Initial release

- Add Block 0: bounded native MODE domain/mesh calibration, consecutive-step reference checks, PML/joint checks, coordinate screening, local overrides, adaptive probes and expanded-envelope gates.
- Check complex detuning/coupling, accumulated phase, numerical loss, transverse E-field overlap, pair subspace and TE fraction alongside effective index.
- Add checksummed numeric checkpoints, resumable solve/time budgets, measured timings and HTML/CSV/JSON reports.
- Issue isolated/coupled JSON/MAT/configuration profiles only after a completed, real, full calibration. Trials and synthetic runs cannot certify production settings.
- Apply certificates to Python/native Blocks 1–3 and standalone bundles; add certificate-aware caches/handoffs and selected-design gates before package publication.
- Add `block0`, `verify-block0` and `--block0-dir`. Preserve Block 2's two-band-edge objective and saved wavelength grid.

See [the execution guide](docs/methods/block0-convergence.md). Verification for this revision is offline. The licensed trial was deferred by the user pending further instruction. No new real MODE convergence or runtime improvement is claimed.
