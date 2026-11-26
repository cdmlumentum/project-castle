# Native MODE integration

The Python package can run its FDE calculation core inside Lumerical's scripting environment. `config/native.json` selects native geometry, field processing, mode selection, coefficient extraction and complete profile-grid loops. The package retains its adaptive endpoint search, SciPy Powell optimizer, existing handoff checks and propagation stage. The current Block 2 policy uses only the two band edges and longitudinal interpolation checks.

Standalone MODE bundles provide Blocks 1 through 3 with explicit alternative algorithms: a sampled-width endpoint search and bounded coordinate pattern optimization. Their MAT results can be imported into the standard Python JSON handoff format. Preparation embeds the material tables and records configuration and script checksums. Native bundle identities are checked at block boundaries.

Use `python tools/run_vlac.py` and `python tools/run_tests.py` to select this checkout even when another vlac package is installed. `vlac-native` is also packaged as a console entry point. Native LSF source files are included in the wheel.

SQLite connections now close deterministically, resolving Windows file-lock failures during cache cleanup.

## Verification

Local Lumerical v221 comparisons at 1300 nm, 1.15 micrometre upper width, zero lateral offset and the production 528 by 208 mesh passed in both liquid states. The four selected propagation constants agreed within 2e-9 rad/m; normalized corresponding field overlaps were 1.0. The complete comparison, tolerances and configuration are in `diagnostics/native_mode/comparison.json`.

MODE interpreter checks cover B-splines, PCHIP including two-point interpolation, complex coupling arithmetic, standalone block execution with synthetic optical fixtures, MAT serialization, JSON handoff validation, and import of an existing Python profile into native Block 3. Those fixtures are explicitly marked as synthetic in `diagnostics/native_algorithms/comparison.json`.

The native sample calculations were slower than the original adapter in this local check. Whole-grid caching is available, while fine-grained raw-eigenproblem reuse across changed native profiles remains a future optimization. The native path is opt-in. These checks establish implementation behavior; a full production optimization and physical device validation remain separate work.

See [Native MODE development and execution](docs/methods/native-mode.md) for commands and limitations.
