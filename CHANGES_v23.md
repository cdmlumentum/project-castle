# v23: recover and diagnose zero-mode FDE returns

The reported failure occurs during lower-guide precomputation, before the upper-width search. An empty candidate list does not establish endpoint infeasibility. The old error could also conceal a field-transfer problem if MODE found modes but all transferred transverse fields were discarded.

Changes:

- Explicitly set the XY-plane, z-propagating FDE solver (`solver type=3`, z=0) in preflight and every local solve. Keep the default FDE object name.
- Explicitly set modal search to `near n` with `use max index=1` on every local solve instead of relying on session defaults.
- Do not cache empty raw-mode candidate lists. Ignore/delete any empty entry encountered and bypass previous raw/bare/coefficient cache namespaces.
- When MODE returns no modes, rebuild the project and reinstall materials for one fresh retry with the same geometry, mesh, domain, trial-mode count and boundaries. `solver.zero_mode_retries` defaults to 1, accepts 0–2, and adds calls only for zero-mode cases. API/license exceptions are not automatically retried or converted into mode penalties.
- Confirm a zero batched count through direct `nummodes`. If direct MODE has modes, transfer those cards individually without another eigensolve.
- Treat count/card mismatches and all-zero/non-finite transferred fields as data-processing failures. Do not report them as physical zero-mode results or cache them.
- Write `fde_mode_diagnostics.json` and progress checkpoints for zero-mode attempts, recovery and unusable transfers. Include the actual wavelength, guide flags, geometry, active material rectangles, domain, mesh, search settings, native counts, and discarded-field coordinate/amplitude details where applicable.
- Preserve fresh-solve cache bypass during final endpoint verification. Zero-mode retries can add actual findmodes calls beyond the ten required fresh endpoint mode cases; the progress counters record them.

The dimensions, boundary conditions, material models, 0.3–2.0 um upper width bounds, fixed 2.0 um lower width, liquid indices and endpoint constraints are unchanged. Saved compatible v21/v22 block handoffs remain loadable. Older FDE cache entries are bypassed without deleting the user's cache database.

The user's MODE execution is not available here, so the original root cause has not been established. The correction removes implicit solver setup and persistent empty-result behavior and improves evidence if the solve still fails.

## Retry

From the new project directory:

```powershell
python -m pip install -e .
vlac validate-block1 --config config/default.json --out outputs/endpoint_check
vlac endpoint --config config/default.json
```

If it still fails, inspect `outputs/endpoint_check/fde_mode_diagnostics.json` (or outputs/endpoint for the endpoint command) and MODE's Script Prompt. For a complete layout-update script and a paused session:

```powershell
$env:VLAC_DUMP_FDE_SCRIPT = "outputs/endpoint_check/last_fde.lsf"
$env:VLAC_PAUSE_ON_MODE_ERROR = "1"
vlac validate-block1 --config config/default.json --out outputs/endpoint_check
```

The dump is the layout/FDE script; material installation is a separate step in the live MODE session. Do not change boundary conditions or physical dimensions just to force this test to pass. No successful MODE/device result is bundled; tests use recording API stubs and synthetic modes.

## Official command references

- [Ansys addfde: XY-plane solver example](https://optics.ansys.com/hc/en-us/articles/360034404294-addfde-Script-command)
- [Ansys optical mode search setup example](https://optics.ansys.com/hc/en-us/articles/360042328774-Traveling-Wave-Mach-Zehnder-Modulator)
- [Ansys nummodes](https://optics.ansys.com/hc/en-us/articles/360034405294-nummodes-Script-command)
- [Ansys tips for finding modes](https://optics.ansys.com/hc/en-us/articles/360034396914-Tips-for-finding-modes-in-FDE-and-the-mode-source)
