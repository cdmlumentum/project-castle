# v20: Updated liquid indices and waveguide widths

| Parameter | Value |
|---|---:|
| Cross-state liquid refractive index | 1.55 |
| Bar-state liquid refractive index | 1.37 |
| Fixed bottom-guide width | 2.0 um |
| Upper-guide width range | 0.5–2.0 um |
| Optional grid-search seed widths | 0.5, 1.25, 2.0 um |

The supplied configuration and Python configuration defaults agree. Exported material descriptions read the configured indices.

The derived compact endpoint domain is 10.0 × 8.2 um and the full coupled domain is 26.0 × 8.2 um. Full-domain x cells were increased from 512 to 528 to approximately preserve the prior nominal lateral spacing as the geometry widened. The compact endpoint x count scales to 203; y cells remain 208. Other geometry, materials, search and propagation settings retain their prior values. Raw FDE cache identities include actual geometry, liquid index, mesh and domain, so the new cases cannot alias older parameter sets.

The adaptive endpoint method, strict positive-input/negative-output cross-state detunings, raw-mode cache and reusable MODE layout continue to apply. Endpoint feasibility with these dimensions and indices requires a licensed MODE solve.

All 38 software tests and the analytical material, geometry, coupling and propagation checks passed. Tests use synthetic modes/session stubs and do not claim a device performance result.

From the extracted project folder:

```powershell
python -m pip install -e .
vlac validate-endpoint --config config/default.json
vlac validate-fde --config config/default.json
vlac endpoint --config config/default.json
```
