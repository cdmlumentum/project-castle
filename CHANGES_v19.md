# v19: Revised liquid refractive indices

The liquid indices are now constant real values of 1.5 in the cross state and 1.2 in the bar state. Both the supplied configuration and the Python configuration defaults use these values.

| Parameter | Value |
|---|---:|
| Cross-state liquid index | 1.5 |
| Bar-state liquid index | 1.2 |
| Upper-guide width range | 300–1200 nm |
| Fixed lower-guide width | 1000 nm |

The complete geometry configuration was compared with v18 and is unchanged. The adaptive endpoint search, signed-detuning restrictions, mesh/domain settings and computational-efficiency improvements are retained.

Exported material descriptions now obtain liquid indices from configuration, so output metadata reflects the selected material settings. Existing raw FDE cache keys include the actual liquid index; entries from the old indices cannot match new-index solves. No cache deletion is needed.

All 38 software tests and the existing analytical validation checks passed using the revised configuration. Synthetic mode fixtures verify software behavior only. A licensed MODE run is needed to determine endpoint feasibility and device coefficients with the new indices.

From the extracted project folder:

```powershell
python -m pip install -e .
vlac validate-endpoint --config config/default.json
vlac endpoint --config config/default.json
```
