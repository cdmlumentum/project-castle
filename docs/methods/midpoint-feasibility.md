# Aligned midpoint feasibility diagnostic

`check-midpoint` is a separate diagnostic, named `aligned_midpoint_feasibility_v1`.
It never changes Block 1 endpoints, the Blackman amplitude, Block 2 geometry,
production numerical settings, or handoffs. It asks whether one common upper
width can provide both zero detuning and coupling D/2 at the band edges with
zero lateral offset. A magnitude ratio D/|kappa|=2 alone is insufficient: the
full complex coupling residual and signed detuning are reported separately.

```powershell
python tools/run_vlac.py check-midpoint --config <saved-configuration.json> --profile-dir <block2-output> --out <new-diagnostic-directory> --reuse-cache <old-raw-cache.sqlite> --max-solves 300 --max-seconds 5400
```

The default 17 widths span the fixed optimized endpoint interval; saved anchor
widths are included. The known profile midpoint seeds width tracking outward
in both directions. Both band edges are tracked, with bounded real midpoint
recovery. Exact-key raw packets can be copied from a checksummed read-only cache;
previous selected coefficients and their square-root gauges are not reused.
Rejected coefficients remain in the report and never supply a feasible width.
Estimated detuning crossings are evaluated with additional direct FDE samples.
No sweep of widths outside the fixed endpoint interval is implied.

Two sentinels are selected: the best common detuning fit and the strongest
common coupling width. Separate mesh factors 1.25/1.5 and x-domain factors
1.25/1.5 compare their coefficients, field identities and adjacent settings
against the existing convergence criteria. The x-domain probes retain nominal
mesh spacing. They do not test vertical domain/PML placement, interior
wavelengths, all widths, or full-device propagation and are not Block 0
certificates. Raw complex values and failed checks are retained.

`report.json` is saved after each probe; `width_sweep.csv` and PNG/PDF plots
summarize valid common widths. The CLI time budget is checked between native
batches; use an external owned-process supervisor for a hard wall cutoff.
Completion means execution finished, not that the target was reachable or all
numerical sensitivity checks passed. No target-fit acceptance threshold is
introduced. Tests use explicitly synthetic fixtures only.
