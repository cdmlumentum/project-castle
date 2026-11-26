# v17: Mandatory cross-state endpoint detuning signs

The endpoint search now requires positive real detuning at the input and negative real detuning at the output in the cross state, at both configured wavelength-band edges. Signed detuning is beta_upper minus beta_lower. The orientation matches the existing Blackman target, which goes from +Delta_beta_max to -Delta_beta_max.

- One shared sign validator is used by Python PSO, the alternate Python grid search, fresh-solve verification, endpoint summary, validation report and full-run metadata.
- Wrong signs and exactly zero detuning reject the candidate. Among feasible pairs, the existing minimum absolute detuning objective across all eight cases is retained. Bar-state signs receive no additional constraint.
- Particle CSVs distinguish mode failures from sign failures and retain all eight signed detunings for pairs rejected by the sign check. Rejected candidates cannot become a successful winner, even when no feasible candidate is found.
- Final ten-solve verification repeats the strict sign checks before comparing the fresh detunings. Sign failure writes a failed verification report containing the four checks and all eight records; it does not write a successful best-candidate file.
- The three-particle smoke run now samples upper-guide width pairs high/low, high/mid and mid/low: 1200/300, 1200/750 and 750/300 nm with default bounds. The full swarm additionally seeds three-quarter/one-quarter, then samples remaining particles randomly. There is no new preliminary scan or automatic bound adjustment.
- The requirement is mandatory without a new configuration switch. Existing configurations continue to load. Matching raw FDE caches remain reusable because geometry and materials did not change; candidate eligibility is evaluated again with the new restriction.
- Added regression coverage for same-width and reversed endpoints, individual band-edge violations, zero/nonfinite detuning, unrestricted bar signs, lack of a feasible pair, rejection diagnostics, grid-search enforcement, fresh verification failures and agreement with the Blackman endpoint orientation.

## Validation

All 24 numerical/integration tests passed with explicitly synthetic modes. The full software path through profile optimization, propagation and export is exercised by an integration fixture; its coefficients are not device predictions. Licensed MODE validation is still required for this constrained release.

```powershell
python -m pip install -e .
vlac validate-pso --config config/default.json
vlac endpoint --config config/default.json
```

The smoke run has only three candidate pairs and one generation. If none meets the mode and sign conditions, it reports failure. This is not proof that the entire configured interval is infeasible. The full `endpoint` command uses the configured search budget and also fails explicitly if it finds no valid pair.
