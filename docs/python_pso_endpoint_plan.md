# Optional Python PSO endpoint backend (v21)

The default endpoint method is adaptive_1d with constrained pair selection; see [endpoint_balance_v21.md](endpoint_balance_v21.md). This document describes the optional swarm comparison.

`vlac endpoint --config config/default.json --endpoint-method python_pso` runs PSO in Python. MODE is called only for individual FDE solves through the Python API. Native MODE sweeps, optimization objects and analysis scripts are not used.

The two variables are upper input/output widths, each bounded by 0.3–2.0 µm by default. h_gap and shared h_rib remain fixed. Detuning is beta_upper-beta_lower. All eight cases (two ends × two liquid states × two band edges) contribute to the minimum absolute real detuning objective. Require input width < output width, negative cross input and positive cross output at each band edge, and 2*abs(A-B)/(A+B)<=cross_magnitude_balance_rtol for each edge's cross magnitudes. Default balance tolerance is 0.10. Neither bounds nor tolerance are automatically changed.

Invalid modes fail TE/neff/overlap checks. Sign, order or balance failures retain signed diagnostics and receive an internal -inf objective, exported as an invalid row with a blank objective. Unexpected API/license/filesystem errors propagate. Exact repeated pairs and raw eigenproblems can reuse cache entries. The lower guide uses one cross-state mode per edge in both states, preserving the explicit state-independence assumption.

Endpoint seeds use low/high, low/mid, mid/high and quarter/three-quarter normalized widths, up to the population size; remaining particles are random. Each seed must pass all constraints based on actual FDE results. `validate-pso` is a three-particle one-generation smoke test and can fail if no seed is feasible; that does not establish interval infeasibility.

Settings remain in endpoint_search: pso_generation_size=8, pso_max_generations=15, pso_tolerance=0, pso_random_seed=20261120, pso_inertia=0.7298, pso_cognitive=pso_social=1.49618, pso_velocity_limit=0.2, pso_guided_neff_margin=1e-5. Zero stopping tolerance disables early stopping. Otherwise best-score change and normalized swarm spread must remain below tolerance for three generations. All search state remains in Python.

A winner receives ten fresh mode solves (two lower plus eight upper), bypassing all cache reads/writes. Recheck all pair constraints and compare every signed detuning with the search evaluation using rtol=1e-6 and atol=1e-3 rad/m. Only a verified winner produces endpoint_pso_best.json. Failure writes endpoint_pso_verification.json and suppresses the best file.

Files: endpoint_pso_particles.csv, endpoint_pso_history.csv, endpoint_pso_generation_detunings.csv, endpoint_pso_checkpoint.json, endpoint_pso_verification.json, and endpoint_pso_best.json on success. Checkpoints are diagnostic, not automatic resume. Convergence is separate from feasibility and is not a global optimum certificate.

Maximum nominal cold workload: one startup + two lower precomputations + eight upper solves per particle evaluation + ten final verification solves. Raw caching of repeated widths and invalid-case early exits can reduce actual calls.
