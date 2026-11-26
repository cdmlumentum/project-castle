# v0.26.1

Block 2 can plan a completed optimization, longitudinal coefficient check and
export within a requested wall-time allowance using measured evaluation costs.
The explicit `bounded_powell_time_planned` alternative reserves validation and
report time rather than using execution timeout as its normal stop condition.
Budget-limited optimization/refinement retains honest review flags. Optical
mesh, solver tolerances, fixed endpoints and complex coupling are preserved.

Every profile export now includes exact spline geometry controls, sampled
geometry, and Blackman target comparisons in JSON/CSV/PNG/PDF. Error summaries
include complex kappa residuals and reproduce the saved minimax objective.
No physical-device validation or speedup is claimed from these changes.
