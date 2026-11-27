# Sequential linear sweeps and Blackman anchors (v0.27.0)

`sweep_anchored_v1` is the default in `config/default.json` and `config/native.json`.
`config/native_sweep.json` selects it explicitly. It replaces whole-profile Powell
optimization with two sequential families of one-dimensional optical sweeps. The
geometry representation is `pchip_anchors_v1`: an interpolant through selected
geometry values at specified z positions. These values are not B-spline controls.

## Algorithm

1. Load the compatible Block 1 endpoints and perform the mandatory strict
   endpoint-mode consistency check. Preserve the optimized widths, rib height,
   vertical gap, signed offset endpoints, materials and optical solver settings.
2. At both wavelength-band edges, linearly sweep upper width between the fixed
   endpoint widths. Track isolated upper modes on the full production domain at
   x=0. Reuse the invariant lower-guide mode within each wavelength sweep. Form
   upper-minus-lower detuning from the directly calculated propagation constants.
3. Evaluate the Blackman targets at the initial z grid:
   `delta_target(z) = -D*cos(theta(z))`, `kappa_target(z) = D/2*sin(theta(z))`.
   Linearly interpolate the optical table onto a finer candidate width grid.
   Select an ordered width path minimizing the worst normalized detuning residual
   over both wavelengths and all z anchors, with the endpoints fixed exactly.
   The finite-grid dynamic program resolves ties deterministically. This is not
   a claim of a continuous global optimum or an exact fit to an unreachable target.
4. Freeze those widths. For each distinct selected width, linearly sweep signed
   center-to-center offset between its fixed endpoints at both band edges. Include
   the aligned position if it lies in that interval. Each row tracks modes and
   the complex coupling branch across offsets and then wavelengths. Select an
   ordered offset path minimizing the worst full complex coupling residual.
   Coupling depends on width, offset and wavelength, so one universal offset table
   cannot replace these conditional one-dimensional sweeps.
5. Interpolate geometry through the exact `(z, width)` and `(z, offset)` anchors
   using PCHIP. It preserves monotonicity and avoids overshoot. Its first derivative
   is continuous, but curvature can jump at anchors. Optional slope/curvature
   limits are checked at polynomial extrema, including both sides of knots; an
   invalid profile fails rather than silently relaxing the constraints. There is
   no global spline fit and no Powell call. PCHIP itself is piecewise cubic; the
   expensive optimization over clamped cubic control points has been removed.
6. Calculate direct cross-state FDE coefficients at the selected geometry anchors.
   Check predicted versus actual complex coupling there, and check the width
   table against direct isolated detuning at the selected widths. Separately
   evaluate a grid with all z midpoints inserted and compare complex coefficients
   against interpolation on the original z grid. Refine failing sweep axes and/or
   z by `2*N-1`, within the configured limits. The reported objective uses direct
   coefficients, never table predictions.

Rejected offset samples form invalid intervals; they are not replaced by zero or
interpolated across. Bounded tracking recovery can add real midpoint solves. A
path that cannot connect its fixed endpoints fails explicitly. API failures and
endpoint inconsistencies still stop the run. Target amplitudes are never rescaled
to hide insufficient coupling, and imaginary coupling is not discarded.

## Running and validation

From the repository root, with a compatible completed endpoint directory:

```powershell
python tools/run_vlac.py block2 --config config/native_sweep.json --endpoint-dir outputs/endpoint --out outputs/sweep_profile --max-solves 2000 --max-seconds 28800
python tools/run_vlac.py validate-block2 --config config/native_sweep.json --endpoint-dir outputs/endpoint --out outputs/sweep_validation --max-solves 2000 --max-seconds 28800
```

Both commands construct and check the same geometry. `validate-block2` additionally
writes its validation report and returns exit code 3 for a usable review result;
`block2` returns success for review results. Both stop with exit code 2 on an
execution failure or 3 on an exhausted hard budget (incomplete). Use distinct
output directories and choose budgets for the machine
and study; the example budgets do not guarantee completion.

Defaults are 17 width samples, 17 offset samples, at most 65 samples per sweep,
and 8 inexpensive linear-interpolation subdivisions per sampled interval. An
extra aligned offset counts toward the cap. `device.z_samples_initial` sets the
initial anchor count; `device.z_samples_max` must accommodate independent midpoint
checks. Flags `--sweep-width-samples`, `--sweep-offset-samples`,
`--sweep-samples-max` and `--max-refinements` control the work. `--max-evaluations`
is rejected for this method because it has no Powell iterations. Spline control
count and Powell tolerances are unused. `--complete-within-seconds` uses observed
sample costs to defer optional refinement while reserving validation/export time;
the initial sweep and checks are mandatory. `--max-seconds` and `--max-solves`
remain hard execution guards.

Completed width tables and offset rows are checkpointed. A compatible resume
reuses these after endpoint preflight; backend raw caches can reuse partial row
solves. Changing geometry/method settings invalidates incompatible checkpoints.

The numerical checks test table interpolation **at selected anchors** and z
interpolation at independent midpoints, at the **two band edges only**. They do
not validate all points in each table, continuous target optimality, interior
wavelengths, transverse mesh convergence or physical device performance. The
legacy `optimization_converged` summary field means the selected table checks
passed for this method. `grid_converged` also requires the z check. Neither field
imposes a target-fit acceptance threshold: inspect the published objective and
separate width/coupling target errors. A sample/refinement cap yields review.

## Saved geometry and downstream blocks

Block handoff schema 4 stores `profile_kind`, `anchor_z_m`, and anchor values.
The legacy `w1_control_m` and `d_c2c_control_m` payload fields carry anchor values
only when `profile_kind=pchip_anchors_v1`; consumers must honor that discriminator.
Public geometry JSON names these as anchors. Geometry knots remain independent
of coefficient sampling grids, including retained characterization. Blocks 3/4,
exports, sensitivity and diagnostics reconstruct the same anchored geometry.
Schema 1/2 legacy spline handoffs remain readable; experimental schema 3 is not
silently reinterpreted. Standalone native LSF does not implement this method.

`block2_sweep_report.json` records the optical tables, validity masks, chosen
anchors and sample-slot counts. Counts are not fresh FDE solves: caching,
isolated/coupled decomposition and recovery determine actual solver calls.
Existing geometry CSVs and target plots evaluate the saved interpolant exactly.
`config/native_sequential.json` retains the sequential Powell alternative;
`config/native_joint.json` and `config/python_joint.json` retain joint Powell.
Saved/custom configurations keep their explicitly named method.

## Expected runtime effect

Let `W` be width samples, `S` offset samples, `A` distinct selected widths, `Z`
anchors and `L=2` wavelength edges. Before refinement, the width sweep requests
roughly `L*(W+1)` isolated modes, and the coupling tables request `L*A*S`
coefficient samples. Direct checks add approximately `L*Z` coefficient samples,
`L*(Z+1)` isolated modes and `L*(2*Z-1)` coefficient samples for the midpoint
grid. Duplicate grids/widths can be served by caches; endpoint preflight and
retained characterization add work outside these counts.

With `W=S=17`, `A=Z=5`, this is 36 initial isolated requests and 170 table
coefficient requests, followed by 12 isolated and 28 coefficient validation
requests. A sequential Powell run consuming its full shared 120-evaluation
budget on five z samples would request about 600 offset-stage coefficient
samples plus repeated isolated width grids, before final checks. Removing those
repeated whole-profile trials is the expected saving. This is a workload model,
not a measured speedup: cache hits, candidate rejection, extra refinement and
large anchor counts can make sweeps less favorable. MODE eigensolve cost per
fresh sample is unchanged. No licensed timing or physical validation is implied
by the offline synthetic tests.

Sources: [source register REF-002/004/015](../references/source-register.md).
