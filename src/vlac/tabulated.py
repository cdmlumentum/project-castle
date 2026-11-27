"""tabulated_width_offset_v1: tracked FDE tables, cheap scalar inversion/fits.

Optical fields and invalid branches are never interpolated. The lookup models
are proposals; the published coefficients always come from direct FDE samples.
"""
from dataclasses import asdict
import time
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import Bounds, minimize, minimize_scalar

from .geometry import ClampedCubicProfile, make_geometry, blackman_theta
from .monotone import controls, linear_coordinates
from .modes import ModeReference
from .profiles import ProfileSolution, target_profiles, minimax_objective, evaluate_design_profiles
from .profile_planning import ProfilePlanner
from .tracking import CandidateRejected, require_candidate, require_anchor_match

METHOD = 'tabulated_width_offset_v1'


class TableAllowanceFinished(Exception):
    """Retain measured tables and reserve final direct validation/export."""


def signed_offset_nodes(start,end,count):
    """Include zero and both signed endpoints within the exact node allowance."""
    low,high = sorted((start,end))
    return np.r_[np.linspace(low,0.,count//2+1),np.linspace(0.,high,count-count//2)[1:]]


def pchip(x, y, query, axis=0):
    """Complex, non-extrapolating, shape-preserving interpolation."""
    x, y = np.asarray(x), np.asarray(y)
    query = np.asarray(query)
    if np.any(query<x[0]) or np.any(query>x[-1]):
        raise ValueError('Table query outside measured bounds; extrapolation is forbidden.')
    return (PchipInterpolator(x,y.real,axis=axis,extrapolate=False)(query)
            +1j*PchipInterpolator(x,y.imag,axis=axis,extrapolate=False)(query))


def scalar_best(function, low, high):
    """Resolve branches on a cheap dense grid before local scalar polishing."""
    grid = np.linspace(low,high,129)
    values = np.array([function(x) for x in grid])
    best = int(np.argmin(values))
    if best in (0,len(grid)-1):
        return float(grid[best])
    result = minimize_scalar(lambda u:function(low+u*(high-low)),bounds=(
        (grid[best-1]-low)/(high-low),(grid[best+1]-low)/(high-low)),
        method='bounded',options={'xatol':1e-10})
    candidate = low+result.x*(high-low)
    return float(candidate if result.fun<values[best] else grid[best])


def select_coupling_scale(delta_max, zero_coupling, sine):
    """One common amplitude, limited by both wavelengths along the profile.

    This is sampled feasibility, not a proof that a fixed-endpoint monotone
    offset spline can exactly realize the entire coupling target.
    """
    requested = .5*delta_max
    sine = np.asarray(sine)
    active = sine>1e-8
    capacity = np.asarray(zero_coupling).real
    if not np.all(np.isfinite(capacity)) or not np.any(active):
        raise ValueError('Missing finite zero-offset coupling envelope.')
    limit = float(np.min(capacity[:,active]/sine[active]))
    chosen = min(requested,max(0.,limit))
    if chosen<=0:
        raise CandidateRejected('No positive common coupling amplitude on the validated branch.')
    return {'policy':'common_zero_offset_envelope_v1','requested_kappa_max_rad_per_m':requested,
        'kappa_max_rad_per_m':chosen,'sampled_envelope_limit_rad_per_m':limit,
        'coupling_target_reduced':bool(chosen<requested),
        'original_amplitude_relationship_holds':bool(chosen==requested),
        'delta_beta_max_over_kappa_max':delta_max/chosen,
        'kappa_max_over_delta_beta_max':chosen/delta_max,
        'half_delta_beta_max_over_kappa_max':.5*delta_max/chosen,
        'nominal_delta_beta_max_over_kappa_max':2.,
        'note':'Feasibility estimate at band edges; reduced modal gap requires propagation/adiabatic validation.'}


class OpticalTables:
    def __init__(self, backend, config, bank, endpoint, progress, checkpoint_dir=None):
        self.backend,self.config,self.bank,self.endpoint,self.progress = backend,config,bank,endpoint,progress
        self.waves = np.array([config.device.wavelength_min_m,config.device.wavelength_max_m])
        self.delta_cache,self.coupling_cache = {},{}
        self.lower,self.upper,self.references = {},{},{}
        self.events = []
        self.refinement_work = False
        self.planner = ProfilePlanner(config.optimizer.complete_within_seconds,started=getattr(progress,'started',None))
        self.checked_z = min(config.device.z_samples_max,2*config.device.z_samples_initial-1)
        self.reserve = 2*len(self.waves)*self.checked_z  # base + midpoint profile packets
        self.path = None if progress is None else progress.output_dir/'block2_tables.json'
        self.checkpoint = None
        if checkpoint_dir is not None:
            from .checkpoints import Checkpoint
            from .provenance import checkpoint_identity
            self.checkpoint = Checkpoint(checkpoint_dir,'optical_tables',checkpoint_identity(config,2,asdict(endpoint)))
            saved = self.checkpoint.load() if config.execution.resume else None
            if saved:
                for key in ('delta_cache','coupling_cache','lower','upper','references','events'):
                    setattr(self,key,saved[key])

    def geometry(self,width,offset,wavelength):
        return make_geometry(self.config.geometry,w1_m=float(width),h_rib_m=self.endpoint.h_rib_m,
            h_gap_m=self.endpoint.h_gap_m,d_c2c_m=float(offset),state='cross',
            liquid_index=float(self.bank.liquid_cross.index(wavelength)))

    def save(self,*,checkpoint=False):
        if self.checkpoint and checkpoint:
            self.checkpoint.save({key:getattr(self,key) for key in
                ('delta_cache','coupling_cache','lower','upper','references','events')})
        if self.path:
            from .block_io import write_json
            write_json(self.path,{'method':METHOD,'wavelengths_m':self.waves,'events':self.events,
                'detuning_samples':[{'width_m':w,'wavelength_m':l,'delta_beta':v}
                    for (w,l),v in sorted(self.delta_cache.items())],
                'coupling_samples':[{'width_m':w,'offset_m':d,'wavelength_m':l,'coefficients':asdict(v)}
                    for (w,d,l),v in sorted(self.coupling_cache.items())],
                'tracking_fields':'retained only in compatible checkpoint/raw caches',
                'physical_device_validated':False})

    def optional_work_allowed(self):
        budget = getattr(self.backend,'budget',None)
        if budget is not None and budget.settings.max_solves is not None:
            if budget.solves+3*(self.reserve+1)>budget.settings.max_solves: return False
        return self.planner.allows(1,self.reserve)

    def measured(self,call):
        if self.refinement_work and not self.optional_work_allowed():
            raise TableAllowanceFinished()
        if hasattr(self.backend,'_budget'): self.backend._budget().check()
        started = time.monotonic()
        try: return call()
        finally: self.planner.observe(time.monotonic()-started,1)

    def delta(self,width,wavelength,depth=None):
        key = (float(width),float(wavelength))
        if key in self.delta_cache: return self.delta_cache[key]
        cfg = self.config
        depth = cfg.solver.tracking_recovery_depth if depth is None else depth
        geom = self.geometry(width,0.,wavelength)
        def solve(guide,reference):
            mode,overlap = self.measured(lambda:self.backend.solve_isolated_mode(geom,float(wavelength),
                guide=guide,reference=reference,compact_x=False))
            if (not np.isfinite(mode.neff) or not np.isfinite(overlap) or overlap<cfg.solver.minimum_mode_overlap
                    or mode.te_fraction<cfg.solver.minimum_te_fraction or not np.all(np.isfinite(mode.ex))
                    or not np.all(np.isfinite(mode.ey)) or np.sum(abs(mode.ex)**2+abs(mode.ey)**2)<=0):
                raise CandidateRejected('Invalid isolated mode in width table.')
            return mode
        if wavelength not in self.lower: self.lower[wavelength] = solve('lower',None)
        candidates = [(abs(w-width),w,mode) for (w,l),mode in self.upper.items() if l==wavelength]
        _,prior_width,ref = min(candidates,key=lambda v:v[0]) if candidates else (0,width,None)
        try:
            upper = solve('upper',ref)
        except CandidateRejected:
            if ref is None or depth<=0 or prior_width==width: raise
            self.delta(.5*(prior_width+width),wavelength,depth-1)
            return self.delta(width,wavelength,depth-1)
        for offset in (cfg.geometry.d_c2c_input_m,cfg.geometry.d_c2c_output_m):
            anchor = getattr(self.backend,'_endpoint_full_modes',{}).get((float(wavelength),float(width),offset,geom.liquid_index))
            if anchor:
                require_anchor_match(upper,anchor[0],cfg)
                require_anchor_match(self.lower[wavelength],anchor[1],cfg)
        value = 2*np.pi*(upper.neff-self.lower[wavelength].neff)/wavelength
        self.upper[key],self.delta_cache[key] = upper,value
        if len(self.upper)>16: self.upper.pop(next(iter(self.upper)))
        self.save()
        return value

    def coupling(self,width,offset,wavelength,depth=None):
        key = tuple(float(v) for v in (width,offset,wavelength))
        if key in self.coupling_cache: return self.coupling_cache[key].kappa
        depth = self.config.solver.tracking_recovery_depth if depth is None else depth
        # Follow a single signed offset branch. No +/- symmetry is assumed.
        candidates = [(abs(d-offset)+abs(w-width),w,d,ref,self.coupling_cache[(w,d,l)].kappa)
            for (w,d,l),ref in self.references.items() if l==wavelength]
        _,prior_width,prior_offset,ref,previous = min(candidates,key=lambda v:v[0]) if candidates else (0,width,offset,None,None)
        try:
            value,reference = self.measured(lambda:self.backend.evaluate_coefficients(self.geometry(width,offset,wavelength),
                float(wavelength),references=() if ref is None else (ref,),previous_kappa=previous))
            require_candidate(value,self.config)
            if not reference.valid: raise CandidateRejected('Rejected reference in coupling table.')
        except CandidateRejected:
            if ref is None or depth<=0 or (prior_width==width and prior_offset==offset): raise
            self.coupling(.5*(prior_width+width),.5*(prior_offset+offset),wavelength,depth-1)
            return self.coupling(width,offset,wavelength,depth-1)
        # Failed points never enter the interpolator or future tracking chain.
        self.coupling_cache[key],self.references[key] = value,reference
        if len(self.references)>8: self.references.pop(next(iter(self.references)))
        self.save()
        return value.kappa

    def tolerance(self,actual,predicted):
        c = self.config.convergence
        bound = np.minimum(c.coefficient_atol_rad_per_m+c.coefficient_rtol*np.maximum(abs(actual),abs(predicted)),
                           c.phase_error_rad/self.config.device.length_m)
        return np.abs(actual-predicted)<=bound

    def width_table(self):
        o = self.config.optimizer
        nodes = np.linspace(min(self.endpoint.w1_input_m,self.endpoint.w1_output_m),
                            max(self.endpoint.w1_input_m,self.endpoint.w1_output_m),o.table_initial_widths)
        converged = False
        while True:
            values = np.array([[self.delta(w,l) for w in nodes] for l in self.waves])
            if any(not (np.all(np.diff(row.real)>0) or np.all(np.diff(row.real)<0)) for row in values):
                raise CandidateRejected('Detuning table is not a single monotone branch; inspect mode identities.')
            middle = .5*(nodes[:-1]+nodes[1:])
            if not self.optional_work_allowed(): break
            self.refinement_work = True
            try:
                actual = np.array([[self.delta(w,l) for w in middle] for l in self.waves])
            except TableAllowanceFinished:
                break
            finally: self.refinement_work = False
            predicted = pchip(nodes,values,middle,axis=1)
            bad = np.any(~self.tolerance(actual,predicted),axis=0)
            converged = not bool(np.any(bad))
            self.events.append({'stage':'width_table','nodes':len(nodes),'midpoints':len(middle),
                'max_complex_error_rad_per_m':float(np.max(abs(actual-predicted))),'converged':converged})
            self.save(checkpoint=True)
            if converged or len(nodes)+sum(bad)>o.table_max_widths: break
            nodes = np.sort(np.r_[nodes,middle[bad]])
        self.save(checkpoint=True)
        return nodes,values,converged

    def coupling_table(self,width_nodes):
        o,g = self.config.optimizer,self.config.geometry
        widths = np.linspace(width_nodes[0],width_nodes[-1],5)
        offsets = signed_offset_nodes(g.d_c2c_input_m,g.d_c2c_output_m,o.table_initial_offsets)
        converged = False
        while True:
            values = np.array([[[self.coupling(w,d,l) for d in offsets] for w in widths] for l in self.waves])
            if not self.optional_work_allowed(): break
            wm,dm = .5*(widths[:-1]+widths[1:]),.5*(offsets[:-1]+offsets[1:])
            # Independent checks in both axes, including cell centers. A failed
            # interpolation check refines the associated interval, never its optics.
            self.refinement_work = True
            try:
                offset_actual = np.array([[[self.coupling(w,d,l) for d in dm] for w in widths] for l in self.waves])
                width_actual = np.array([[[self.coupling(w,d,l) for d in offsets] for w in wm] for l in self.waves])
                centers = np.array([[[self.coupling(w,d,l) for d in dm] for w in wm] for l in self.waves])
            except TableAllowanceFinished:
                break
            finally: self.refinement_work = False
            offset_predicted = pchip(offsets,values,dm,axis=2)
            width_predicted = pchip(widths,values,wm,axis=1)
            center_predicted = pchip(offsets,pchip(widths,values,wm,axis=1),dm,axis=2)
            bad_d = np.any(~self.tolerance(offset_actual,offset_predicted),axis=(0,1))
            bad_w = np.any(~self.tolerance(width_actual,width_predicted),axis=(0,2))
            center_bad = ~self.tolerance(centers,center_predicted)
            bad_d |= np.any(center_bad,axis=(0,1)); bad_w |= np.any(center_bad,axis=(0,2))
            converged = not bool(np.any(bad_d) or np.any(bad_w))
            self.events.append({'stage':'coupling_table','width_nodes':len(widths),'offset_nodes':len(offsets),
                'max_complex_error_rad_per_m':float(max(np.max(abs(offset_actual-offset_predicted)),
                    np.max(abs(width_actual-width_predicted)),np.max(abs(centers-center_predicted)))),
                'converged':converged})
            self.save(checkpoint=True)
            if (converged or len(widths)+sum(bad_w)>o.table_max_widths
                    or len(offsets)+sum(bad_d)>o.table_max_offsets): break
            # All added rows/cells were independently measured above.
            widths,offsets = np.sort(np.r_[widths,wm[bad_w]]),np.sort(np.r_[offsets,dm[bad_d]])
        self.save(checkpoint=True)
        return widths,offsets,values,converged


def fit_spline(config,start,end,objective):
    """Cheap monotone cubic fit; no optical calls inside Powell."""
    from .optimize import _check_fabrication_constraints
    best = [float('inf'),None]
    def value(unit):
        c = controls(unit,start,end)
        profile = ClampedCubicProfile(config.device.length_m,c)
        penalty = _check_fabrication_constraints(config,profile,profile)
        result = float(penalty or objective(profile))
        if not penalty and result<best[0]: best[:] = [result,np.asarray(c)]
        return result
    initial = linear_coordinates(config.optimizer.control_points)
    result = minimize(value,initial,method='Powell',bounds=Bounds(np.zeros_like(initial),np.ones_like(initial)),
        options={'maxfev':config.optimizer.max_function_evaluations_per_grid,
                 'xtol':config.optimizer.minimum_control_step_fraction,'ftol':config.device.grid_convergence_rtol})
    if best[1] is None: raise ValueError('No spline satisfies the configured fabrication constraints.')
    return best[1],{'success':bool(result.success),'message':str(result.message),
        'evaluations':int(result.nfev),'surrogate_objective':best[0],'fde_solves_inside_optimizer':0}


def optimize_tabulated(backend,config,bank,endpoint,*,progress=None,checkpoint_dir=None):
    from .optimize import _axis_validation_errors, _check_fabrication_constraints
    backend.require_endpoint_consistency(endpoint,progress=progress)
    tables = OpticalTables(backend,config,bank,endpoint,progress,checkpoint_dir)
    d,g,o = config.device,config.geometry,config.optimizer
    wn,delta,width_ok = tables.width_table()
    dense = np.linspace(0,d.length_m,129)
    delta_target = target_profiles(dense,d.length_m,endpoint.delta_beta_max_rad_per_m)[0]
    D = endpoint.delta_beta_max_rad_per_m
    def predict_delta(width): return pchip(wn,delta,width,axis=1)
    desired_width = np.array([scalar_best(lambda w:float(np.max(abs(predict_delta(w).real-target))),wn[0],wn[-1])
                              for target in delta_target])
    wc,width_fit = fit_spline(config,endpoint.w1_input_m,endpoint.w1_output_m,
        lambda p:np.max(abs(predict_delta(p(dense)).real-delta_target))/D)
    wp = ClampedCubicProfile(d.length_m,wc)
    cw,cd,kappa,coupling_ok = tables.coupling_table(wn)
    def predict_kappa(width,offset):
        at_width = pchip(cw,kappa,float(width),axis=1)
        return pchip(cd,at_width,float(offset),axis=1)
    envelope_z = np.linspace(0,d.length_m,501)
    zero = np.array([predict_kappa(w,0.) for w in wp(envelope_z)]).T
    metadata = select_coupling_scale(D,zero,np.sin(blackman_theta(envelope_z,d.length_m)))
    metadata.update(method=METHOD,envelope_z_m=envelope_z.tolist(),
        zero_offset_real_coupling_rad_per_m=zero.real.tolist(),
        detuning_table_converged=width_ok,coupling_table_converged=coupling_ok,
        width_scalar_inverse_z_m=dense.tolist(),width_scalar_inverse_m=desired_width.tolist(),
        unreachable_detuning_samples_by_wavelength=[int(np.sum((delta_target<min(row.real))|(delta_target>max(row.real)))) for row in delta])
    K = metadata['kappa_max_rad_per_m']
    target_k = target_profiles(dense,d.length_m,D,K)[1]
    # Choose the signed approach/departure branch, without assuming symmetry.
    desired_offset = []
    for z,target in zip(dense,target_k):
        bounds = sorted((g.d_c2c_input_m,0.)) if z<d.length_m/2 else sorted((0.,g.d_c2c_output_m))
        desired_offset.append(scalar_best(lambda offset:float(np.max(abs(predict_kappa(float(wp(z)),offset)-target))),*bounds))
    fixed_widths = wp(dense)
    def offset_error(profile):
        actual = np.array([predict_kappa(w,offset) for w,offset in zip(fixed_widths,profile(dense))]).T
        return np.max(abs(actual-target_k))/K
    dc,offset_fit = fit_spline(config,g.d_c2c_input_m,g.d_c2c_output_m,offset_error)
    metadata['offset_scalar_inverse_m'] = desired_offset
    if _check_fabrication_constraints(config,wp,ClampedCubicProfile(d.length_m,dc)):
        raise ValueError('Table fit violates fabrication constraints.')
    if progress:
        progress.checkpoint('tabulated_target_scaling',metadata)
        if metadata['coupling_target_reduced']:
            progress.note(f'Coupling target reduced to {K:.8g} rad/m; delta_beta_max/kappa_max={D/K:.8g}. Original ratio 2 no longer holds.',force=True)
    # Direct FDE on the final geometry; independently check z midpoints using
    # the original coefficient tolerances and full complex residuals.
    z = np.linspace(0,d.length_m,d.z_samples_initial)
    history,grid_history = [],[]
    grid_ok = False
    for refinement in range(o.max_grid_refinements+1):
        actual = evaluate_design_profiles(backend,config,bank,endpoint,wc,dc,z,tables.waves,state='cross')
        value,dt,kt = minimax_objective(actual,z,d.length_m,D,K)
        refined = np.linspace(0,d.length_m,2*len(z)-1)
        error,checked = None,None
        if len(refined)<=d.z_samples_max:
            checked = evaluate_design_profiles(backend,config,bank,endpoint,wc,dc,refined,tables.waves,state='cross')
            error = _axis_validation_errors(actual,checked,z,refined,1,D,K)
            # Also use the reduced coupling amplitude in the grid check.
            predicted = pchip(z,np.array([[c.kappa for c in row] for row in actual]),refined,axis=1)
            check_k = np.array([[c.kappa for c in row] for row in checked])
            error = max(error,float(np.max(abs(check_k-predicted))/K))
            grid_ok = error<=d.grid_convergence_rtol
        success = bool(width_fit['success'] and offset_fit['success'] and width_ok and coupling_ok)
        validation_z,validation_actual = (refined,checked) if checked is not None else (z,actual)
        realized_width = wp(validation_z)
        realized_offset = ClampedCubicProfile(d.length_m,dc)(validation_z)
        predicted_delta = predict_delta(realized_width)
        predicted_coupling = np.array([predict_kappa(w,offset) for w,offset in zip(realized_width,realized_offset)]).T
        direct_delta = np.array([[c.delta_beta for c in row] for row in validation_actual])
        direct_coupling = np.array([[c.kappa for c in row] for row in validation_actual])
        direct_table_ok = bool(np.all(tables.tolerance(direct_delta,predicted_delta)) and
                               np.all(tables.tolerance(direct_coupling,predicted_coupling)))
        success = success and direct_table_ok
        metadata.update(direct_fde_table_prediction_passed=direct_table_ok,
            direct_detuning_table_max_error_rad_per_m=float(np.max(abs(direct_delta-predicted_delta))),
            direct_complex_kappa_table_max_error_rad_per_m=float(np.max(abs(direct_coupling-predicted_coupling))))
        history.append({'method':METHOD,'success':success,'objective':value,
            'width_fit':width_fit,'offset_fit':offset_fit,'width_frozen_during_offset_fit':True,
            'table_converged':bool(width_ok and coupling_ok),'target_scaling':metadata,
            'evaluations':width_fit['evaluations']+offset_fit['evaluations']})
        grid_history.append({'grid_refinement':refinement,'n_z':len(z),'n_wavelengths':2,
            'z_coefficient_interpolation_error':error,'grid_converged':grid_ok,
            'wavelength_coefficient_interpolation_error':None,'wavelength_check':'band_edges_only_no_interior_check',
            'tolerance':d.grid_convergence_rtol})
        if grid_ok or checked is None or not tables.optional_work_allowed(): break
        if refinement<o.max_grid_refinements: z = refined
    # Include midpoint residuals in exports/objective, not only the coarse grid.
    if checked is not None:
        z,actual = refined,checked
        value,dt,kt = minimax_objective(actual,z,d.length_m,D,K)
        history[-1]['objective'] = value
    achieved = {str(float(l)):{'max_abs_delta_beta_rad_per_m':float(max(abs(c.delta_beta.real) for c in row)),
        'max_abs_complex_kappa_rad_per_m':float(max(abs(c.kappa) for c in row))} for l,row in zip(tables.waves,actual)}
    for row in achieved.values():
        row['delta_beta_peak_over_kappa_peak'] = (row['max_abs_delta_beta_rad_per_m']/row['max_abs_complex_kappa_rad_per_m']
                                                 if row['max_abs_complex_kappa_rad_per_m']>0 else None)
    metadata['achieved_sampled_amplitudes_by_wavelength'] = achieved
    tables.save(checkpoint=True)
    return ProfileSolution(wc,dc,z,tables.waves,dt,kt,actual,value,history,grid_history,grid_ok,metadata)
