"""Sequential linear optical sweeps and exact PCHIP geometry anchors.

No whole-profile optimizer or synthetic optical fallback is used. Table
interpolation guides construction; published coefficients are direct FDE data.
"""
from dataclasses import asdict
import time
import numpy as np

from .geometry import make_geometry, AnchoredProfile
from .profiles import ProfileSolution, evaluate_design_profiles, minimax_objective, target_profiles
from .sequential import isolated_detuning_grid
from .tracking import CandidateRejected, require_candidate
from .optimize import _axis_validation_errors, _check_fabrication_constraints
from .checkpoints import Checkpoint
from .provenance import checkpoint_identity
from .block_io import endpoint_data
from .profile_planning import ProfilePlanner

METHOD = 'sweep_anchored_v1'


def signed_offset_grid(start, stop, count):
    """Linear sweep with an extra aligned sample when it lies between endpoints."""
    values = np.linspace(start, stop, count)
    if values.min() <= 0 <= values.max():
        values = np.unique(np.r_[values, 0.])
    return values[::-1] if start > stop else values


def interpolate_samples(x, values, query):
    """Complex linear interpolation; invalid samples split valid intervals."""
    order = np.argsort(x)
    x, values = np.asarray(x)[order], np.asarray(values)[:,order]
    return np.asarray([np.interp(query,x,row.real,left=np.nan,right=np.nan)
                       +1j*np.interp(query,x,row.imag,left=np.nan,right=np.nan) for row in values])


def monotone_path(cost, candidates, z, slope_limit=None):
    """Finite-grid minimax path with exact endpoints and ordered coordinates.

    A deterministic secondary sum of residuals then squared steps resolves ties.
    The secant slope test is necessary; exact PCHIP derivatives are checked later.
    """
    cost, candidates, z = np.asarray(cost), np.asarray(candidates), np.asarray(z)
    n, m = cost.shape
    best = np.full(m,np.inf); sums = best.copy(); motion = best.copy()
    best[0], sums[0], motion[0] = cost[0,0], cost[0,0], 0.
    parents = np.full((n,m),-1,int)
    span = max(abs(candidates[-1]-candidates[0]),1e-30)
    for i in range(1,n):
        new = np.full(m,np.inf); total=new.copy(); moves=new.copy()
        choices = [m-1] if i==n-1 else range(m)
        for j in choices:
            if not np.isfinite(cost[i,j]):
                continue
            previous = np.arange(j+1)
            valid = np.isfinite(best[previous])
            if slope_limit is not None:
                valid &= abs(candidates[j]-candidates[previous]) <= slope_limit*(z[i]-z[i-1])
            previous=previous[valid]
            if not len(previous):
                continue
            worst=np.maximum(best[previous],cost[i,j])
            summed=sums[previous]+cost[i,j]
            movement=motion[previous]+((candidates[j]-candidates[previous])/span)**2
            selected=np.lexsort((previous,movement,summed,worst))[0]
            k=previous[selected]
            new[j],total[j],moves[j]=worst[selected],summed[selected],movement[selected]
            parents[i,j]=k
        best,sums,motion=new,total,moves
    if not np.isfinite(best[-1]):
        raise ValueError('No valid monotone sweep path connects the fixed endpoints.')
    indices=np.empty(n,int);indices[-1]=m-1
    for i in range(n-1,0,-1):
        indices[i-1]=parents[i,indices[i]]
    return indices


def coupling_row(backend, config, bank, endpoint, width, offsets, wavelengths):
    """Each row has its own repeatable signed-offset and wavelength tracking chain."""
    result=np.full((len(wavelengths),len(offsets)),np.nan+1j*np.nan)
    previous_lambda=[None]*len(offsets)
    rejected=0
    for iw,wavelength in enumerate(wavelengths):
        previous_ref=None; previous_kappa=None; previous_offset=None
        current=[]
        def sample(offset,last_offset,ref,kappa,depth,lambda_ref=None):
            geometry=make_geometry(config.geometry,w1_m=float(width),h_rib_m=endpoint.h_rib_m,
                h_gap_m=endpoint.h_gap_m,d_c2c_m=float(offset),state='cross',
                liquid_index=float(bank.liquid_cross.index(wavelength)))
            try:
                value,reference=backend.evaluate_coefficients(geometry,float(wavelength),
                    references=tuple(r for r in (ref,lambda_ref) if r is not None),previous_kappa=kappa)
                require_candidate(value,config)
                if not reference.valid:
                    raise CandidateRejected('Sweep returned a rejected reference.')
                return value,reference
            except CandidateRejected:
                if ref is None or depth<=0 or offset==last_offset:
                    raise
                middle=(last_offset+offset)/2
                value,reference=sample(middle,last_offset,ref,kappa,depth-1)
                return sample(offset,middle,reference,value.kappa,depth-1,lambda_ref)
        for j,offset in enumerate(offsets):
            try:
                value,reference=sample(offset,previous_offset,previous_ref,previous_kappa,
                                       config.solver.tracking_recovery_depth,previous_lambda[j])
            except CandidateRejected:
                rejected+=1
                current.append(None)
                # A hole splits interpolation AND tracking. No rejected reference propagates.
                previous_ref=previous_kappa=previous_offset=None
                continue
            result[iw,j]=value.kappa
            current.append(reference)
            previous_ref,previous_kappa,previous_offset=reference,value.kappa,offset
        previous_lambda=current
    return result,rejected


def optimize_sweep(backend,config,bank,endpoint,*,progress=None,checkpoint_dir=None):
    if progress: progress.start_stage('Sweep-anchored profile construction',unit='FDE sweep samples')
    backend.require_endpoint_consistency(endpoint,progress=progress)
    o,d,g=config.optimizer,config.device,config.geometry
    waves=np.array([d.wavelength_min_m,d.wavelength_max_m])
    settings=asdict(o)
    for key in ('max_grid_refinements','max_function_evaluations_per_grid','complete_within_seconds'):
        settings.pop(key)
    checkpoint=(Checkpoint(checkpoint_dir,'sweep_anchors',checkpoint_identity(config,2,
        {'endpoint':endpoint_data(endpoint),'settings':settings,'initial_z':d.z_samples_initial,
         'max_z':d.z_samples_max})) if checkpoint_dir is not None else None)
    state=checkpoint.load() if checkpoint is not None and config.execution.resume else None
    if state is None:
        state={'width_tables':{},'rows':{},'history':[],'grid_history':[], 'refinement':0,
               'z':np.linspace(0,d.length_m,d.z_samples_initial),
               'nw':o.sweep_width_samples,'nd':o.sweep_offset_samples}
    if state.get('solution') is not None:
        if progress: progress.finish_stage('Reused checked sweep checkpoint after endpoint preflight.')
        return state['solution']
    def save():
        if checkpoint is not None: checkpoint.save(state)
    planner=ProfilePlanner(o.complete_within_seconds,started=getattr(progress,'started',None))
    def note(text):
        if progress: progress.note(text,force=True)
    def timed(fn,count):
        started=time.monotonic()
        value=fn()
        planner.observe(time.monotonic()-started,count)
        return value
    for refinement in range(state['refinement'],max(state['refinement'],o.max_grid_refinements)+1):
        z=np.asarray(state['z']); nw,nd=state['nw'],state['nd']
        widths=np.linspace(endpoint.w1_input_m,endpoint.w1_output_m,nw)
        key=tuple(widths)
        if key not in state['width_tables']:
            note(f'Sweep width: {nw} widths at the two band edges; isolated modes only.')
            state['width_tables'][key]=timed(lambda: isolated_detuning_grid(backend,config,bank,endpoint,
                None,np.linspace(0,d.length_m,nw),waves,width_samples=widths),nw*len(waves))
            save()
        delta=state['width_tables'][key]
        width_candidates=np.linspace(widths[0],widths[-1],(nw-1)*o.sweep_interpolation_subdivisions+1)
        predicted=interpolate_samples(widths,delta,width_candidates).real
        target_delta,target_kappa=target_profiles(z,d.length_m,endpoint.delta_beta_max_rad_per_m)
        scale=endpoint.delta_beta_max_rad_per_m
        costs=np.max(abs(predicted[:,None,:]-target_delta[None,:,None]),axis=0)/scale
        wi=monotone_path(costs,width_candidates,z,o.maximum_slope_m_per_m)
        width_anchors=width_candidates[wi]
        # Keep a linear signed sweep including both fixed endpoints; zero is added explicitly.
        offsets=signed_offset_grid(g.d_c2c_input_m,g.d_c2c_output_m,nd)
        if len(offsets)>o.sweep_samples_max:
            raise ValueError('Offset sweep including the aligned sample exceeds sweep_samples_max.')
        offset_candidates=np.unique(np.r_[np.linspace(offsets.min(),offsets.max(),
            (len(offsets)-1)*o.sweep_interpolation_subdivisions+1),offsets])
        if offsets[0]>offsets[-1]: offset_candidates=offset_candidates[::-1]
        rows=[]; rejected=0
        for j,width in enumerate(width_anchors):
            key=(float(width),tuple(offsets))
            if key not in state['rows']:
                note(f'Sweep offset: anchor {j+1}/{len(z)}, width={width*1e6:.6g} um, {len(offsets)} signed offsets.')
                state['rows'][key]=timed(lambda: coupling_row(backend,config,bank,endpoint,width,offsets,waves),
                                        len(offsets)*len(waves))
                save()
            row,bad=state['rows'][key];rows.append(row);rejected+=bad
        interpolated=np.array([interpolate_samples(offsets,row,offset_candidates) for row in rows])
        costs=np.max(abs(interpolated-target_kappa[:,None,None]),axis=1)/(scale/2)
        costs[~np.isfinite(costs)]=np.inf
        di=monotone_path(costs,offset_candidates,z,o.maximum_slope_m_per_m)
        offset_anchors=offset_candidates[di]
        wp,dp=(AnchoredProfile(d.length_m,z,v) for v in (width_anchors,offset_anchors))
        if _check_fabrication_constraints(config,wp,dp):
            raise ValueError('Sweep anchor interpolation violates configured slope/curvature limits.')
        note('Checking selected anchors and longitudinal midpoints with direct FDE coefficients.')
        def evaluate(zz):
            return evaluate_design_profiles(backend,config,bank,endpoint,width_anchors,offset_anchors,
                zz,waves,state='cross',anchor_z_m=z)
        actual=timed(lambda:evaluate(z),len(z)*len(waves))
        value,dt,kt=minimax_objective(actual,z,d.length_m,scale)
        selected_kappa=np.array([interpolated[j,:,di[j]] for j in range(len(z))]).T
        actual_kappa=np.array([[c.kappa for c in row] for row in actual])
        coupling_error=float(np.max(abs(actual_kappa-selected_kappa))/(scale/2))
        # Direct isolated samples distinguish width-table error from translated-domain detuning.
        isolated=timed(lambda:isolated_detuning_grid(backend,config,bank,endpoint,None,z,waves,
                                                    width_samples=width_anchors),len(z)*len(waves))
        width_error=float(np.max(abs(isolated-interpolate_samples(widths,delta,width_anchors)))/scale)
        zz=np.sort(np.r_[z,(z[:-1]+z[1:])/2])
        z_error=None
        if len(zz)<=d.z_samples_max:
            checked=timed(lambda:evaluate(zz),len(zz)*len(waves))
            z_error=_axis_validation_errors(actual,checked,z,zz,1,scale)
        table_ok=max(width_error,coupling_error)<=d.grid_convergence_rtol
        converged=bool(table_ok and z_error is not None and z_error<=d.grid_convergence_rtol)
        state['history'].append({'method':METHOD,'grid_refinement':refinement,'n_z':len(z),
            'n_wavelengths':len(waves),'width_sweep_samples':nw,'offset_sweep_samples':len(offsets),
            'evaluations':0,'success':bool(table_ok),'message':'Finite sweep-path selection; no Powell fit.',
            'objective':value,'rejected_offset_samples':rejected,'width_frozen_during_offset_fit':True,
            'target_fit_acceptance_threshold':None,'width_table_error':width_error,
            'coupling_table_error':coupling_error,'table_validation_scope':'selected anchors only',
            'width_target_error':float(np.max(abs(isolated.real-dt[None,:]))/scale),
            'offset_target_error':float(np.max(abs(actual_kappa-kt[None,:]))/(scale/2))})
        record={'grid_refinement':refinement,'n_z':len(z),'n_wavelengths':2,
            'z_coefficient_interpolation_error':z_error,'wavelength_coefficient_interpolation_error':None,
            'wavelength_check':'band_edges_only_no_interior_check','grid_converged':converged,
            'width_table_error':width_error,'coupling_table_error':coupling_error,
            'tolerance':d.grid_convergence_rtol}
        state['grid_history'].append(record)
        sweep_report={'method':METHOD,'units':'SI','wavelengths_m':waves.tolist(),
            'width_sweep_m':widths.tolist(),'isolated_delta_real_rad_per_m':delta.real.tolist(),
            'isolated_delta_imag_rad_per_m':delta.imag.tolist(),'anchor_z_m':z.tolist(),
            'selected_width_m':width_anchors.tolist(),'offset_sweep_m':offsets.tolist(),
            'selected_offset_m':offset_anchors.tolist(),
            'kappa_real_rad_per_m':np.asarray(rows).real.tolist(),
            'kappa_imag_rad_per_m':np.asarray(rows).imag.tolist(),
            'valid_samples':np.isfinite(np.asarray(rows)).tolist(),
            'coupling_array_order':'anchor, wavelength, signed offset',
            'sample_slot_counts':{'isolated_width':int(delta.size),'coupling_table':int(np.asarray(rows).size)},
            'counts_are_not_fresh_fde_solves':True,'table_validation_scope':'selected anchors only',
            'target_amplitudes_changed':False,'optimizer_called':False}
        solution=ProfileSolution(width_anchors,offset_anchors,z,waves,dt,kt,actual,value,
            list(state['history']),list(state['grid_history']),converged,'pchip_anchors_v1',z.copy(),sweep_report)
        if progress: progress.checkpoint(f'sweep_grid_{refinement}',record)
        if converged:
            record['stop_reason']='anchor_tables_and_z_checks_passed'
            state['solution']=solution;save()
            if progress: progress.finish_stage(f'J={value:.6g}; sweep and z checks passed')
            return solution
        next_nw=2*nw-1 if width_error>d.grid_convergence_rtol else nw
        next_nd=2*nd-1 if coupling_error>d.grid_convergence_rtol else nd
        next_z=zz if z_error is not None and z_error>d.grid_convergence_rtol else z
        next_offsets=signed_offset_grid(g.d_c2c_input_m,g.d_c2c_output_m,next_nd)
        next_cost=(next_nw+len(next_z)*len(next_offsets))*len(waves)
        if (refinement>=o.max_grid_refinements or next_nw>o.sweep_samples_max or len(next_offsets)>o.sweep_samples_max
                or np.array_equal(next_z,z) and next_nw==nw and next_nd==nd
                or 2*len(next_z)-1>d.z_samples_max):
            record['stop_reason']='refinement_budget_or_sample_limit';save()
            if progress: progress.finish_stage(f'J={value:.6g}; review required at sample/refinement limit')
            return solution
        if not planner.allows(next_cost,(2*len(next_z)-1)*len(waves)):
            record['stop_reason']='completion_plan_refinement_deferred';save()
            if progress: progress.finish_stage(f'J={value:.6g}; refinement deferred by time plan')
            return solution
        state.update(refinement=refinement+1,z=next_z,nw=next_nw,nd=next_nd)
        save()
    return solution
