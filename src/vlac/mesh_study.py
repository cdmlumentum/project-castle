"""Endpoint-wavelength, shared-grid FDE study; never a Block 0 certificate.

Named alternative: endpoint_shared_grid_mesh_study_v2_neff_kappa_only. The shared-slab reference
is an explicitly recorded diagnostic model, not a production-model migration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .block0 import (base_profiles, compare_samples, domain_reference_profile,
                     implementation_identity, installation_identity)
from .block_io import digest, load_block, write_json
from .config import load_config
from .geometry import ClampedCubicProfile, make_geometry
from .modes import ModeReference, ModeSnapshot
from .native_backend import NativeLumapiBackend, scalar
from .progress import ProgressReporter
from .runtime import BudgetExceeded

METHOD = 'endpoint_shared_grid_mesh_study_v2_neff_kappa_only'
SPACINGS_M = (40e-9, 30e-9, 20e-9, 15e-9, 10e-9, 7.5e-9)


class StudyBackend(NativeLumapiBackend):
    """Diagnostic profile included in every native/cache identity."""
    study_profile = None
    retain_slabs = True

    def native_config(self, mesh_accuracy=None):
        payload = super().native_config(mesh_accuracy)
        payload.update(diagnostic_retain_both_slabs=float(self.retain_slabs),
                       diagnostic_shared_mesh=1., diagnostic_record_mesh=1.)
        if self.study_profile is not None:
            payload['full_profile'] = self.study_profile.native()
            payload['compact_profile'] = self.study_profile.native()
        return payload


def matching_grids(first, second, *, bounds=None):
    """Require equal actual coordinates, not just equal requested cell counts."""
    errors = {}
    for axis in ('x', 'y'):
        a, b = (np.asarray(value[axis+'_m']).reshape(-1) for value in (first, second))
        if bounds is not None:
            lo, hi = bounds[axis]
            a, b = a[(a >= lo) & (a <= hi)], b[(b >= lo) & (b <= hi)]
        equal_shape = a.shape == b.shape and a.size >= 2
        errors[axis+'_maximum_error_m'] = float(np.max(abs(a-b))) if equal_shape else None
        errors[axis+'_counts'] = [len(a), len(b)]
        errors[axis+'_passed'] = bool(equal_shape and np.allclose(a,b,rtol=0,atol=1e-15))
    return {'passed': errors['x_passed'] and errors['y_passed'], **errors}


def optical_bounds(geometry, padding):
    return {'x': (min(-geometry.w2_m/2, geometry.d_c2c_m-geometry.w1_m/2)-padding,
                  max(geometry.w2_m/2, geometry.d_c2c_m+geometry.w1_m/2)+padding),
            'y': (-padding, geometry.upper_top_y_m+padding)}


def sample_cases(config, endpoint):
    """Frozen sections of the same initial monotone profile; no width search."""
    from scipy.optimize import brentq
    length = config.device.length_m
    width = ClampedCubicProfile(length,np.linspace(endpoint.w1_input_m,endpoint.w1_output_m,
                                                    config.optimizer.control_points))
    offset = ClampedCubicProfile(length,np.linspace(config.geometry.d_c2c_input_m,
                      config.geometry.d_c2c_output_m,config.optimizer.control_points))
    center = brentq(lambda u: float(offset(u*length)),0,1)*length
    shoulder = brentq(lambda u: float(offset(u*length))-3e-6,center/length,1)*length
    result = []
    for state in ('cross','bar'):
        for name,z in (('input',0.),('center',center),('shoulder',shoulder),('output',length)):
            for wavelength in (config.device.wavelength_min_m,config.device.wavelength_max_m):
                result.append(dict(name=f'{state}_{name}_{wavelength*1e9:.0f}',state=state,
                    z_m=float(z),w1_m=float(width(z)),d_c2c_m=float(offset(z)),
                    wavelength_m=float(wavelength),h_gap_m=endpoint.h_gap_m,h_rib_m=endpoint.h_rib_m))
    return result


def two_pass_choice(profiles, comparisons):
    """Middle of a passing triple, also tested directly against its finest."""
    for i in range(2,len(profiles)):
        if comparisons[i-2]['passed'] and comparisons[i-1]['passed']:
            return profiles[i-1], profiles[i]
    return None


class Study:
    def __init__(self, config, endpoint, output):
        self.config, self.endpoint, self.out = config, endpoint, Path(output).resolve()
        if self.out.exists():
            raise FileExistsError('Use a new mesh-study directory; budgets/checkpoints are not reset.')
        self.out.mkdir(parents=True)
        self.started = time.monotonic()
        self.progress = ProgressReporter(METHOD,self.out)
        self.backend = StudyBackend(config,progress=self.progress)
        base = base_profiles(config)['coupled']
        # One physical comparison grid for ALL settings; fields outside each
        # solver domain remain zero. It has a nested every-other-point grid.
        xspan = base.x_span_m+8e-6
        ymin,ymax = base.y_min_m-2e-6,base.y_max_m+4e-6
        self.base = replace(base,local_padding_m=.6e-6,
            tracking_x_span_m=xspan,tracking_y_min_m=ymin,tracking_y_max_m=ymax,
            tracking_nx=2*int(np.ceil(xspan/config.convergence.comparison_dx_m/2))+1,
            tracking_ny=2*int(np.ceil((ymax-ymin)/config.convergence.comparison_dy_m/2))+1)
        self.cases = sample_cases(config,endpoint)
        self.probes = [r for r in self.cases if r['state']=='cross' and
                       (r['name'].split('_')[1] in ('input','center'))]
        self.rows = {}
        self.physical_samples = {}
        self.report = dict(method=METHOD,status='running',execution_completed=False,
            started_utc=datetime.now(timezone.utc).isoformat(),
            implementation_id=implementation_identity(),installation=installation_identity(config),
            study_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            input_id=digest(config.as_dict()),limits=asdict(config.convergence),
            acceptance_policy=config.convergence.acceptance_policy,
            delta_beta_acceptance_enforced=False,
            device_length_m=config.device.length_m,
            budget=asdict(config.execution),spacing_ladder_m=SPACINGS_M,
            geometry_source='Frozen initial monotone spline using existing fixed endpoint geometry; center is zero offset, not an assumed phase-matching width.',
            cases=self.cases,samples=[],comparisons=[],axes={},
            numerical_convergence_passed=False,production_certified=False,
            physical_basis_validated=False,physical_device_validated=False,
            production_definition_changed=False,block2_launched=False,
            clipping_applied=False)
        write_json(self.out/'configuration.json',config.as_dict())
        self.save()

    def save(self):
        self.report.update(fresh_fde_solves=self.backend.budget.solves,
            elapsed_seconds=time.monotonic()-self.started)
        write_json(self.out/'mesh_study.json',self.report)

    def geometry(self, case):
        liquid = self.backend.bank.liquid_cross if case['state']=='cross' else self.backend.bank.liquid_bar
        return make_geometry(self.config.geometry,**{k:case[k] for k in
            ('w1_m','d_c2c_m','h_gap_m','h_rib_m','state')},
            liquid_index=float(liquid.index(case['wavelength_m'])))

    def sample(self, label, profile, case, *, retain=True, anchor=None):
        key = label,case['name']
        if key in self.rows:
            return self.rows[key]
        physical_key = digest(dict(profile=asdict(profile),case=case,retain_slabs=retain))
        if physical_key in self.physical_samples:
            # Identical frozen geometry/profile and selected branch, already
            # archived with fields; no need to repeat an eigensolve for an alias.
            return self.physical_samples[physical_key]
        self.backend.study_profile,self.backend.retain_slabs = profile,retain
        reference = self.load(anchor)[1] if anchor is not None else None
        references = (reference,) if reference is not None and reference.valid else ()
        geometry = self.geometry(case)
        self.progress.note(f'{label}: {case["name"]}; local dx/dy {profile.local_dx_m*1e9:g}/{profile.local_dy_m*1e9:g} nm',force=True)
        solves = self.backend.budget.solves
        started = time.monotonic()
        coefficient, ref = self.backend._guarded_coefficients(geometry,case['wavelength_m'],
            references,None,diagnostic=True)
        raw = self.backend._raw_packets([self.backend._raw_request(geometry,case['wavelength_m'],u,l)
                                         for u,l in ((1,0),(0,1),(1,1))])
        grids = [{axis+'_m':np.asarray(packet['items'][0]['mesh'][axis+'_m']).reshape(-1)
                  for axis in ('x','y')} for packet in raw]
        shared = [matching_grids(grids[0],grid) for grid in grids[1:]]
        modes = (ref.upper,ref.lower,*ref.supermodes)
        value = dict(neff=np.array([m.neff for m in modes]),
            fields=np.array([[m.ex,m.ey] for m in modes]),te=np.array([m.te_fraction for m in modes]),
            core=np.array([m.core_fraction for m in modes]),wavelength_m=case['wavelength_m'],
            delta_beta=coefficient.delta_beta,kappa=coefficient.kappa,flag=coefficient.extraction_flag)
        path = self.out/'samples'/f'{label}_{case["name"]}.npz'
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,**{k:v for k,v in value.items() if k!='flag'},
            tracking_x_m=modes[0].x_m,tracking_y_m=modes[0].y_m,
            **{f'raw_{i}_{axis}_m':grid[axis+'_m'] for i,grid in enumerate(grids) for axis in ('x','y')})
        row = dict(label=label,case=case,profile=asdict(profile),retain_both_slabs=retain,
            artifact=path.relative_to(self.out).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            elapsed_seconds=time.monotonic()-started,fresh_solves=self.backend.budget.solves-solves,
            coefficients=asdict(coefficient),shared_grid_checks=shared,
            mode_identities=[m.identity for m in modes],reference_valid=ref.valid,
            core_fractions=value['core'].tolist(),te_fractions=value['te'].tolist(),
            actual_mesh=[{k:scalar(v) for k,v in packet['items'][0]['mesh'].items()
                          if k not in ('x_m','y_m')} for packet in raw])
        self.rows[key] = row
        self.physical_samples[physical_key] = row
        self.report['samples'].append(row)
        self.save()
        # Avoid retaining all eight candidate fields for every geometry.
        self.backend._native_memory.clear()
        if not all(r['passed'] for r in shared):
            raise RuntimeError('Actual upper/lower/coupled grids differ; coefficient comparison is not controlled.')
        return row

    def load(self, row):
        path = self.out/row['artifact']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Mesh-study sample checksum mismatch.')
        with np.load(path,allow_pickle=False) as data:
            sample = {k:np.array(data[k]) for k in ('neff','fields','te','core','wavelength_m','delta_beta','kappa')}
            grids = [{axis+'_m':np.array(data[f'raw_{i}_{axis}_m']) for axis in ('x','y')} for i in range(3)]
            modes = [ModeSnapshot(complex(sample['neff'][i]),sample['fields'][i,0],sample['fields'][i,1],
                float(sample['te'][i]),float(sample['core'][i]),row['mode_identities'][i],
                np.array(data['tracking_x_m']),np.array(data['tracking_y_m']),
                row['case']['d_c2c_m'] if i==0 else 0.) for i in range(4)]
        sample.update(flag=row['coefficients']['extraction_flag'])
        return sample,ModeReference(modes[0],modes[1],tuple(modes[2:]),valid=row['reference_valid']),grids

    def compare(self, label, actual, reference, *, fixed_lattice=False):
        a,_,ag = self.load(actual)
        b,_,bg = self.load(reference)
        result = compare_samples(self.config,a,b)
        if fixed_lattice:
            bounds = optical_bounds(self.geometry(actual['case']),self.base.local_padding_m)
            checks = [matching_grids(x,y,bounds=bounds) for x,y in zip(ag,bg)]
            result.update(fixed_optical_lattice_checks=checks)
            result['passed'] &= all(c['passed'] for c in checks)
        record = dict(stage=label,actual=actual['label'],reference=reference['label'],
                      case=actual['case']['name'],**result)
        self.report['comparisons'].append(record)
        self.save()
        return result

    def group(self,label,profile,cases,*,anchors=None):
        return [self.sample(label,profile,case,anchor=None if anchors is None else anchors[i])
                for i,case in enumerate(cases)]

    def compare_group(self,label,actual,reference,*,fixed_lattice=False):
        checks = [self.compare(label,a,b,fixed_lattice=fixed_lattice) for a,b in zip(actual,reference)]
        return dict(passed=all(c['passed'] for c in checks),checks=checks)

    def axis(self,axis,anchor_profile):
        profiles,groups,checks = [],[],[]
        for spacing in SPACINGS_M:
            profile = replace(anchor_profile,**{'local_d'+axis+'_m':spacing})
            label = f'mesh_{axis}_{spacing*1e9:g}nm'
            group = self.group(label,profile,self.probes,anchors=groups[-1] if groups else None)
            profiles.append(profile);groups.append(group)
            if len(groups)>1:
                checks.append(self.compare_group('mesh_'+axis,groups[-2],groups[-1]))
                choice = two_pass_choice(profiles,checks)
                if choice:
                    self.report['axes'][axis] = dict(passed=True,
                        candidate_profile=asdict(choice[0]),reference_profile=asdict(choice[1]))
                    self.save()
                    return choice,groups[-2],groups[-1]
        self.report['axes'][axis] = dict(passed=False,reason='No two consecutive passing refinements within the declared ladder.')
        self.save()
        return None,groups[-2],groups[-1]

    def execute(self):
        try:
            first = replace(self.base,local_dx_m=SPACINGS_M[0],local_dy_m=SPACINGS_M[0])
            # Background comparison changes only the retained slabs; the complete
            # geometry-derived mesh is identical in both diagnostic alternatives.
            for case in self.probes:
                legacy = self.sample('legacy_shared_grid',first,case,retain=False)
                matched = self.sample('matched_shared_grid',first,case,anchor=legacy)
                self.compare('background_definition',legacy,matched)
            xchoice,_,_ = self.axis('x',first)
            xprofile = xchoice[1] if xchoice else replace(first,local_dx_m=SPACINGS_M[-1])
            ychoice,_,_ = self.axis('y',xprofile)
            if not xchoice or not ychoice:
                self.report.update(status='mesh_gate_failed',execution_completed=True,
                    next_action='Extend/revise the failed mesh ladder using the saved per-axis evidence; do not launch Block 2.')
                return
            candidate = replace(ychoice[0],local_dx_m=xchoice[0].local_dx_m)
            finest = ychoice[1]
            # Cartesian cross-check: x must remain adequate after y refinement.
            cg = self.group('combined_candidate',candidate,self.cases)
            rg = self.group('combined_reference',finest,self.cases,anchors=cg)
            joint = self.compare_group('candidate_to_finest_full_cases',cg,rg)
            self.report['combined_mesh_passed'] = joint['passed']
            if not joint['passed']:
                self.report.update(status='combined_mesh_gate_failed',execution_completed=True)
                return
            current = finest
            current_group = self.group('domain_anchor',current,self.probes)
            for axis in ('domain_x','domain_top','domain_bottom'):
                anchor = current
                axis_checks = []
                for factor in (1.5,2.):
                    expanded = domain_reference_profile(self.config,anchor,'coupled',axis,factor)
                    label = f'{axis}_{factor:g}'
                    group = self.group(label,expanded,self.probes,anchors=current_group)
                    axis_checks.append(self.compare_group(axis,current_group,group,fixed_lattice=True))
                    current,current_group = expanded,group
                self.report['axes'][axis] = dict(passed=all(c['passed'] for c in axis_checks))
                if not self.report['axes'][axis]['passed']:
                    self.report.update(status='domain_gate_failed',execution_completed=True)
                    return
            pml = replace(current,pml_layers=current.pml_layers+self.config.convergence.pml_test_extra_layers)
            pg = self.group('pml_guard',pml,self.probes,anchors=current_group)
            pml_check = self.compare_group('pml_guard',current_group,pg)
            # Final enlarged-domain/PML reference across every frozen case.
            final = self.group('final_reference',pml,self.cases)
            final_check = self.compare_group('joint_mesh_domain_pml',cg,final)
            passed = pml_check['passed'] and final_check['passed']
            self.report.update(status='scoped_numerical_pass' if passed else 'joint_gate_failed',
                execution_completed=True,numerical_convergence_passed=passed,
                selected_profile=asdict(candidate) if passed else None,
                reference_profile=asdict(pml),
                next_action='Reverify endpoint handoff and reference-basis adequacy before production; this study issues no certificate.')
        except BudgetExceeded as exc:
            self.report.update(status='incomplete_budget',error=str(exc),execution_completed=False)
            self.progress.fail(exc)
        except Exception as exc:
            self.report.update(status='failed',error=f'{type(exc).__name__}: {exc}',execution_completed=False)
            self.progress.fail(exc)
        finally:
            self.backend.close()
            self.save()
            render_report(self.out,self.report)
            if self.report.get('execution_completed'):
                self.progress.finish()


def render_report(out, report):
    """Scalar journal and scientific plots also survive a failed/incomplete study."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = report['comparisons']
    columns = ['stage','case','actual','reference','passed','maximum_real_neff_error',
               'delta_beta_error_rad_per_m','kappa_error_rad_per_m','minimum_transverse_e_power_overlap']
    with (out/'comparisons.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    fig,axes = plt.subplots(2,1,figsize=(10,8),layout='constrained')
    for ax,axis in zip(axes,('x','y')):
        for name in sorted({r['case'] for r in rows if r['stage']=='mesh_'+axis}):
            selected = [r for r in rows if r['stage']=='mesh_'+axis and r['case']==name]
            x = np.arange(1,len(selected)+1)
            ax.plot(x,[r['delta_beta_error_rad_per_m'] for r in selected],'o-',label=name+' Δβ')
            ax.plot(x,[r['kappa_error_rad_per_m'] for r in selected],'x--',label=name+' κ')
        ax.axhline(report['limits']['phase_error_rad']/report['device_length_m'],
                   color='k',linestyle=':',label='κ phase ceiling (κ absolute/relative limits also apply)')
        ax.set(title=f'{axis} refinement: successive complex-coefficient differences',
               xlabel='Refinement comparison (coarse to fine)',ylabel='Absolute error (rad/m)',yscale='symlog')
        handles,labels=ax.get_legend_handles_labels()
        ax.legend(handles,labels,fontsize=7,ncol=2)
    fig.suptitle(f'{METHOD}\n{report["status"]}; actual MODE data; no production certificate')
    fig.savefig(out/'mesh_convergence.png',dpi=160)
    fig.savefig(out/'mesh_convergence.pdf')
    plt.close(fig)
    worst = lambda key: max((r.get(key,0) for r in rows if r['stage'].startswith('mesh_')),default=0)
    (out/'report.md').write_text(
        f'# Targeted mesh study\n\nStatus: **{report["status"]}**. '
        f'Execution completed: {report["execution_completed"]}. '
        f'Numerical convergence passed: {report["numerical_convergence_passed"]}.\n\n'
        f'Fresh FDE solves: {report["fresh_fde_solves"]}; elapsed minutes: {report["elapsed_seconds"]/60:.2f}.\n\n'
        'Both endpoint wavelengths only. Frozen input, zero-offset center, +3 µm shoulder and output sections; '
        'initial probes cover cross-state input and center. Broader checks run only after independent axes pass. '
        'The shared-slab reference is a named diagnostic alternative; no production model or handoff was replaced. '
        'No clipping, Block 0 certificate or physical device validation.\n\n'
        f'Acceptance policy: {report.get("acceptance_policy","legacy recorded policy")}; '
        f'effective-index limit {report["limits"]["neff_atol"]:g}. '
        'Under neff_kappa_only_v1, detuning errors are diagnostic outputs only; '
        'the existing coupling, loss, field and tracking gates remain active.\n\n'
        f'Worst recorded mesh changes: Δβ {worst("delta_beta_error_rad_per_m"):.9g} rad/m; '
        f'complex κ {worst("kappa_error_rad_per_m"):.9g} rad/m. '
        'See comparisons.csv for each refinement, not just these maxima.\n\n'
        f'{report.get("next_action",report.get("error","Review the failed axis evidence."))}\n',encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--endpoint-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-solves',type=int,default=300)
    parser.add_argument('--max-seconds',type=float,default=18000)
    args = parser.parse_args(argv)
    if args.max_solves<6 or not 0<args.max_seconds<=18000:
        parser.error('Require >=6 solves and a positive time budget <=5 hours.')
    cfg = load_config(args.config)
    endpoint,_,_,_ = load_block(args.endpoint_dir,1,cfg)
    cfg = replace(cfg,solver=replace(cfg.solver,coupling_zero_evidence=None,hide_gui=True,
                      raw_cache_memory_entries=3),
                  execution=replace(cfg.execution,max_solves=args.max_solves,max_seconds=args.max_seconds,
                                    resume=False,native_batch_size=1))
    cfg.validate()
    study = Study(cfg,endpoint,args.out)
    study.execute()
    return 0 if study.report['numerical_convergence_passed'] else 3
