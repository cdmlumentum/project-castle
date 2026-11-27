"""Bounded real FDE sensitivity checks on frozen Block 2 sentinel sections.

This is a targeted cross-state study, not a Block 0 certificate or full profile
convergence claim. Flagged complex coupling is retained for diagnosis only.
"""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlac.block_io import load_block, decode_solution, write_json
from vlac.config import load_config
from vlac.coefficient_stability import compare_coefficients
from vlac.geometry import profile_from_values, make_geometry
from vlac.native_backend import NativeLumapiBackend
from vlac.progress import ProgressReporter
from vlac.runtime import BudgetExceeded
from vlac.provenance import artifact_identity, file_digest
from vlac.tracking import require_anchor_match
from diagnose_block2_tracking import ReadOnlyCache


class DiagnosticCache:
    """Read matching checksummed pilot entries without modifying its cache."""
    def __init__(self,primary,pilot_path):
        self.primary=primary
        self.previous=ReadOnlyCache(pilot_path) if pilot_path.is_file() else None
        self.reused=0
    def get(self,key,*,kind='other'):
        value=self.primary.get(key,kind=kind)
        if value is None and self.previous is not None:
            value=self.previous.get(key)
            if value is not None:
                self.primary.put(key,value,kind=kind)
                self.reused+=1
        return value
    def put(self,key,value,*,kind='other'):
        return self.primary.put(key,value,kind=kind)


def frozen_profiles(config,endpoint,pilot):
    if (pilot/'block2_result.json').is_file():
        _,solution,_,_=load_block(pilot,2,config)
        return solution.w1_control_m,solution.d_c2c_control_m,solution.anchor_z_m,'completed pilot geometry'
    candidates=list((pilot/'.attempts').glob('*/block2_best_candidate.json'))
    if candidates:
        path=max(candidates,key=lambda p:p.stat().st_mtime_ns)
        solution=decode_solution(json.loads(path.read_text())['solution'])
        return solution.w1_control_m,solution.d_c2c_control_m,solution.anchor_z_m,'best valid pilot candidate (unpublished)'
    n=config.optimizer.control_points
    return (np.linspace(endpoint.w1_input_m,endpoint.w1_output_m,n),
            np.linspace(config.geometry.d_c2c_input_m,config.geometry.d_c2c_output_m,n),
            None,
            'initial monotone geometry; no valid pilot candidate was published')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--endpoint-dir',type=Path,required=True)
    p.add_argument('--pilot-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    out=args.out.resolve()
    if out.exists(): raise FileExistsError('Use a new diagnostic directory.')
    cfg=load_config(args.config)
    cfg.execution.max_solves=60
    cfg.execution.max_seconds=2700
    cfg.execution.native_batch_size=1
    cfg.execution.resume=False
    endpoint,_,_,_=load_block(args.endpoint_dir,1,cfg)
    wc,dc,anchor_z,geometry_source=frozen_profiles(cfg,endpoint,args.pilot_dir)
    wp=profile_from_values(cfg.device.length_m,wc,anchor_z)
    dp=profile_from_values(cfg.device.length_m,dc,anchor_z)
    center=brentq(lambda u:float(dp(u*cfg.device.length_m)),0,1,xtol=1e-13)*cfg.device.length_m
    positions=[cfg.device.length_m,center]
    cases=[(float(wavelength),float(z)) for wavelength in endpoint.wavelengths_m for z in positions]
    settings=[('baseline',1.,1.),('mesh_1p25',1.25,1.),('mesh_1p5',1.5,1.),
              ('x_domain_1p25',1.,1.25),('x_domain_1p5',1.,1.5)]
    out.mkdir(parents=True)
    progress=ProgressReporter('targeted Block 2 coefficient stability',out)
    report={'scope':'cross-state output and aligned-guide sentinel sections at both band edges; no full-device or Block 0 certificate',
        'source_identity':artifact_identity(cfg),'script_sha256':file_digest(__file__),
        'comparison_helper_sha256':file_digest(ROOT/'src/vlac/coefficient_stability.py'),
        'status':'running','geometry_source':geometry_source,'w_control_m':wc,'d_control_m':dc,
        'anchor_z_m':anchor_z,'profile_kind':'pchip_anchors_v1' if anchor_z is not None else 'clamped_cubic_v1',
        'positions_m':positions,'settings':settings,'max_fde_solves':60,'max_seconds':2700,
        'production_mesh_and_domain_changed':False,
        'acceptance':asdict(cfg.convergence),'records':[],'comparisons':[],
        'long_run_eligible':False,'physical_device_validated':False}
    write_json(out/'configuration.json',cfg.as_dict())
    backend=NativeLumapiBackend(cfg,cache_path=out/'cache/native_fde_cache.sqlite',progress=progress)
    cache=DiagnosticCache(backend.cache,args.pilot_dir/'cache/native_fde_cache.sqlite')
    backend.cache=cache
    base_x=backend.x_span_m
    base_y=(backend.y_min_m,backend.y_max_m)
    baseline,previous={},{'mesh':{},'x_domain':{}}
    started=time.monotonic()
    try:
        for label,mesh,domain in settings:
            updated=replace(cfg,solver=replace(cfg.solver,
                mesh_cells_x=round(cfg.solver.mesh_cells_x*mesh*domain),
                mesh_cells_y=round(cfg.solver.mesh_cells_y*mesh)))
            backend.reconfigure(updated)
            with backend._temporary_domain(x_span_m=base_x*domain,y_min_m=base_y[0],y_max_m=base_y[1]):
                for wavelength,z in cases:
                    backend.budget.check()
                    geometry=make_geometry(cfg.geometry,w1_m=float(wp(z)),h_rib_m=endpoint.h_rib_m,
                        h_gap_m=endpoint.h_gap_m,d_c2c_m=float(dp(z)),state='cross',
                        liquid_index=float(backend.bank.liquid_cross.index(wavelength)))
                    key=(wavelength,z)
                    prior=baseline.get(key)
                    coefficient,reference=backend._guarded_coefficients(geometry,wavelength,(),
                        None if prior is None else prior[0].kappa,diagnostic=True)
                    identity=[]
                    if prior is not None:
                        old=prior[1]
                        identity=[require_anchor_match(a,b,cfg) for a,b in zip(
                            (reference.upper,reference.lower,*reference.supermodes),
                            (old.upper,old.lower,*old.supermodes))]
                    record={'setting':label,'mesh_factor':mesh,'x_domain_factor':domain,
                        'wavelength_m':wavelength,'z_m':z,'geometry':asdict(geometry),
                        'x_span_m':backend.x_span_m,'mesh_cells_x':updated.solver.mesh_cells_x,
                        'mesh_cells_y':updated.solver.mesh_cells_y,
                        'anchor_field_overlaps':identity,'extraction_valid':reference.valid,
                        'coefficients':asdict(coefficient)}
                    report['records'].append(record)
                    if label=='baseline':
                        baseline[key]=(coefficient,reference)
                        previous['mesh'][key]=coefficient
                        previous['x_domain'][key]=coefficient
                    else:
                        axis='mesh' if mesh>1 else 'x_domain'
                        comparison=compare_coefficients(cfg,coefficient,prior[0],wavelength)
                        adjacent=compare_coefficients(cfg,coefficient,previous[axis][key],wavelength)
                        report['comparisons'].append({'setting':label,'axis':axis,'wavelength_m':wavelength,'z_m':z,
                            'against_production_baseline':comparison,'against_previous_setting':adjacent})
                        previous[axis][key]=coefficient
                    report.update(elapsed_seconds=time.monotonic()-started,fresh_fde_solves=backend.budget.solves)
                    write_json(out/'coefficient_stability.json',report)
                    print(f'{label}, {wavelength*1e9:.0f} nm, z={z*1e6:.3f} um: '
                          f'delta={coefficient.delta_beta.real:.6g}, kappa={coefficient.kappa}, '
                          f'flag={coefficient.extraction_flag}',flush=True)
        report['execution_completed']=True
        report['coefficient_checks_passed']=all(c['against_production_baseline']['passed'] and
            c['against_previous_setting']['passed'] for c in report['comparisons'])
        report['all_extractions_valid']=all(r['extraction_valid'] for r in report['records'])
        report['status']='passed_targeted_checks' if report['coefficient_checks_passed'] and report['all_extractions_valid'] else 'review'
    except Exception as exc:
        report.update(status='incomplete' if isinstance(exc,BudgetExceeded) else 'failed',
                      execution_completed=False,error=str(exc),traceback=traceback.format_exc())
        progress.fail(exc)
        print(report['error'],flush=True)
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,fresh_fde_solves=backend.budget.solves)
        report['exact_key_pilot_cache_entries_reused']=cache.reused
        backend.close()
        write_json(out/'coefficient_stability.json',report)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    with (out/'coefficient_stability.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=('setting','wavelength_m','z_m','delta_beta_real_rad_per_m','delta_beta_imag_rad_per_m',
                'kappa_real_rad_per_m','kappa_imag_rad_per_m','extraction_flag','extraction_valid')
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for record in report['records']:
            value=record['coefficients']
            writer.writerow({'setting':record['setting'],'wavelength_m':record['wavelength_m'],'z_m':record['z_m'],
                'delta_beta_real_rad_per_m':value['delta_beta'].real,
                'delta_beta_imag_rad_per_m':value['delta_beta'].imag,
                'kappa_real_rad_per_m':value['kappa'].real,'kappa_imag_rad_per_m':value['kappa'].imag,
                'extraction_flag':value['extraction_flag'],'extraction_valid':record['extraction_valid']})
    for wavelength,z in cases:
        rows=[r for r in report['records'] if r['wavelength_m']==wavelength and r['z_m']==z]
        values=[r['coefficients'] for r in rows]
        for ax,name,part in ((axes[0,0],'delta_beta','real'),(axes[0,1],'delta_beta','imag'),
                             (axes[1,0],'kappa','real'),(axes[1,1],'kappa','imag')):
            ax.plot(range(len(rows)),[getattr(v[name],part) for v in values],'o-',
                    label=f'{wavelength*1e9:.0f} nm, z={z*1e6:.1f} um')
            ax.set_title(name+' '+part+' (rad/m)')
            ax.set_xticks(range(5),[s[0] for s in settings],rotation=25)
            ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Real FDE sentinel sensitivity; flagged complex values retained for diagnosis')
    fig.savefig(out/'coefficient_stability.png',dpi=160)
    plt.close(fig)
    if report.get('execution_completed'): progress.finish()
    return 0 if report['status']=='passed_targeted_checks' else 3

if __name__=='__main__': raise SystemExit(main())
