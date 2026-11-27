"""Targeted endpoint/center matched-slab diagnostic, at band edges only.

Does not run Block 1/2 or generate an approved uncertainty bound. The named
diagnostic alternative retains both continuous slabs while removing the other
ridge. Production definitions, mesh and handoffs remain unchanged.
"""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlac.block_io import load_block, write_json
from vlac.config import load_config
from vlac.native_backend import NativeLumapiBackend
from vlac.geometry import ClampedCubicProfile, make_geometry
from vlac.progress import ProgressReporter
from vlac.tracking import require_anchor_match
from vlac.coefficient_stability import compare_coefficients
from vlac.coupling_uncertainty import case_id, evidence_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--endpoint-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.solver.coupling_zero_evidence = None
    cfg.execution.max_solves,cfg.execution.max_seconds = 60,2700
    cfg.execution.resume,cfg.execution.native_batch_size = False,1
    cfg.validate()
    endpoint,_,_,_ = load_block(args.endpoint_dir,1,cfg)
    out = args.out.resolve()
    if out.exists(): raise FileExistsError('Use a new diagnostic output directory.')
    out.mkdir(parents=True)
    progress = ProgressReporter('matched-slab coefficient diagnostic',out)
    backend = NativeLumapiBackend(cfg,cache_path=out/'cache/native_fde_cache.sqlite',progress=progress)
    width = ClampedCubicProfile(cfg.device.length_m,np.linspace(endpoint.w1_input_m,endpoint.w1_output_m,cfg.optimizer.control_points))
    offsets = np.array([cfg.geometry.d_c2c_input_m,0.,cfg.geometry.d_c2c_output_m])
    from scipy.optimize import brentq
    offset_profile = ClampedCubicProfile(cfg.device.length_m,np.linspace(offsets[0],offsets[-1],cfg.optimizer.control_points))
    center = brentq(lambda u:float(offset_profile(u*cfg.device.length_m)),0,1)*cfg.device.length_m
    positions = [0.,center,cfg.device.length_m]
    report = {'method':'matched_slab_background_diagnostic_v1','status':'running',
        'identity':evidence_identity(cfg),'budget':{'max_solves':60,'max_seconds':2700},
        'geometry_source':'initial monotone profile with verified Block 1 endpoints',
        'production_definition_changed':False,'records':[],'comparisons':[],
        'uncertainty_bound_approved':False,'physical_device_validated':False}
    baseline,matched,previous = {},{},{}
    started = time.monotonic()
    try:
        for label,retain,factor,domain in [('legacy',False,1.,1.),('matched',True,1.,1.),
                ('matched_mesh_1p25',True,1.25,1.),('matched_mesh_1p5',True,1.5,1.)]:
            updated = replace(cfg,solver=replace(cfg.solver,mesh_cells_x=round(cfg.solver.mesh_cells_x*factor),
                mesh_cells_y=round(cfg.solver.mesh_cells_y*factor)))
            backend.reconfigure(updated)
            backend.diagnostic_retain_both_slabs = retain
            for wavelength in endpoint.wavelengths_m:
                for z,offset in zip(positions,offsets):
                    geometry = make_geometry(cfg.geometry,w1_m=float(width(z)),d_c2c_m=float(offset),
                        h_gap_m=endpoint.h_gap_m,h_rib_m=endpoint.h_rib_m,state='cross',
                        liquid_index=float(backend.bank.liquid_cross.index(wavelength)))
                    value,ref = backend._guarded_coefficients(geometry,float(wavelength),(),None,diagnostic=True)
                    key = (float(wavelength),float(z))
                    anchor = baseline.get(key)
                    overlaps = [] if anchor is None else [require_anchor_match(a,b,cfg) for a,b in zip(
                        (ref.upper,ref.lower,*ref.supermodes),(anchor[1].upper,anchor[1].lower,*anchor[1].supermodes))]
                    q = value.delta_pm**2-value.delta_12**2
                    report['records'].append({'setting':label,'wavelength_m':float(wavelength),'z_m':z,
                        'geometry':asdict(geometry),'case_id':case_id(geometry,wavelength),
                        'coefficients':asdict(value),'raw_q_real':q.real,'raw_q_imag':q.imag,
                        'aligned_field_overlaps_against_legacy':overlaps})
                    if label=='legacy': baseline[key] = value,ref
                    elif label=='matched': matched[key] = value
                    else:
                        report['comparisons'].append({'setting':label,'wavelength_m':float(wavelength),'z_m':z,
                            'against_matched_baseline':compare_coefficients(cfg,value,matched[key],wavelength),
                            'against_previous_mesh':compare_coefficients(cfg,value,previous[key],wavelength)})
                    if retain: previous[key] = value
                    write_json(out/'coupling_background.json',report)
        report.update(status='review',execution_completed=True,
            note='Sensitivity evidence only. Systematic basis error is not bounded by mesh differences. Review matched-background and domain evidence before constructing a reviewed_bound file.')
    except Exception as exc:
        report.update(status='incomplete',execution_completed=False,error=f'{type(exc).__name__}: {exc}')
        progress.fail(exc)
    finally:
        report.update(fresh_fde_solves=backend.budget.solves,elapsed_seconds=time.monotonic()-started)
        backend.close()
        write_json(out/'coupling_background.json',report)
        write_json(out/'configuration.json',cfg.as_dict())
    if report.get('execution_completed'): progress.finish()
    return 3  # A targeted diagnostic never certifies production coupling.


if __name__=='__main__': raise SystemExit(main())
