"""Targeted real MODE endpoint fields; never reruns a workflow block."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vlac.block_io import write_json
from vlac.checkpoints import decode
from vlac.config import load_config
from vlac.geometry import make_geometry
from vlac.modes import ModeSnapshot, field_overlap
from vlac.native_backend import NativeLumapiBackend, snapshot_from_native, scalar


class ReadOnlyCache:
    def __init__(self, path):
        self.path = Path(path).resolve()
    def get(self, key, **kwargs):
        with sqlite3.connect(self.path.as_uri()+'?mode=ro', uri=True) as db:
            row = db.execute('SELECT payload,sha256 FROM cache_v2 WHERE cache_key=?', (key,)).fetchone()
        if row is None:
            return None
        if hashlib.sha256(row[0]).hexdigest() != row[1]:
            raise ValueError('Diagnostic cache checksum mismatch')
        return decode(row[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block1', type=Path, required=True)
    parser.add_argument('--block2', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.block2/'configuration.json')
    cfg.execution.max_solves = 60
    cfg.execution.max_seconds = 2700
    cfg.execution.native_batch_size = 1
    cfg.output.save_solver_projects = False
    endpoint = json.loads((args.block1/'endpoint_diagnostics.json').read_text())['mode_profiles']['modes']
    saved = np.load(args.block1/'endpoint_mode_profiles.npz')
    rows = np.genfromtxt(args.block2/'cross_coefficients.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
    print('columns:', rows.dtype.names, flush=True)
    backend = NativeLumapiBackend(cfg)
    old_cache = ReadOnlyCache(cfg.output.cache_sqlite)
    packets, arrays, report = [], {}, []
    began = time.monotonic()
    try:
        for end in ('input', 'output'):
            old = next(m for m in endpoint if m['guide']=='upper' and m['state']=='cross'
                       and m['end']==end and abs(m['wavelength_m']-1.26e-6)<1e-15)
            g = make_geometry(cfg.geometry, w1_m=old['geometry']['w1_m'],
                h_rib_m=old['geometry']['h_rib_m'], h_gap_m=old['geometry']['h_gap_m'],
                d_c2c_m=old['geometry']['d_c2c_m'], state='cross', liquid_index=1.55)
            request = backend._raw_request(g, 1.26e-6, True, False)
            raw = old_cache.get(backend._raw_key(request))
            source = 'original checksummed Block 2 raw-mode cache'
            if raw is None:
                raw = backend._raw_packets([request], use_cache=False)[0]
                source = 'targeted fresh FDE reconstruction at original settings'
            modes = [snapshot_from_native(v) for v in raw['items'][:int(scalar(raw['n']))]]
            x = np.linspace(-backend.x_span_m/2, backend.x_span_m/2, cfg.solver.tracking_grid_nx)
            y = np.linspace(backend.y_min_m, backend.y_max_m, cfg.solver.tracking_grid_ny)
            prefix = old['array_prefix']
            ox, oy = saved[prefix+'_x_m'], saved[prefix+'_y_m']
            ex, ey = saved[prefix+'_ex'], saved[prefix+'_ey']
            # Compare isolated guides in guide-relative coordinates, with zero padding.
            xx, yy = np.meshgrid(x-g.d_c2c_m, y, indexing='ij')
            query = np.stack([xx.ravel(), yy.ravel()], axis=-1)
            aligned = [RegularGridInterpolator((ox,oy), f, bounds_error=False, fill_value=0)(query).reshape(xx.shape)
                       for f in (ex,ey)]
            old_neff = complex(old['neff']['real'], old['neff']['imag'])
            anchor = ModeSnapshot(old_neff, *aligned, old['te_fraction'], 0., old['identity'])
            # The published CSV contains z_m and wavelength_m in SI.
            row = next(r for r in rows if abs(float(r['wavelength_m'])-1.26e-6)<1e-15
                       and abs(float(r['z_m'])-(0 if end=='input' else cfg.device.length_m))<1e-15)
            expected_neff = float(row['beta_upper_real_rad_per_m'])*1.26e-6/(2*np.pi)
            selected = min(modes, key=lambda m: abs(m.neff.real-expected_neff))
            if abs(selected.neff.real-expected_neff)>2e-5:
                raise ValueError('Cannot reproduce published selected mode: '+end)
            recovered = max(modes, key=lambda m: field_overlap(m, anchor))
            entry = {'end':end, 'wavelength_m':1.26e-6, 'geometry':g.__dict__, 'field_source':source,
                     'block1_neff':old_neff.real, 'block2_selected_neff':selected.neff.real,
                     'full_domain_anchor_neff':recovered.neff.real,
                     'selected_anchor_overlap':field_overlap(selected,anchor),
                     'recovered_anchor_overlap':field_overlap(recovered,anchor),
                     'selected_core_fraction':selected.core_fraction,
                     'recovered_core_fraction':recovered.core_fraction,
                     'recovered_neff_error':abs(recovered.neff-old_neff),
                     'candidates':[{'identity':m.identity,'core_fraction':m.core_fraction,
                         'te_fraction':m.te_fraction,'anchor_overlap':field_overlap(m,anchor)} for m in modes]}
            report.append(entry)
            packets.append(((ox,oy,ex,ey,old_neff.real),
                (x-g.d_c2c_m,y,selected.ex,selected.ey,selected.neff.real),
                (x-g.d_c2c_m,y,recovered.ex,recovered.ey,recovered.neff.real)))
            for name, packet in zip(('block1','block2_selected','full_domain_anchor'), packets[-1]):
                for key, value in zip(('x_m','y_m','ex','ey','neff'),packet):
                    arrays[f'{end}_{name}_{key}'] = value
            print(json.dumps(entry), flush=True)
        fig, axes = plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
        for i,(end,pack) in enumerate(zip(('input','output'),packets)):
            for j,(x,y,ex,ey,neff) in enumerate(pack):
                intensity = abs(ex)**2+abs(ey)**2
                ax=axes[i,j]
                im=ax.pcolormesh(x*1e6,y*1e6,(intensity/intensity.max()).T,shading='auto',vmin=0,vmax=1,cmap='magma')
                width=report[i]['geometry']['w1_m']*1e6
                ax.add_patch(Rectangle((-width/2,1.35),width,.35,fill=False,edgecolor='cyan',lw=1))
                ax.axhline(1.525,color='cyan',lw=.6,alpha=.5)
                ax.set(xlim=(-5,5),ylim=(.6,2.6),xlabel='x relative to upper ridge (um)',ylabel='y (um)',
                       title=f"{end.capitalize()} - {('Block 1 verified','Block 2 selected','Full-domain anchor match')[j]}\nneff = {neff:.9f}")
        fig.suptitle('1260 nm cross state: normalized transverse electric intensity |Ex|² + |Ey|²\n'
                     'Same endpoint width; guide-centered display; each panel scaled to its own peak')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Normalized intensity',shrink=.8)
        for suffix in ('png','pdf'):
            fig.savefig(args.out/f'endpoint_mode_comparison.{suffix}',dpi=180)
        plt.close(fig)
        np.savez_compressed(args.out/'endpoint_mode_comparison.npz',**arrays)
        write_json(args.out/'endpoint_mode_comparison.json',{
            'scope':'real saved fields or explicitly labeled targeted reconstruction; no full block runs',
            'fresh_fde_solves':backend.budget.solves,'runtime_seconds':time.monotonic()-began,
            'max_solves':60,'max_seconds':2700,'endpoints':report})
    finally:
        backend.close()

if __name__=='__main__':
    main()
