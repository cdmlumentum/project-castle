"""Prepare a separate capped pilot, preserving every original optical setting."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlac.block_io import write_json
from vlac.config import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    if out.exists():
        raise FileExistsError('Choose a separate, new pilot directory.')
    cfg=load_config(args.baseline)
    before=cfg.as_dict()
    cfg.solver.tracking_method='guarded_translation_v1'
    cfg.solver.tracking_recovery_depth=3
    cfg.optimizer.method='monotone_powell'
    cfg.optimizer.max_function_evaluations_per_grid=6
    cfg.optimizer.max_grid_refinements=0
    cfg.optimizer.complete_within_seconds=None
    cfg.execution.max_solves=120
    cfg.execution.max_seconds=2700
    cfg.execution.native_batch_size=1
    cfg.execution.resume=False
    cfg.output.directory=str(out)
    cfg.output.cache_sqlite=str(out/'cache/native_fde_cache.sqlite')
    cfg.validate()
    out.mkdir(parents=True)
    write_json(out/'pilot_config.json',cfg.as_dict())
    write_json(out/'pilot_plan.json',{
        'method':'monotone_powell + guarded_translation_v1',
        'baseline_configuration':str(args.baseline.resolve()),
        'max_seconds':2700,'max_fresh_fde_solves':120,'max_objective_evaluations':6,
        'midpoint_validation':'unchanged independent z midpoints and 0.01 tolerance',
        'optical_settings_preserved':all(before[k]==cfg.as_dict()[k] for k in ('geometry','materials','device','convergence')),
        'solver_changes':{'tracking_method':cfg.solver.tracking_method,'tracking_recovery_depth':3},
        'longer_run_gate':'Requires passed endpoint consistency, candidate validity and independent midpoint validation; never auto-launches a longer run.'})
    print(out/'pilot_config.json')

if __name__=='__main__': main()
