"""Supervise a time-planned Block 2 run; preserve logs and prevent idle sleep."""
from datetime import datetime, timedelta, timezone
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlac.block_io import load_block, write_json
from vlac.config import config_from_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint-dir',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--hours',type=float,default=8.)
    parser.add_argument('--max-solves',type=int,default=12000,
                        help='Block 2 solve cap, independent of the endpoint run budget.')
    args = parser.parse_args()
    if not 0 < args.hours <= 8:
        parser.error('--hours must be positive and at most 8')
    if args.max_solves < 1:
        parser.error('--max-solves must be positive')
    endpoint = Path(args.endpoint_dir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    record_path = out/'execution.json'
    if record_path.exists():
        raise RuntimeError('Use a new output directory; execution records are immutable run identities.')
    raw = json.loads((endpoint/'configuration.json').read_text(encoding='utf-8'))
    raw['project_root'] = str(ROOT)
    # The only supplied silicon table is reference data; preserve its bytes.
    raw['materials']['silicon_data_csv'] = str(ROOT/'data/green_si_2008_1260_1340.csv')
    for key in ('sin_extinction_csv','liquid_dispersion_csv'):
        if raw['materials'].get(key) and not Path(raw['materials'][key]).is_absolute():
            raise ValueError('Resolve material file paths in the endpoint configuration before supervision.')
    raw['output'].update(directory=str(out),cache_sqlite=str(out/'cache/native_fde_cache.sqlite'))
    seconds = args.hours*3600.
    # Leave another half hour outside the internal completion plan for startup,
    # unexpected timing variation, final file publication and MODE cleanup.
    plan_seconds = max(seconds*.75,seconds-1800.)
    raw['optimizer']['complete_within_seconds'] = plan_seconds
    raw['execution'] = dict(raw.get('execution',{}),max_seconds=seconds,
                            max_solves=args.max_solves,resume=True)
    raw['solver']['hide_gui'] = True
    config = config_from_dict(raw)
    load_block(endpoint,1,config)  # Check compatibility before acquiring a license.
    config_path = out/'run_configuration.json'
    write_json(config_path,config.as_dict())
    command = [sys.executable,'-u',str(ROOT/'tools/run_vlac.py'),'block2','--config',str(config_path),
               '--endpoint-dir',str(endpoint),'--out',str(out),'--hide-gui']
    env = dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
    for key in ('VLAC_CACHE_DIR','VLAC_HIDE_GUI','VLAC_PLATFORM','LUMERICAL_PYTHON_API'):
        env.pop(key,None)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc)
    record = {'status':'starting','started_utc':started_utc.isoformat(),
        'deadline_utc':(started_utc+timedelta(seconds=seconds)).isoformat(),
        'source_checkout':str(ROOT),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'command':command,'maximum_wall_seconds':seconds,'completion_plan_seconds':plan_seconds,
        'maximum_fde_solves':args.max_solves,
        'numerical_changes':'Measured-cost profile work allocation only; endpoint geometry, mesh and tolerances preserved.',
        'block0_certificate_applied':bool(config.solver.calibration),'supervisor_pid':os.getpid()}
    write_json(record_path,record)
    if os.name == 'nt':
        # Keep the system awake during this run; display timeout remains allowed.
        record['idle_sleep_prevention_enabled'] = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:
        with (out/'console.log').open('w',encoding='utf-8') as log:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            record.update(status='running',python_process_id=process.pid)
            write_json(record_path,record)
            while process.poll() is None:
                if time.monotonic()-started >= seconds:
                    record['status'] = 'emergency_timeout'
                    if os.name == 'nt':
                        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
                    else:
                        process.kill()
                    process.wait()
                    break
                time.sleep(5)
            code = process.returncode
        if record['status'] != 'emergency_timeout':
            record['status'] = 'completed' if code == 0 and (out/'block2_result.json').is_file() else 'failed'
        record.update(exit_code=code,elapsed_seconds=time.monotonic()-started,
            finished_utc=datetime.now(timezone.utc).isoformat())
        if record['status'] == 'completed':
            record['geometry'] = json.loads((out/'block2_geometry.json').read_text())
            record['comparison'] = json.loads((out/'block2_target_comparison.json').read_text())
            record['validation'] = json.loads((out/'block2_summary.json').read_text())
        write_json(record_path,record)
        print(json.dumps(record,indent=2),flush=True)
        return code if code else (0 if record['status']=='completed' else 2)
    finally:
        if os.name == 'nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    raise SystemExit(main())
