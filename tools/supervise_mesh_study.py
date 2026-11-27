"""Supervise one explicitly budgeted mesh-study child and its MODE tree."""
import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','endpoint-dir','out','session'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--max-solves',type=int,default=300)
    parser.add_argument('--max-seconds',type=int,default=18000)
    args = parser.parse_args()
    session,output = args.session.resolve(),args.out.resolve()
    if (session/'execution.json').exists() or output.exists():
        raise FileExistsError('Use fresh run/session identities; no budget resets.')
    if not 300<args.max_seconds<=18000:
        raise ValueError('The authorized diagnostic ceiling is five hours, including five minutes for cleanup.')
    session.mkdir(parents=True,exist_ok=True)
    started = datetime.now(timezone.utc)
    monotonic = time.monotonic()
    command = [sys.executable,'-u',str(ROOT/'tools/run_vlac.py'),'mesh-study',
        '--config',str(args.config.resolve()),'--endpoint-dir',str(args.endpoint_dir.resolve()),
        '--out',str(output),'--max-solves',str(args.max_solves),
        '--max-seconds',str(args.max_seconds-300)]
    tracked = [ROOT/p for p in ('src/vlac/mesh_study.py','src/vlac/native_backend.py',
                               'src/vlac/native/vlac_core.lsf','tools/run_vlac.py')]
    record = dict(status='starting',started_utc=started.isoformat(),supervisor_pid=os.getpid(),
        deadline_utc=(started+timedelta(seconds=args.max_seconds)).isoformat(),
        max_fresh_solves=args.max_solves,max_seconds=args.max_seconds,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
        command=command,output=str(output),block2_launched=False)
    def save():
        path = session/'execution.json.tmp'
        path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        path.replace(session/'execution.json')
    awake = False
    if os.name=='nt':
        awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    record['idle_sleep_prevention_enabled'] = awake
    save()
    environment = dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
    for key in ('VLAC_CACHE_DIR','VLAC_HIDE_GUI','VLAC_PLATFORM','LUMERICAL_PYTHON_API'):
        environment.pop(key,None)
    try:
        with (session/'console.log').open('w',encoding='utf-8') as log:
            child = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
            record.update(status='running',child_pid=child.pid)
            save()
            timeout = False
            while child.poll() is None:
                if time.monotonic()-monotonic >= args.max_seconds:
                    timeout = True
                    # Only the child created here and descendants; never by name.
                    if os.name=='nt':
                        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                    else:
                        child.kill()
                    child.wait()
                    break
                time.sleep(2)
        record.update(status='timeout' if timeout else 'finished',exit_code=child.returncode,
            finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-monotonic)
        report_path = output/'mesh_study.json'
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding='utf-8'))
            record.update(study_status=report['status'],fresh_fde_solves=report['fresh_fde_solves'],
                numerical_convergence_passed=report['numerical_convergence_passed'],
                execution_completed=report['execution_completed'])
        record['source_unchanged_during_run'] = all(hashlib.sha256(p.read_bytes()).hexdigest()==
            record['source_sha256'][str(p.relative_to(ROOT))] for p in tracked)
        save()
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__': main()
