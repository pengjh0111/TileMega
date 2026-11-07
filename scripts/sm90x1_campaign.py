#!/usr/bin/env python3
"""Resume the authorized SM90X1 campaign; stop at gates and preserve all attempts."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time

from sm90d1_common import ROOT, PYTHON, read, write
from sm90x1 import RUN, cells, directory


def command(action, phase=0, extra=()):
    path=RUN/'campaign-logs'/f'{time.time_ns()}-{phase}-{action}.log'
    path.parent.mkdir(parents=True,exist_ok=True)
    argv=[PYTHON,str(ROOT/'scripts/sm90x1.py'),action,'--phase',str(phase),*extra]
    write(RUN/'campaign-state.json',dict(status='running',phase=phase,action=action,log=str(path),at=time.time()))
    with path.open('w') as log:
        code=subprocess.call(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    print(f'phase={phase} action={action} code={code} log={path}',flush=True)
    return code


def stop(reason, status='stopped'):
    command('tables')
    command('tables',1,('--cell','all'))
    write(RUN/'campaign-state.json',dict(status=status,reason=reason,at=time.time()))
    print(reason,flush=True)
    return 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-pids',nargs='*',type=int,default=[])
    a=p.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Existing Phase 0 parents own complete units and may chain corr -> final.
    # Wait for those parents rather than duplicate their units during the handoff.
    def started(pid):
        try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
        except (OSError,IndexError):return None
    parents={pid:started(pid) for pid in a.wait_pids if started(pid) is not None}
    write(RUN/'campaign-state.json',dict(status='waiting_phase0',parents=parents,at=time.time()))
    deadline=time.monotonic()+6*3600
    while any(started(pid)==token for pid,token in parents.items()):
        if time.monotonic()>deadline:
            return stop('Phase 0 still waits on existing guarded units after six hours','blocked')
        time.sleep(30)
    for c in cells('pilot'):
        d=directory(c)/'phase0'
        if not (d/'W/binding.json').exists():
            if command('prepare',0,('--cell','all')) or command('build',0,('--cell',c['cell'])):
                return stop('Phase 0 build or frozen-input validation failed')
        for action in ['corr','time']:
            if (d/action/'result.json').exists():continue
            controls=list((d/action).glob('try*/scheduler/control.json'))
            if any(read(path).get('timing_paused') for path in controls):
                return stop('Phase 0 guard exhausted its six-hour idle window','blocked')
            if command(action,0,('--cell',c['cell'])):
                return stop('Phase 0 '+action+' failed or remained blocked')
    if command('tables'):
        return stop('Phase 0 TPOT deviates by more than 2%, or validation is incomplete')
    c=cells('pilot')[0]
    if not (directory(c)/'phase0/trace/unit.json').exists():
        if command('trace',0,('--cell',c['cell'])):
            return stop('Phase 0 trace-overhead unit failed or remained blocked')
    k=4
    for attempt in range(2):
        for action,extra in [('prepare',()),('build',()),('trace',()),('analyze',()),('arms',()),
                             ('build',('--arms-only',)),('corr',()),('time',()),('tables',('--cell','all'))]:
            if command(action,1,('--k',str(k),*extra)):
                return stop('Pilot '+action+' failed; no Phase 2 measurements were launched')
        projection=read(RUN/'pilot-gate.json')
        if projection['continue']:break
        if k==3:return stop('The k=3 pilot projection exceeds 600 builds or 40 GPU-hours')
        k=3
        write(RUN/'k-reduction.json',dict(evidence='verified gate exceeded; inferred projection',
                                         reason=projection,k=3,at=time.time()))
        for c in cells('pilot'):
            path=directory(c)/'phase1'
            path.rename(path.with_name('phase1-k4-'+str(time.time_ns())))
    for action,extra in [('prepare',()),('build',()),('trace',()),('analyze',()),('arms',()),
                         ('build',('--arms-only',)),('corr',()),('time',()),('tables',('--cell','all'))]:
        # Individual failures are retained by the CLI; continue healthy cells through all stages.
        command(action,2,('--k',str(k),*extra))
    complete=read(RUN/'main-complete.json')['complete']
    write(RUN/'campaign-state.json',dict(status='complete' if complete else 'finished_with_failures',
                                        phases=[0,1,2],k=k,at=time.time()))
    return 0 if complete else 1


if __name__=='__main__':
    raise SystemExit(main())
