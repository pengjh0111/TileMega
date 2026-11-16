#!/usr/bin/env python3
"""Keep DNN4 measurement semantics with one guarded protocol reservation."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import dnn4_measure as measurement
from dnn4_cohort import ROOT, archive_sources, save, sha

ORIGINAL_RUN = subprocess.run


def run(command, *args, **kwargs):
    if not ('tilemega.bench.scheduler' in command and '--queue-dir' in command):
        return ORIGINAL_RUN(command,*args,**kwargs)
    queue=Path(command[command.index('--queue-dir')+1])
    original=queue/'queue_measure.json'
    if not original.exists():
        return ORIGINAL_RUN(command,*args,**kwargs)
    steps=json.loads(original.read_text())
    out=queue.parent
    frozen=json.loads((out/'inputs.json').read_text())
    for path in (Path(__file__).resolve(),ROOT/'scripts/dnn4_cohort.py'):
        frozen[str(path)]=sha(path)
    plan=out/'cohort_plan.json'
    archive_sources(out,frozen)
    save(plan,dict(steps=steps,frozen=frozen))
    cohort_queue=out/'queue_cohort';cohort_queue.mkdir()
    name='dnn4-measurement-cohort'
    job=dict(name=name,gpu=True,timing=True,timeout_s=sum(s['timeout_s'] for s in steps),
        needs_free_mib=max(s['needs_free_mib'] for s in steps),cwd=str(ROOT),env=steps[0]['env'],
        command=[sys.executable,str(ROOT/'scripts/dnn4_cohort.py'),'--plan',str(plan),
                 '--out','{attempt_dir}/cohort'])
    save(cohort_queue/'queue_cohort.json',[job])
    changed=list(command);changed[changed.index('--queue-dir')+1]=str(cohort_queue)
    completed=ORIGINAL_RUN(changed,*args,**kwargs)
    state_path=out/'scheduler/state.json'
    state=json.loads(state_path.read_text());save(out/'scheduler/cohort_state.json',state)
    reservation=state[name]
    if reservation['status']!='done':
        save(state_path,{s['name']:dict(status='failed',reservation=reservation) for s in steps})
        return completed
    folder=Path(reservation['out'])
    guard=folder/'guard_result.json'
    accepted=json.loads(guard.read_text())
    result=json.loads((folder/'cohort/result.json').read_text())
    if accepted['code'] or result['status']!='passed' or len(result['processes'])!=len(steps):
        raise ValueError('cohort lacks complete guarded measurements')
    synthetic={}
    for step,process in zip(steps,result['processes']):
        if step['name']!=process['name'] or process['exit_code']:
            raise ValueError('cohort execution order differs')
        child=Path(process['out'])
        measured=json.loads((child/'measure/measurements.json').read_text())
        if measured['pid']!=process['pid']:
            raise ValueError('measurement process identity differs from fresh child')
        shutil.copyfile(guard,child/'guard_result.json')
        synthetic[step['name']]=dict(status='done',out=str(child),pid=process['pid'],
                                   cohort_guard=str(guard),cohort_size=len(steps))
    save(state_path,synthetic)
    return completed


if __name__=='__main__':
    measurement.subprocess.run=run
    raise SystemExit(measurement.main())
