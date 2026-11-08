#!/usr/bin/env python3
"""Collect three fresh P-TUNE processes under one guarded reservation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from dnn4_cohort import ROOT, archive_sources, save, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('library','export','bridge','inputs','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--cpu-set',required=True)
    args=p.parse_args()
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    gpu=str(args.device)
    env=dict(CUDA_VISIBLE_DEVICES=gpu,TILEMEGA_DEVICE_INDEX=gpu,
        TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock',TILEMEGA_NUMA_NODE='0',
        TILEMEGA_GPU_PID_MODE='namespace_limited',TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
        PYTHONPATH=str(ROOT/'python'))
    steps=[]
    for index in range(3):
        command=['taskset','-c',args.cpu_set,str(ROOT/'scripts/dnn4_guard_python.py'),
            str(ROOT/'scripts/dnn4_marks.py'),'--execute','--batch',str(args.batch),
            '--warmup','20','--samples','50','--device',gpu,'--out','{attempt_dir}/trace']
        for name in ('library','export','bridge','inputs'):
            command+=['--'+name,str(getattr(args,name).resolve())]
        steps.append(dict(name=f'process-{index}',command=command,cwd=str(ROOT),env=env,
                          timeout_s=1200))
    paths=[Path(__file__).resolve(),ROOT/'scripts/dnn4_cohort.py',ROOT/'scripts/dnn4_marks.py',
           args.library,args.bridge,args.inputs,args.export/'exported_program.pt2',
           args.export/'checkpoint/model.safetensors',
           *sorted((ROOT/'python/tilemega/dnn').glob('*.py'))]
    paths.extend(Path(str(args.library)+suffix) for suffix in
                 ('.cu','.plan.json','.identity.json','.ptxas.log'))
    frozen={str(path.resolve()):sha(path) for path in paths}
    archive_sources(args.out,frozen)
    plan=args.out/'plan.json';save(plan,dict(steps=steps,frozen=frozen))
    queue=args.out/'queue';queue.mkdir()
    name='dnn4-trace-cohort'
    save(queue/'queue_cohort.json',[dict(name=name,gpu=True,timing=True,timeout_s=3600,
        needs_free_mib=8192,cwd=str(ROOT),env=env,
        command=[sys.executable,str(ROOT/'scripts/dnn4_cohort.py'),'--plan',str(plan),
                 '--out','{attempt_dir}/cohort'])])
    with (args.out/'dispatch.log').open('w') as log:
        subprocess.run([str(ROOT/'scripts/dnn4_guard_python.py'),'-m','tilemega.bench.scheduler',
            '--queue-dir',str(queue),'--out',str(args.out/'scheduler'),'--policy',
            str(ROOT/'configs/bench/sm90r2_guard_policy.json'),'--deadline-hours','6'],
            cwd=ROOT,env=dict(os.environ,**env),stdout=log,stderr=subprocess.STDOUT,check=True)
    state=json.loads((args.out/'scheduler/state.json').read_text())[name]
    if state['status']!='done':
        raise ValueError('three-process trace reservation failed')
    folder=Path(state['out'])
    guard=json.loads((folder/'guard_result.json').read_text())
    cohort=json.loads((folder/'cohort/result.json').read_text())
    if guard['code'] or cohort['status']!='passed' or len(cohort['processes'])!=3:
        raise ValueError('trace cohort did not complete three accepted processes')
    traces=[str(Path(process['out'])/'trace') for process in cohort['processes']]
    save(args.out/'result.json',dict(status='passed',processes=cohort['processes'],traces=traces,
        guard=str(folder/'guard_result.json'),qualification=guard['acceptance']))
    print(json.dumps(dict(status='passed',processes=3,out=str(args.out))))


if __name__=='__main__':
    main()
