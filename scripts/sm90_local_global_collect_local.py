#!/usr/bin/env python3
"""Run the frozen operator shortlist under the existing GPU guard."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from sm90_local_global_local import ROOT, CONFIG, sha, write
from sm90d1_common import PYTHON

def main(args):
    run=args.run.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    policy=ROOT/'configs/serving/sm90_ae/guard_policy.json'
    inventory=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
        '--query-gpu=name,uuid,pci.bus_id','--format=csv,noheader'],text=True).strip()
    if 'H800' not in inventory:raise ValueError('requires H800')
    pci=inventory.split(',')[-1].strip().lower()[-12:]
    numa=Path('/sys/bus/pci/devices',pci,'numa_node').read_text().strip()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),TILEMEGA_DEVICE_INDEX=str(args.gpu),
        TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{args.gpu}.lock',TILEMEGA_NUMA_NODE=numa,
        TILEMEGA_GPU_PID_MODE='namespace_limited',TILEMEGA_TIMING='1',
        PYTHONPATH=str(ROOT/'python'),TILEMEGA_PREFLIGHT_LOCK_TIMEOUT_S='5')
    for key in ('TILEMEGA_GPU_LOCK_HELD','TILEMEGA_DEVICE_GROUP','TILEMEGA_RUN_NONCE',
                'TILEMEGA_NUMA_CONCURRENCY_TRIAL','TILEMEGA_NUMA_CONCURRENCY_VERIFIED'):env.pop(key,None)
    state_path=out/'state.json'
    if state_path.exists():raise ValueError('retain previous collector state')
    state=dict(domain_sha256=sha(CONFIG/'local_domain.json'),gpu=args.gpu,inventory=inventory,
               policy_sha256=sha(policy),attempts=[],status='running')
    while True:
        attempt=out/f'attempt-{len(state["attempts"])}';attempt.mkdir()
        cmd=list(map(str,[PYTHON,ROOT/'scripts/sm90r6_guard.py','--policy',policy,'--out',attempt,
            '--needs-free-mib','8192','--timeout-s','1800','--','numactl',f'--cpunodebind={numa}',
            PYTHON,ROOT/'scripts/sm90_local_global_local.py','measure','--run',run,'--out',attempt/'measurements']))
        item=dict(path=str(attempt),command=cmd,started=time.time());state['attempts'].append(item);write(state_path,state)
        with (attempt/'run.log').open('w') as log:
            p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            item['pid']=p.pid;write(state_path,state);code=p.wait()
        item.update(code=code,ended=time.time());guard=json.loads((attempt/'guard_result.json').read_text())
        item['guard']=guard
        if code==75 and guard['code']==75:
            write(state_path,state);time.sleep(5);continue
        if code or guard['code']:
            state['status']='failed';write(state_path,state);raise RuntimeError(str(attempt))
        state.update(status='complete',accepted=str(attempt/'measurements'));write(state_path,state)
        print(state['accepted']);return

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',type=int,default=0)
    main(p.parse_args())
