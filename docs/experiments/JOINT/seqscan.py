#!/usr/bin/env python3
"""Re-solve selected materialized Plans at the established R4 SEQSCAN subset."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time
from measure import HERE,REPO,build,run
from search import exported,table
CASES=((1,0),(128,512),(2048,0))
def prepare(cell,m,selected_seq,seq,past,driver,arch):
    chosen=json.loads((cell/'choice.json').read_text())['choice']
    d=cell/'seqscan_plans'/f's{seq}_p{past}';d.mkdir(parents=True,exist_ok=True)
    if (d/'process.json').exists():
        old=json.loads((d/'process.json').read_text())
        if old['exit_code']==0 and (cell/'bin'/f'seqscan_s{seq}_p{past}').exists():return
        backup=d/('attempt_'+str(time.time_ns()));backup.mkdir()
        for name in ('process.json','project.log'):
            if (d/name).exists():(d/name).rename(backup/name)
    env=os.environ.copy();env.update(JOINT_PAST=str(past),JOINT_PHASE_SEQ=str(selected_seq),JOINT_PARTITION_WORKERS='1',JOINT_VALIDATE_PLACEMENT=chosen['placement'],JOINT_ONLY_PLACEMENT=chosen['placement'])
    cmd=[str(driver),str(REPO),str(exported(m)),m,str(seq),*[chosen[k] for k in ('m','n','k','stages','split','kappa','residency')],str(d.resolve()),'3']
    begin=time.time_ns()
    with (d/'project.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=7200)
    (d/'process.json').write_text(json.dumps(dict(command=cmd,exit_code=r.returncode,environment={k:v for k,v in env.items() if k.startswith('JOINT_')},started_ns=begin,elapsed_ns=time.time_ns()-begin))+'\n')
    r.check_returncode()
    selected=next(r for r in table(d/'predicted.tsv') if r['placement']==chosen['placement'])
    if selected['status']!='ok' or selected['simulated']!='1':raise RuntimeError('selected SEQSCAN group/queue validation failed')
    spec=dict(chosen,source=str(d/(chosen['placement']+'.cu')),placement_macro='0')
    if build(cell,m,f'seqscan_s{seq}_p{past}',spec,arch):raise RuntimeError('SEQSCAN compile failed')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);ap.add_argument('--raw',type=Path,default=HERE/'raw')
    ap.add_argument('--models',nargs='+',default=['gqa2','mha4']);ap.add_argument('--seqs',nargs='+',type=int,default=[4,128]);ap.add_argument('--driver',type=Path,default=Path('/tmp/r5-project-theta'));ap.add_argument('--arch',default='sm_89');ap.add_argument('--jobs',type=int,default=2);a=ap.parse_args()
    if a.action=='prepare':
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            jobs=[pool.submit(prepare,a.raw/f'{m}_s{s}',m,s,seq,past,a.driver,a.arch) for m in a.models for s in a.seqs for seq,past in CASES]
            for job in jobs:job.result()
    else:
        session=str(time.time_ns())
        for m in a.models:
            for selected_seq in a.seqs:
                cell=a.raw/f'{m}_s{selected_seq}'
                for seq,past in CASES:
                    for i in range(50):run(cell,m,seq,f'seqscan_s{seq}_p{past}',cell/'seqscan'/f's{seq}_p{past}',i,0,session,past=past)
                    print('SEQSCAN',m,selected_seq,seq,past,'50/50',flush=True)
if __name__=='__main__':main()
