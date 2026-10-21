#!/usr/bin/env python3
"""Keep a same-card comparison inside one guard and two fresh GPU processes."""
import argparse
from pathlib import Path
import os
import subprocess
import time

from sm90d1_common import ROOT, PYTHON, read, sha, write
from run_r16_with_device_identity import annotate


def measurement(directory):
    paths=list((directory/'job/measurements').glob('B*/measurements.json'))
    if len(paths)!=1:
        raise ValueError('one batch result is required per arm')
    value=read(paths[0])
    if value['max_tokens']!=1024 or len(value['generation_runs'])!=8:
        raise ValueError('paired development requires warmup 1, repeats 3, N=1/1024')
    return value


def validate_pair(before, after):
    if before['pid']==after['pid']:
        raise ValueError('comparison arms must run in fresh GPU processes')
    for key in ('checkpoint','batch','prompt_ids_sha256','prompt_length','max_tokens','round_index'):
        if before[key]!=after[key]:
            raise ValueError('paired input identity differs: '+key)
    if before['physical_device_identity']['cuda_uuid']!=after['physical_device_identity']['cuda_uuid']:
        raise ValueError('comparison arms changed the physical GPU')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--baseline-tag',default='reference')
    p.add_argument('--tag',required=True)
    p.add_argument('--candidate-so',type=Path)
    p.add_argument('--round',type=int,default=0)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or os.environ.get('TILEMEGA_DEVICE_INDEX')=='6':
        raise ValueError('paired runs require the shared GPU guard and an allowed card')
    a.out.mkdir(parents=True,exist_ok=True)
    path=a.out/'paired.json'
    if path.exists():
        raise FileExistsError('retain the previous paired receipt')
    record=dict(schema=1,kind='sm90r2_guarded_pair',status='running',started=time.time(),
        config=str(a.config.resolve()),config_sha256=sha(a.config),fresh_process=a.round,
        baseline_tag=a.baseline_tag,candidate_tag=a.tag,arms=[],runner_sha256=sha(Path(__file__)))
    write(path,record)
    for role,tag in (('reference',a.baseline_tag),('candidate',a.tag)):
        directory=a.out/role;directory.mkdir()
        command=[PYTHON,str(ROOT/'scripts/run_sm90r2.py'),'--config',str(a.config.resolve()),
            '--tag',tag,'--stage','measure','--round',str(a.round),'--out',str(directory/'job')]
        if role=='candidate' and a.candidate_so:
            command += ['--decode-library',str(a.candidate_so.resolve())]
        arm=dict(role=role,tag=tag,started=time.time(),command=command)
        with (directory/'stdout.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            arm['launcher_pid']=child.pid;code=child.wait()
        arm.update(ended=time.time(),exit_code=code);record['arms'].append(arm)
        write(path,record)
        if code:
            record.update(status='failed',ended=time.time());write(path,record)
            return code
        annotate(directory/'job',a.out/'physical-device-identity.json')
        arm['gpu_process_pid']=measurement(directory)['pid']
    before,after=measurement(a.out/'reference'),measurement(a.out/'candidate')
    validate_pair(before,after)
    record.update(status='completed',ended=time.time(),
        physical_uuid=before['physical_device_identity']['cuda_uuid'],
        baseline_pid=before['pid'],candidate_pid=after['pid'])
    write(path,record)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
