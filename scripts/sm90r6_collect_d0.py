#!/usr/bin/env python3
"""Retain three guarded D0 traces and their uninstrumented token references."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from sm90d1_common import ROOT,read,sha,write
from sm90r4_inputs import model_dir
import sm90r5_dispatch as dispatch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--decode',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--prompts',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    inputs={str(path.resolve()):sha(path) for path in
            (a.decode,a.reference,a.prompts,Path(__file__),ROOT/'scripts/sm90r4_ledger.py')}
    write(a.out/'inputs.json',inputs);accepted=[]
    while len(accepted)<3:
        trial=Path(tempfile.mkdtemp(prefix=f'p{len(accepted)}-',dir=a.out.resolve()))
        gpu,claim=dispatch.claim('trial',str(trial),timing=True)
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TILEMEGA_DEVICE_INDEX=str(gpu),
                TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock',TILEMEGA_TIMING='1',
                TILEMEGA_NUMA_NODE=str(dispatch.numa(gpu)),TILEMEGA_GPU_PID_MODE='namespace_limited',
                TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',PYTHONPATH=str(ROOT/'python'),
                OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            for key in ('TILEMEGA_GPU_LOCK_HELD','TILEMEGA_DEVICE_GROUP','TILEMEGA_RUN_NONCE'):env.pop(key,None)
            worker=[sys.executable,ROOT/'scripts/sm90r4_ledger.py','run','--prefill','prompt',
                '--decode',a.decode.resolve(),'--reference',a.reference.resolve(),
                '--prompt-ids',a.prompts.resolve(),'--batch',a.batch,'--tokens','520','--out',trial/'trace']
            checkpoint=[sys.executable,ROOT/'python/tilemega/serving/random_checkpoint.py',
                '--config',model_dir(a.model),'--out',trial/'checkpoint','--seed','0','--max-gib','64','--',*worker]
            command=[sys.executable,'-m','tilemega.bench.gpu_guard','--policy',dispatch.POLICY,
                '--out',trial/'guard','--needs-free-mib','70000','--timeout-s','7200','--',
                'numactl',f'--cpunodebind={dispatch.numa(gpu)}',*checkpoint]
            write(trial/'command.json',list(map(str,command)))
            with (trial/'run.log').open('w') as log:
                code=subprocess.call(list(map(str,command)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(trial/'returncode.json',dict(returncode=code,gpu=gpu))
            if any(sha(path)!=digest for path,digest in inputs.items()):raise ValueError('D0 inputs changed during collection')
            if code==75:continue
            if code:return code
            result=read(trial/'trace/run.json')
            if read(trial/'guard/guard_result.json')['code'] or not result['full_token_equivalence']:
                raise ValueError('unaccepted D0 process')
            if accepted and result['checkpoint']!=accepted[0]['checkpoint']:raise ValueError('D0 checkpoint changed')
            accepted.append(dict(directory=str(trial),gpu=gpu,checkpoint=result['checkpoint'],
                raw_sha256=sha(trial/'trace/stage_trace.tsv'),overhead_pct=result['overhead_pct']))
            write(a.out/'result.json',dict(passed=len(accepted)==3,processes=accepted,
                diagnostic_only=True,performance_eligible=False))
            print(f'{a.model} B{a.batch}: {len(accepted)}/3 D0 processes',flush=True)
        finally:claim.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
