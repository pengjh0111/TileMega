#!/usr/bin/env python3
"""Collect a complete AB1a family in three fresh processes on one guarded card."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from sm90d1_common import ROOT,sha,write
from sm90r4_inputs import model_dir
import sm90r5_dispatch as dispatch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--prompts',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--development',action='store_true')
    a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());a.out.mkdir(parents=True,exist_ok=False)
    if not a.development and a.reference is None:
        request=json.loads((Path(manifest['input_hashes']).parent/'cells.json').read_text())
        if len(request)!=1:raise ValueError('ambiguous probe input reference')
        coordinate=Path(request[0]['seed']);a.reference=coordinate.with_name('decode.so')
        identity=json.loads(Path(str(a.reference)+'.identity.json').read_text())
        if (identity['binary_sha256']!=sha(a.reference) or
                identity['plan']['coordinates_sha256']!=sha(coordinate)):
            raise ValueError('probe input reference does not match the selected A0 coordinate')
    inputs={str(path.resolve()):sha(path) for path in
            (a.manifest,a.prompts,Path(__file__),ROOT/'scripts/sm90r6_run_probe.py')}
    if a.reference:inputs[str(a.reference.resolve())]=sha(a.reference)
    write(a.out/'inputs.json',inputs)
    accepted=[];measurements=[]
    processes=5 if a.development else 3
    for part in manifest['parts']:
        if sha(part['binary'])!=part['binary_sha256']:raise ValueError('probe binary changed')
    gpu,claim=dispatch.claim('trial',str(a.out),timing=True)
    try:
        for process in range(processes):
            while True:
                directory=Path(tempfile.mkdtemp(prefix=f'p{process}-',dir=a.out.resolve()))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TILEMEGA_DEVICE_INDEX=str(gpu),
                    TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock',TILEMEGA_TIMING='1',
                    TILEMEGA_NUMA_NODE=str(dispatch.numa(gpu)),TILEMEGA_GPU_PID_MODE='namespace_limited',
                    TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',PYTHONPATH=str(ROOT/'python'),
                    TILEMEGA_TRACE_LAUNCHES='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                for key in ('TILEMEGA_GPU_LOCK_HELD','TILEMEGA_DEVICE_GROUP','TILEMEGA_RUN_NONCE'):
                    env.pop(key,None)
                collect=[sys.executable,'-u',str(ROOT/'scripts/sm90r6_run_probe.py'),
                    '--manifest',str(a.manifest.resolve()),'--batch',str(a.batch),
                    '--prompts',str(a.prompts.resolve()),'--out',str(directory/'trace')]
                if a.development:collect.append('--development')
                if a.reference:collect+=['--reference',str(a.reference.resolve())]
                checkpoint=[sys.executable,'-u',str(ROOT/'python/tilemega/serving/random_checkpoint.py'),
                    '--config',str(model_dir(a.model)),'--out',str(directory/'checkpoint'),
                    '--seed','0','--max-gib','64','--',*collect]
                command=[sys.executable,'-m','tilemega.bench.gpu_guard','--policy',str(dispatch.POLICY),
                    '--out',str(directory/'guard'),'--needs-free-mib','70000','--timeout-s','7200','--',
                    'numactl',f'--cpunodebind={dispatch.numa(gpu)}',*checkpoint]
                write(directory/'command.json',dict(command=command,gpu=gpu,
                    manifest_sha256=sha(a.manifest),worker_sha256=sha(ROOT/'scripts/sm90r6_run_probe.py')))
                with (directory/'run.log').open('w') as log:
                    code=subprocess.call(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                write(directory/'returncode.json',dict(returncode=code))
                if any(sha(path)!=digest for path,digest in inputs.items()):
                    raise ValueError('probe collection inputs changed during measurement')
                if code==75:
                    # No accepted timing exists yet, so reselection cannot mix
                    # cards within a family. Foreign work keeps its own locks.
                    if not accepted:
                        previous=gpu;claim.close()
                        gpu,claim=dispatch.claim('trial',str(a.out),timing=True,exclude=[previous])
                        write(directory/'reselected.json',dict(previous_gpu=previous,gpu=gpu,
                            reason='guard rejected before the first accepted process'))
                    continue
                if code:return code
                guard=json.loads((directory/'guard/guard_result.json').read_text())
                result=json.loads((directory/'trace/result.json').read_text())
                if guard['code'] or not result['passed']:raise ValueError('unaccepted probe process')
                if accepted and result['checkpoint']!=accepted[0]['checkpoint']:
                    raise ValueError('probe checkpoint changed across fresh processes')
                rows=json.loads((directory/'trace/measurements.json').read_text())
                for row in rows:
                    if row['raw_sha256']!=sha(row['raw']) or row['gpu']!=gpu:
                        raise ValueError('probe raw ledger or physical card changed')
                    row.update(guard_code=0,guard=str(directory/'guard/guard_result.json'))
                measurements.extend(rows)
                accepted.append(dict(directory=str(directory),process=process,
                    gpu=gpu,parts=len(manifest['parts']),checkpoint=result['checkpoint']))
                (a.out/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in measurements))
                write(a.out/'result.json',dict(passed=len(accepted)==processes,
                    processes=accepted,development=a.development,timing=True,
                    same_card=True,ownership='provisional_container_pid_namespace'))
                print(json.dumps(dict(process=process,parts=len(manifest['parts']),passed=True,gpu=gpu)),flush=True)
                break
    finally:
        claim.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
