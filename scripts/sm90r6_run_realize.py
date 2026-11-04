#!/usr/bin/env python3
"""Run fresh guarded P5 token-equivalence/C-2 processes at fixed theta."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from sm90d1_common import ROOT, sha, write
from sm90r4_inputs import model_dir
import sm90r5_dispatch as dispatch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--batch', type=int, choices=[1, 2, 4, 8, 16], required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--processes', type=int, default=5)
    p.add_argument('--steps', type=int, default=1024)
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 7])
    a = p.parse_args()
    if a.processes < 5 or not a.gpus or any(g not in (0, 1, 2, 3, 4, 5, 7) for g in a.gpus):
        p.error('use at least five processes and cards 0-5,7')
    if not 1 <= a.steps <= 1024:
        p.error('steps must lie in [1,1024]')
    artifacts = {}
    for name in ('reference', 'candidate'):
        binary = getattr(a, name).resolve()
        identity = json.loads(Path(str(binary) + '.identity.json').read_text())
        if identity['binary_sha256'] != sha(binary):
            raise ValueError('binary differs from its identity')
        artifacts[name] = dict(path=str(binary), sha256=sha(binary))
    a.out.mkdir(parents=True, exist_ok=False)
    accepted = []
    attempt = 0
    while len(accepted) < a.processes:
        directory = Path(tempfile.mkdtemp(prefix=f'p{len(accepted)}-try{attempt}-', dir=a.out.resolve()))
        attempt += 1
        gpu, claim = dispatch.claim('trial', directory.name, exclude=set(range(8)) - set(a.gpus))
        try:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_DEVICE_INDEX=str(gpu),
                TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', TILEMEGA_TIMING='0',
                TILEMEGA_NUMA_NODE=str(dispatch.numa(gpu)), TILEMEGA_GPU_PID_MODE='namespace_limited',
                TILEMEGA_NUMA_CONCURRENCY_TRIAL='1', PYTHONPATH=str(ROOT / 'python'),
                OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            for key in ('TILEMEGA_GPU_LOCK_HELD', 'TILEMEGA_DEVICE_GROUP', 'TILEMEGA_RUN_NONCE'):
                env.pop(key, None)
            collect = [sys.executable, '-u', str(ROOT / 'scripts/sm90r6_realize_check.py'),
                '--reference', artifacts['reference']['path'], '--candidate', artifacts['candidate']['path'],
                '--prompts', str(a.prompts.resolve()), '--batch', str(a.batch), '--steps', str(a.steps),
                '--out', str(directory / 'checks')]
            checkpoint = [sys.executable, '-u', str(ROOT / 'python/tilemega/serving/random_checkpoint.py'),
                '--config', str(model_dir(a.model)), '--out', str(directory / 'checkpoint'),
                '--seed', '0', '--max-gib', '64', '--', *collect]
            command = [sys.executable, '-m', 'tilemega.bench.gpu_guard', '--policy', str(dispatch.POLICY),
                '--out', str(directory / 'guard'), '--needs-free-mib', '70000', '--timeout-s', '7200',
                '--', 'numactl', f'--cpunodebind={dispatch.numa(gpu)}', *checkpoint]
            write(directory / 'command.json', dict(command=command, gpu=gpu, artifacts=artifacts))
            print(json.dumps(dict(directory=str(directory), gpu=gpu)), flush=True)
            with (directory / 'run.log').open('w') as log:
                code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(directory / 'result.json', dict(returncode=code))
            if code == 75:
                continue
            if code:
                return code
            result = json.loads((directory / 'checks/result.json').read_text())
            if not result['passed'] or json.loads((directory / 'guard/guard_result.json').read_text())['code']:
                raise ValueError('unaccepted realization result')
            if accepted and (result['checkpoint'] != accepted[0]['checkpoint'] or
                    result['checks'][0]['tokens_sha256'] != accepted[0]['tokens_sha256']):
                raise ValueError('fresh process results differ')
            accepted.append(dict(directory=str(directory), gpu=gpu, checkpoint=result['checkpoint'],
                tokens_sha256=result['checks'][0]['tokens_sha256']))
            write(a.out / 'result.json', dict(passed=len(accepted) == a.processes, processes=accepted,
                model=a.model, batch=a.batch, steps=a.steps, artifacts=artifacts,
                evidence='verified', timing=False, ownership='provisional_container_pid_namespace'))
        finally:
            claim.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
