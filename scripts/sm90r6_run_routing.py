#!/usr/bin/env python3
"""Run one H2 cell in an immutable attempt directory under the shared guard."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from sm90d1_common import ROOT
from sm90r4_inputs import model_dir
import sm90r5_dispatch as dispatch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=['qwen3_30b_a3b', 'olmoe_1b_7b'])
    p.add_argument('--batch', type=int, required=True, choices=[1, 2, 4, 8, 16])
    p.add_argument('--steps', type=int, default=1024)
    p.add_argument('--sanitize', action='store_true', help='run memcheck on the routing collector')
    p.add_argument('--cell', type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 5, 7])
    p.add_argument('--out', type=Path, default=ROOT / 'runs/sm90r6/h2/attempts')
    a = p.parse_args()
    if not a.gpus or any(g not in range(8) for g in a.gpus) or (6 in a.gpus and a.gpus != [6]):
        p.error('select timing cards 0-5,7, or explicitly select functional card 6 alone')
    cell = a.cell.resolve()
    for name in ['decode.so', 'decode.export.json']:
        if not (cell / name).is_file():
            p.error(f'missing cell artifact: {cell / name}')
    a.out.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f'{a.model}-B{a.batch}-', dir=a.out.resolve()))
    if a.gpus == [6]:
        # H2 records integer routes, never timing. The physical-card guard
        # still validates the functional allowlist and owns its normal lock.
        gpu = 6
        claim = open('/tmp/tilemega_claim6.lock', 'a')
        fcntl.flock(claim, fcntl.LOCK_EX)
        dispatch.write('/tmp/tilemega_claim6.json', dict(
            pid=os.getpid(), unit=directory.name, **{'class': 'trial'}, since=time.time(), timing=False))
    else:
        gpu, claim = dispatch.claim('trial', directory.name,
                                   exclude=set(range(8)) - set(a.gpus))
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_DEVICE_INDEX=str(gpu),
            TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', TILEMEGA_TIMING='0',
            TILEMEGA_NUMA_NODE=str(dispatch.numa(gpu)), TILEMEGA_GPU_PID_MODE='namespace_limited',
            PYTHONPATH=str(ROOT / 'python'), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        for key in ['TILEMEGA_GPU_LOCK_HELD', 'TILEMEGA_DEVICE_GROUP', 'TILEMEGA_RUN_NONCE']:
            env.pop(key, None)
        collect = [sys.executable, '-u', str(ROOT / 'scripts/sm90r6_moe_routing.py'),
            '--model', a.model, '--batch', str(a.batch), '--steps', str(a.steps),
            '--decode', str(cell / 'decode.so'), '--bridge', str(cell / 'decode.export.json'),
            '--out', str(directory)]
        if a.sanitize:
            collect = ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '86',
                       '--log-file', str(directory / 'memcheck.log'), *collect]
        checkpoint = [sys.executable, '-u', str(ROOT / 'python/tilemega/serving/random_checkpoint.py'),
            '--config', str(model_dir(a.model)), '--out', str(directory / 'checkpoint'),
            '--seed', '0', '--max-gib', '64', '--', *collect]
        command = [sys.executable, '-m', 'tilemega.bench.gpu_guard',
            '--policy', str(dispatch.POLICY), '--out', str(directory / 'guard'),
            '--needs-free-mib', '70000', '--timeout-s', '7200', '--',
            'numactl', f'--cpunodebind={dispatch.numa(gpu)}', *checkpoint]
        (directory / 'command.json').write_text(json.dumps(dict(command=command, gpu=gpu,
            model=a.model, batch=a.batch, steps=a.steps, timing=False), indent=2) + '\n')
        print(json.dumps({'directory': str(directory), 'gpu': gpu}), flush=True)
        with (directory / 'run.log').open('w') as log:
            code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        (directory / 'result.json').write_text(json.dumps({'returncode': code}) + '\n')
        print(json.dumps({'directory': str(directory), 'returncode': code}), flush=True)
        return code
    finally:
        claim.close()


if __name__ == '__main__':
    raise SystemExit(main())
