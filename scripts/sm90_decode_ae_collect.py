#!/usr/bin/env python3
"""Collect twelve interleaved timing processes on one guarded H800."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from sm90d1_common import ROOT, PYTHON, read, sha, write


ORDER = ('A', 'B', 'C', 'E', 'B', 'C', 'E', 'A', 'C', 'E', 'A', 'B')


def collect(args):
    spec = read(args.spec)
    policy = Path(spec.get('guard_policy', ROOT / 'configs/bench/sm90r2_guard_policy.json'))
    if set(spec['arms']) != {'A', 'B', 'C', 'E'}:
        raise ValueError('only A/B/C/E may be measured; D must remain historical')
    if not spec.get('body_version_resolution'):
        raise ValueError('resolve the historical/current task-body version difference before timing')
    inventory = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-gpu=name,uuid,pci.bus_id', '--format=csv,noheader'], text=True).strip()
    if 'H800' not in inventory:
        raise ValueError('this collection requires an H800')
    node = Path('/sys/bus/pci/devices', inventory.split(',')[-1].strip().lower()[-12:], 'numa_node')
    numa = int(node.read_text())
    out = args.out.resolve();out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'state.json'
    state = read(state_path) if state_path.exists() else dict(spec=str(args.spec.resolve()),
        spec_sha256=sha(args.spec), gpu=args.gpu, inventory=inventory, order=list(ORDER), items=[],
        guard_policy=str(policy), guard_policy_sha256=sha(policy))
    if state['spec_sha256'] != sha(args.spec) or state['gpu'] != args.gpu or state['inventory'] != inventory:
        raise ValueError('resume must retain the same spec and physical GPU')
    if any(item['status'] == 'running' for item in state['items']):
        raise ValueError('inspect interrupted running attempts before resuming')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), TILEMEGA_DEVICE_INDEX=str(args.gpu),
        TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{args.gpu}.lock', TILEMEGA_NUMA_NODE=str(numa),
        TILEMEGA_GPU_PID_MODE='namespace_limited', TILEMEGA_TIMING='1', OMP_NUM_THREADS='8',
        MKL_NUM_THREADS='8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTHONPATH=str(ROOT / 'python'), TILEMEGA_PREFLIGHT_LOCK_TIMEOUT_S='5')
    for key in ('TILEMEGA_GPU_LOCK_HELD', 'TILEMEGA_DEVICE_GROUP', 'TILEMEGA_RUN_NONCE',
                'TILEMEGA_NUMA_CONCURRENCY_TRIAL', 'TILEMEGA_NUMA_CONCURRENCY_VERIFIED'):
        env.pop(key, None)
    # This experiment pins one physical card for the complete ordered matrix.
    with Path(f'/tmp/tilemega_claim{args.gpu}.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX)
        for slot, arm in enumerate(ORDER):
            if slot < len(state['items']):
                item = state['items'][slot]
                if item['arm'] != arm:raise ValueError('measurement order changed')
                if item['status'] == 'done':continue
            else:
                item = dict(slot=slot, arm=arm, status='pending', attempts=[])
                state['items'].append(item)
            while True:
                attempt = out / f'{slot:02d}-{arm}' / f'attempt-{len(item["attempts"])}'
                attempt.mkdir(parents=True)
                command = [PYTHON, ROOT / 'scripts/sm90r6_guard.py', '--policy',
                    policy, '--out', attempt,
                    '--needs-free-mib', '32768', '--timeout-s', '1800', '--',
                    'numactl', f'--cpunodebind={numa}', PYTHON, '-m',
                    'tilemega.serving.random_checkpoint', '--config',
                    ROOT / 'configs/serving/sm90d1/inputs/qwen3_1_7b',
                    '--out', attempt / 'checkpoint', '--seed', '0', '--max-gib', '32', '--',
                    PYTHON, ROOT / 'scripts/sm90_decode_ae_measure.py', '--spec', args.spec.resolve(),
                    '--arm', arm, '--fresh', str(slot), '--out', attempt / 'job']
                command = list(map(str, command))
                record = dict(directory=str(attempt), started=time.time(), command=command)
                item['attempts'].append(record);item['status'] = 'running';write(state_path, state)
                with (attempt / 'run.log').open('w') as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    write(attempt / 'process.json', dict(pid=process.pid, command=command))
                    code = process.wait()
                record.update(ended=time.time(), code=code)
                guard = read(attempt / 'guard_result.json')
                if code == 75 and guard['code'] == 75:
                    item['status'] = 'pending';write(state_path, state);time.sleep(5);continue
                if code or guard['code']:
                    item['status'] = 'failed';write(state_path, state)
                    raise RuntimeError(f'{arm} failed; inspect {attempt}')
                result_path = attempt / 'job/B1/measurements.json'
                result = read(result_path)
                if result['configuration']['decode_sha256'] != spec['arms'][arm]['decode_sha256']:
                    raise ValueError('timed binary differs from the frozen arm')
                item.update(status='done', result=str(result_path),
                    result_sha256=sha(result_path), tpot_ms=result['tpot_seconds'] * 1000)
                write(state_path, state)
                print(f'{slot + 1}/12 {arm}: {item["tpot_ms"]:.9f} ms', flush=True)
                break
    state.update(status='complete', ended=time.time(), performance_processes=12,
        correctness_processes=0, D_remeasured=False)
    write(state_path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    collect(parser.parse_args())
