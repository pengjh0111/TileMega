#!/usr/bin/env python3
"""Small, independent evidence helpers for the SM90D1 diagnostic round."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/sm90d1'
PYTHON = '/workspace/llm-stack/envs/vllm/bin/python'
POLICY = ROOT / 'configs/bench/sm90d1_guard_policy.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def budgeted(command, timeout, label):
    """Reserve worst-case GPU wall time before launching, refund unused time."""
    if os.environ.get('TILEMEGA_DEVICE_INDEX') == '6':
        raise ValueError('GPU6 is excluded from SM90D1')
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('GPU work requires the scheduler and physical-card guard')
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / 'gpu-budget.json'
    token = f'{os.getpid()}-{time.time_ns()}'
    with (RUN / 'gpu-budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = read(path) if path.exists() else dict(limit_s=36000, used_s=0, active={}, history=[])
        for key, row in list(data['active'].items()):
            if not Path(f"/proc/{row['pid']}").exists():
                data['active'].pop(key)
                row.update(ended=time.time(), elapsed_s=row['reserved_s'],
                           exit_code=None, billing='interrupted reservation charged conservatively')
                data['used_s'] += row['reserved_s']
                data['history'].append(row)
        write(path, data)
        reserved = sum(row['reserved_s'] for row in data['active'].values())
        if data['used_s'] + reserved + timeout + 15 > data['limit_s']:
            raise RuntimeError('SM90D1 GPU budget cannot reserve this job')
        data['active'][token] = dict(label=label, gpu=os.environ['TILEMEGA_DEVICE_INDEX'],
                                     pid=os.getpid(), started=time.time(), reserved_s=timeout+15)
        write(path, data)
    begin = time.monotonic()
    code = 125
    try:
        process = subprocess.Popen(command)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            code = 124
        return code
    finally:
        elapsed = time.monotonic() - begin
        with (RUN / 'gpu-budget.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = read(path)
            row = data['active'].pop(token)
            row.update(ended=time.time(), elapsed_s=elapsed, exit_code=code)
            data['history'].append(row)
            data['used_s'] += elapsed
            write(path, data)
