"""Small resumable GPU jobs using the repository's per-card guard."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
PYTHON = os.environ.get('TILEMEGA_SQ2_PYTHON', '/workspace/llm-stack/envs/vllm/bin/python')
GPU_FIELDS = ('index', 'uuid', 'pci.bus_id', 'name', 'driver_version',
              'clocks.sm', 'clocks.max.sm', 'memory.used', 'memory.total', 'power.draw')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory():
    raw = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=' + ','.join(GPU_FIELDS), '--format=csv,noheader,nounits'],
        text=True,
    )
    cards = []
    for row in csv.reader(raw.splitlines(), skipinitialspace=True):
        card = dict(zip(GPU_FIELDS, row, strict=True))
        card['index'] = int(card['index'])
        card['pci.bus_id'] = card['pci.bus_id'][-12:].lower()
        card['numa'] = int((Path('/sys/bus/pci/devices') / card['pci.bus_id'] / 'numa_node').read_text())
        cards.append(card)
    return sorted(cards, key=lambda card: card['index'])


def run(command, directory, env=None):
    """Record each command, result, and log in a newly allocated directory."""
    directory = Path(directory)
    command = list(map(str, command))
    started = time.time()
    write(directory / 'command.json', {'argv': command, 'started': started})
    with (directory / 'run.log').open('xb') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    ended = time.time()
    write(directory / 'exit.json', {
        'returncode': result.returncode, 'started': started, 'ended': ended,
        'seconds': ended - started,
    })
    return result.returncode


def gpu_attempt(command, folder, policy, card, free_mib=32768, timeout_s=3600, timing=True):
    """Run one attempt. The caller may retry a guard rejection at item granularity."""
    folder = Path(folder)
    live = next((c for c in inventory() if c['uuid'] == card['uuid']), None)
    if live is None or any(live[k] != card[k] for k in ('index', 'pci.bus_id', 'numa', 'driver_version')):
        raise RuntimeError(f'physical GPU mapping changed for {card["uuid"]}')
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / 'gpu_before.json', live)
    index = str(card['index'])
    numa = str(card['numa'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=index, TILEMEGA_DEVICE_INDEX=index,
               TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{index}.lock',
               TILEMEGA_GPU_PID_MODE='namespace_limited', TILEMEGA_NUMA_NODE=numa,
               TILEMEGA_TIMING=str(int(timing)), PYTHONPATH=str(ROOT / 'python'),
               OMP_NUM_THREADS='8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TILEMEGA_NUMA_CONCURRENCY_TRIAL='1', TILEMEGA_NUMA_CONCURRENCY_VERIFIED='0')
    for key in ('TILEMEGA_GPU_LOCK_HELD', 'TILEMEGA_DEVICE_GROUP', 'TILEMEGA_RUN_NONCE'):
        env.pop(key, None)
    resolved = [str(x).replace('@ATTEMPT@', str(folder)) for x in command]
    bound = [
        PYTHON, ROOT / 'scripts/run_r16_with_device_identity.py',
        '--out', folder / 'job', '--guard-policy', policy, '--expected-uuid', card['uuid'],
        '--expected-pci', card['pci.bus_id'], '--numa-node', numa,
        '--allow-no-measurements', '--', *resolved,
    ]
    guarded = [
        PYTHON, ROOT / 'scripts/sm90r6_guard.py', '--policy', policy,
        '--out', folder / 'guard', '--needs-free-mib', free_mib,
        '--timeout-s', timeout_s, '--', 'numactl', f'--cpunodebind={numa}', *bound,
    ]
    code = run(guarded, folder, env)
    result = read(folder / 'guard/guard_result.json')
    assert result['code'] == code, result
    return result
