#!/usr/bin/env python3
"""Prepare three guarded fresh-process timing jobs for one validated SM90R1 cell."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.bench.queue_check import check
from r16_measurement_validation import file_sha, native_identity
from run_sm90r1_final_cell import validate_numeric, validate_pair


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2, 3, 4, 7), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config_path, out = args.config.resolve(), args.out.resolve()
    config, raw = read(config_path), ROOT / 'runs/sm90r1'
    llm = 'isl' in config
    progress = read(raw / ('final-llm' if llm else 'final-dnn') / 'progress.json')
    cells = dict(progress['settled'])
    revisions = raw / ('final-llm' if llm else 'final-dnn') / 'revisions.json'
    if revisions.exists():
        cells.update(read(revisions)['settled'])
    if llm:
        matches = [value for value in cells.values()
                   if Path(value['config']).resolve() == config_path]
        if len(matches) != 1:
            parser.error('configuration is not the current validated selection')
        cell = matches[0]
        validate_numeric(Path(cell['numeric_receipt']), file_sha(config_path), validate_pair(config),
                         config['batch'], 256 if config['model'] == 'qwen3_30b_a3b' else 1024)
        source_job = cell['rows'][0]['job']
    else:
        cell = cells[config['model']]
        selected = ROOT / cell.get('config', f"configs/dnn/sm90r1/{config['model']}_b1.json")
        assert selected.resolve() == config_path
        identity, gate = native_identity(ROOT / config['library']), read(cell['gate'])
        assert gate['passed'] and identity['artifact_id'] == gate['artifact_id'] == config['artifact_id']
        source_job = cell['jobs'][0]
    originals = [job for gpu in range(8) for path in (raw / f'queue/gpu{gpu}').glob('queue_*.json')
                 for job in read(path) if job['name'] == source_job]
    assert len(originals) == 1, 'source queue is missing or ambiguous'
    physical = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid,pci.bus_id',
                                        '--format=csv,noheader'], text=True).strip().split(',')
    gpu_uuid, bus = [value.strip() for value in physical]
    assert gpu_uuid.startswith('GPU-')
    bus = bus.lower()
    domain, bus_id, function = bus.split(':')
    pci = f'{int(domain, 16):04x}:{bus_id}:{function}'
    node = int(Path(f'/sys/bus/pci/devices/{pci}/numa_node').read_text())
    assert node >= 0, 'NUMA node is unavailable'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    jobs = []
    for index in range(3):
        job = copy.deepcopy(originals[0])
        job.update(name=f"sm90r1_replay_{config['model']}_B{config['batch']}_{stamp}_r{index}", priority=index)
        job.pop('after', None)
        job['env'].update(CUDA_VISIBLE_DEVICES=str(args.gpu), TILEMEGA_DEVICE_INDEX=str(args.gpu),
                          TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{args.gpu}.lock', TILEMEGA_NUMA_NODE=str(node))
        command = job['command']
        command[1] = f'--cpunodebind={node}'
        for flag, value in [('--expected-uuid', gpu_uuid), ('--expected-pci', pci),
                            ('--numa-node', str(node))]:
            command[command.index(flag) + 1] = value
        if '--round-index' in command:
            command[command.index('--round-index') + 1] = str(index)
        else:
            command += ['--round-index', str(index)]
        jobs.append(job)
    assert check([], jobs, {})['pass_check']
    out.mkdir(parents=True, exist_ok=False)
    queue = out / 'queue'
    queue.mkdir()
    (queue / 'queue_replay.json').write_text(json.dumps(jobs, indent=2) + '\n')
    if config['model'] == 'qwen3_30b_a3b':
        registry = ROOT / 'runs/r16/native-moe-shared-lease-v1/declared-jobs.json'
        checkpoint = ROOT / config['checkpoint']['path']
        with (checkpoint / 'declarations.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = read(registry)
            value['jobs'] = list(dict.fromkeys(value['jobs'] + [job['name'] for job in jobs]))
            temporary = registry.with_suffix('.pending')
            temporary.write_text(json.dumps(value, indent=2) + '\n')
            temporary.replace(registry)
    command = ['env', f'CUDA_VISIBLE_DEVICES={args.gpu}', f'TILEMEGA_DEVICE_INDEX={args.gpu}',
               f'TILEMEGA_NUMA_NODE={node}', 'TILEMEGA_GPU_PID_MODE=namespace_limited',
               f'PYTHONPATH={ROOT / "python"}', originals[0]['command'][2],
               '-m', 'tilemega.bench.scheduler', '--queue-dir', str(queue), '--out', str(out / 'scheduler'),
               '--policy', str(ROOT / 'configs/bench/guard_policy.json'), '--deadline-hours', '24']
    (out / 'run.sh').write_text('#!/bin/bash\nset -eu\ncd ' + shlex.quote(str(ROOT)) + '\nexec ' +
                                shlex.join(command) + '\n')
    print('Prepared three fresh processes; numerical proofs and frozen artifacts are reused.')
    print('bash ' + shlex.quote(str(out / 'run.sh')))


if __name__ == '__main__':
    main()
