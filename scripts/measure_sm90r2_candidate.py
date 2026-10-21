#!/usr/bin/env python3
"""Measure a native solver candidate through fresh-process correctness and guarded pairs."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

from sm90d1_common import ROOT, PYTHON, read, sha, write
from run_sm90r2_pair import measurement, validate_pair


def require_correctness(c1, c2, batch):
    if (not c1.get('pass') or c1.get('batch') != batch or c1.get('max_new_tokens', 0) < 256
            or not c2.get('pass') or not c2.get('same_binary') or len(c2.get('checks', [])) != 2
            or not all(v.get('pass') for v in c2['checks'])
            or {v.get('decode_loop') for v in c2['checks']} != {0, 1}):
        raise ValueError('solver candidate lacks required native C-1/C-2 coverage')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--gpu', type=int, choices=(0, 1, 2, 3, 4, 7), required=True)
    p.add_argument('--so', type=Path, required=True)
    p.add_argument('--native-config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out = a.out.resolve();a.so = a.so.resolve();a.config = a.config.resolve()
    if a.out.exists() or not a.out.is_relative_to(ROOT / 'runs/sm90r2'):
        raise ValueError('choose a new solver measurement directory under runs/sm90r2')
    identity = read(str(a.so) + '.identity.json')
    plan = read(str(a.so) + '.plan.json')
    if (sha(a.so) != identity['binary_sha256'] or identity['trace'] or
            plan['native_dense'] != read(a.native_config) or plan['native_origin'] != 'solver'):
        raise ValueError('solver artifact or selected coordinates are inconsistent')
    c = read(a.config);rounds = 3 if c['model'] == 'llama3_1_8b' else 1
    policy = ROOT / 'configs/bench/sm90r2_guard_policy.json'
    uuid = read(policy)['idle_power_mapping_provenance']['uuid_by_gpu'][str(a.gpu)]
    node = 0 if a.gpu < 4 else 1
    pci = {0: '0f', 1: '10', 2: '41', 3: '44', 4: '86', 7: 'b7'}[a.gpu]
    queue = a.out / 'queue';scheduler = a.out / 'scheduler/gpu'
    jobs = []
    def job(name, timing, command, directory, after):
        wrapper = ['numactl', f'--cpunodebind={node}', PYTHON,
            str(ROOT / 'scripts/run_r16_with_device_identity.py'),
            '--guard-policy', str(policy), '--out', '{attempt_dir}/' + directory,
            '--expected-uuid', uuid, '--expected-pci', f'0000:{pci}:00.0',
            '--numa-node', str(node), '--allow-no-measurements', '--', *command]
        return dict(name=name, gpu=True, timing=timing, timeout_s=3600,
            needs_free_mib=32768, after=after, cwd=str(ROOT),
            env=dict(CUDA_VISIBLE_DEVICES=str(a.gpu), TILEMEGA_DEVICE_INDEX=str(a.gpu),
                TILEMEGA_NUMA_NODE=str(node), TILEMEGA_GPU_PID_MODE='namespace_limited',
                PYTHONPATH=str(ROOT / 'python'), OMP_NUM_THREADS='8'), command=wrapper)
    command = [PYTHON, str(ROOT / 'scripts/run_sm90r2.py'), '--config', str(a.config),
        '--tag', 'solver', '--decode-library', str(a.so), '--stage', 'correctness',
        '--positions', '256', '--round', '0', '--out', '{attempt_dir}/job']
    jobs.append(job('correctness', False, command, 'job', []))
    for fresh in range(rounds):
        command = [PYTHON, str(ROOT / 'scripts/run_sm90r2_pair.py'), '--config', str(a.config),
            '--tag', 'solver', '--candidate-so', str(a.so), '--round', str(fresh),
            '--out', '{attempt_dir}/pair']
        jobs.append(job(f'pair-f{fresh}', True, command, 'pair',
                        ['correctness' if not fresh else f'pair-f{fresh-1}']))
    write(queue / 'queue_candidate.json', jobs)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), TILEMEGA_DEVICE_INDEX=str(a.gpu),
        TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{a.gpu}.lock', PYTHONPATH=str(ROOT / 'python'))
    env.pop('TILEMEGA_GPU_LOCK_HELD', None)
    command = [PYTHON, '-m', 'tilemega.bench.scheduler', '--queue-dir', str(queue),
               '--out', str(scheduler), '--policy', str(policy)]
    with (a.out / 'scheduler.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    states = read(scheduler / 'state.json')
    if any(s['status'] != 'done' for s in states.values()):
        write(a.out / 'result.json', dict(accepted=False, states=states, binary_sha256=sha(a.so)))
        return 1
    check = Path(states['correctness']['out'])
    c1 = read(check / 'job/measurements/c1.json');c2 = read(check / 'job/measurements/c2.json')
    if read(check / 'guard_result.json')['code']:
        raise ValueError('native correctness guard rejected the process')
    require_correctness(c1, c2, c['batch'])
    # r16_run exits nonzero on either failure; retain both detailed receipts.
    timings, references, receipts = [], [], []
    ownership = True
    for fresh in range(rounds):
        row = states[f'pair-f{fresh}'];folder = Path(row['out']);pair = folder / 'pair'
        guard = read(folder / 'guard_result.json')
        if guard['code'] or read(pair / 'paired.json')['status'] != 'completed':
            raise ValueError('solver pair lacks accepted complete guard evidence')
        before, after = measurement(pair / 'reference'), measurement(pair / 'candidate')
        validate_pair(before, after)
        if read(pair / 'candidate/job/selected-config.json')['decode_sha256'] != sha(a.so):
            raise ValueError('solver measured another binary')
        timings.append(after['tpot_seconds'] * 1e9);references.append(before['tpot_seconds'] * 1e9)
        ownership &= guard['ownership_verified']
        receipts.append(dict(pair=str(pair / 'paired.json'), guard_sha256=sha(folder / 'guard_result.json')))
    write(a.out / 'result.json', dict(accepted=True, binary_sha256=sha(a.so),
        native_coordinates_sha256=sha(a.native_config), reference_config_sha256=sha(a.config),
        c1_pass=True, c2_pass=True, c1=c1, c2=c2, processes=rounds,
        tpot_ns=statistics.median(timings), process_medians_ns=timings,
        reference_ns=statistics.median(references), ownership_verified=ownership,
        process_range_pct=(max(timings)-min(timings))/statistics.median(timings)*100,
        paired_receipts=receipts, callback_sha256=sha(__file__)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
