#!/usr/bin/env python3
"""Prepare immutable paired development jobs for the shared GPU scheduler."""
import argparse
from pathlib import Path

from sm90d1_common import ROOT, PYTHON, read, write

RUN = ROOT / 'runs/sm90r2'
POLICY = ROOT / 'configs/bench/sm90r2_guard_policy.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True, choices=(0, 1, 2, 3, 4, 5, 7))
    p.add_argument('--name', required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--stage', choices=('correctness', 'measure', 'trace', 'trace-control'), required=True)
    p.add_argument('--rounds', type=int, required=True)
    p.add_argument('--after', nargs='*', default=[])
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--queue-dir', type=Path)
    a = p.parse_args()
    if a.gpu == 5 and a.stage != 'correctness':
        p.error('GPU5 is functional-only')
    node = 0 if a.gpu < 4 else 1
    uuid = read(POLICY)['idle_power_mapping_provenance']['uuid_by_gpu'][str(a.gpu)]
    pci = {0: '0f', 1: '10', 2: '41', 3: '44', 4: '86', 5: '87', 7: 'b7'}[a.gpu]
    jobs, previous = [], list(a.after)
    for fresh in range(a.rounds):
        for tag in (('reference', a.tag) if a.stage == 'measure' else (a.tag,)):
            name = f'{a.name}-{tag}-f{fresh}'
            command = ['numactl', f'--cpunodebind={node}', PYTHON,
                       str(ROOT / 'scripts/run_r16_with_device_identity.py'),
                       '--guard-policy', str(POLICY), '--out', '{attempt_dir}/job',
                       '--expected-uuid', uuid, '--expected-pci', f'0000:{pci}:00.0',
                       '--numa-node', str(node), '--allow-no-measurements', '--',
                       PYTHON, str(ROOT / 'scripts/run_sm90r2.py'),
                       '--config', str(a.config.resolve()), '--tag', tag,
                       '--stage', a.stage, '--round', str(fresh), '--out', '{attempt_dir}/job']
            jobs.append(dict(name=name, gpu=True, timing=a.stage != 'correctness',
                priority=5 if a.stage == 'correctness' else 10,
                timeout_s=a.timeout, needs_free_mib=32768, cwd=str(ROOT), after=previous,
                env=dict(CUDA_VISIBLE_DEVICES=str(a.gpu), TILEMEGA_DEVICE_INDEX=str(a.gpu),
                    TILEMEGA_NUMA_NODE=str(node), TILEMEGA_GPU_PID_MODE='namespace_limited',
                    PYTHONPATH=str(ROOT / 'python'), OMP_NUM_THREADS='8'), command=command))
            previous = [name]
    directory=a.queue_dir or RUN / f'queues/gpu{a.gpu}'
    write(directory / f'queue_{a.name}.json', jobs)


if __name__ == '__main__':
    main()
