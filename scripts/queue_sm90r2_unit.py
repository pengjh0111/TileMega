#!/usr/bin/env python3
"""Prepare untimed fresh-process checks for a frozen diagnostic executable."""
import argparse
from pathlib import Path

from sm90d1_common import ROOT, PYTHON, read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True, choices=(0, 1, 2, 3, 4, 5, 7))
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--header', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--queue-dir', type=Path, required=True)
    p.add_argument('--after', nargs='*', default=[])
    a = p.parse_args()
    policy = ROOT / 'configs/bench/sm90r2_guard_policy.json'
    uuid = read(policy)['idle_power_mapping_provenance']['uuid_by_gpu'][str(a.gpu)]
    node = 0 if a.gpu < 4 else 1
    pci = {0: '0f', 1: '10', 2: '41', 3: '44', 4: '86', 5: '87', 7: 'b7'}[a.gpu]
    previous, jobs = list(a.after), []
    for fresh in range(a.rounds):
        name = f'{a.name}-f{fresh}'
        jobs.append(dict(name=name, gpu=True, timing=False, priority=20, timeout_s=120,
            needs_free_mib=2048, cwd=str(ROOT), after=previous,
            env=dict(CUDA_VISIBLE_DEVICES=str(a.gpu), TILEMEGA_DEVICE_INDEX=str(a.gpu),
                TILEMEGA_NUMA_NODE=str(node), TILEMEGA_GPU_PID_MODE='namespace_limited',
                PYTHONPATH=str(ROOT / 'python'), OMP_NUM_THREADS='8'),
            command=['numactl', f'--cpunodebind={node}', PYTHON,
                str(ROOT / 'scripts/run_r16_with_device_identity.py'),
                '--guard-policy', str(policy), '--out', '{attempt_dir}/unit',
                '--expected-uuid', uuid, '--expected-pci', f'0000:{pci}:00.0',
                '--numa-node', str(node), '--allow-no-measurements', '--', str(a.binary.resolve())],
            artifact_sha256=sha(a.binary), source_sha256=sha(a.source), header_sha256=sha(a.header)))
        previous = [name]
    write(a.queue_dir / f'queue_{a.name}.json', jobs)


if __name__ == '__main__':
    main()
