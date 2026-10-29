#!/usr/bin/env python3
"""Queue the SM90R4 constants probe: five fresh guarded processes per section."""
import argparse
import json
from pathlib import Path

from sm90d1_common import ROOT, PYTHON, read, sha, write

POLICY = ROOT / 'configs/bench/sm90r2_guard_policy.json'
PCI = {0: '0f', 1: '10', 2: '41', 3: '44'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--section', default='', help='probe flag, e.g. --attention-only')
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--processes', type=int, default=5)
    p.add_argument('--timeout-s', type=int, default=3600)
    p.add_argument('--queue-dir', type=Path, required=True)
    a = p.parse_args()
    uuid = read(POLICY)['idle_power_mapping_provenance']['uuid_by_gpu'][str(a.gpu)]
    jobs, previous = [], []
    for fresh in range(a.processes):
        name = f'{a.name}-f{fresh}'
        command = ['numactl', '--cpunodebind=0', PYTHON, str(ROOT / 'scripts/run_r16_with_device_identity.py'),
                   '--guard-policy', str(POLICY), '--out', '{attempt_dir}/unit', '--expected-uuid', uuid,
                   '--expected-pci', f'0000:{PCI[a.gpu]}:00.0', '--numa-node', '0',
                   '--allow-no-measurements', '--', str(a.binary.resolve())]
        if a.section:
            command.append(a.section)
        jobs.append(dict(name=name, gpu=True, timing=True, priority=20, timeout_s=a.timeout_s,
                         needs_free_mib=8192, cwd=str(ROOT), after=list(previous),
                         env=dict(CUDA_VISIBLE_DEVICES=str(a.gpu), TILEMEGA_DEVICE_INDEX=str(a.gpu),
                                  TILEMEGA_NUMA_NODE='0', TILEMEGA_GPU_PID_MODE='namespace_limited',
                                  TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{a.gpu}.lock',
                                  PYTHONPATH=str(ROOT / 'python'), OMP_NUM_THREADS='8'),
                         command=command, artifact_sha256=sha(a.binary),
                         section=a.section or 'full'))
        previous = [name]
    a.queue_dir.mkdir(parents=True, exist_ok=True)
    write(a.queue_dir / f'queue_{a.name}.json', jobs)
    print(json.dumps(dict(queue=str(a.queue_dir / f'queue_{a.name}.json'), jobs=len(jobs))))


if __name__ == '__main__':
    main()
