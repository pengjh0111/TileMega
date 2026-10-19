#!/usr/bin/env python3
"""Write complete GPU job inputs before starting the existing event scheduler."""
import argparse
from pathlib import Path

from sm90d1_common import ROOT, RUN, PYTHON, POLICY, read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True, choices=(0, 1, 2, 3, 4, 5, 7))
    p.add_argument('--name', required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--tag', default='reference')
    p.add_argument('--paired-tag')
    p.add_argument('--candidate-loop', type=int, choices=(0, 1))
    p.add_argument('--candidate-mode', choices=('L1', 'L2'))
    p.add_argument('--stage', choices=('measure', 'overhead', 'correctness', 'trace', 'trace-control', 'primitive', 'resources', 'solver'), required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--after', nargs='*', default=[])
    p.add_argument('--priority', type=int, default=10)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--extra', nargs='*', default=[])
    a = p.parse_args()
    if a.tag.startswith('G2'):
        a.priority = max(a.priority, 70 if a.stage == 'correctness' else 80)
    elif a.tag.startswith('G1'):
        a.priority = max(a.priority, 40 if a.stage == 'correctness' else 50)
    if a.stage == 'measure' and a.tag != 'reference' and a.paired_tag is None:
        a.paired_tag = a.tag
    if a.gpu == 5 and a.stage in ('measure', 'trace', 'primitive'):
        p.error('GPU5 retains its functional-only restriction')
    node = 0 if a.gpu < 4 else 1
    uuid = read(POLICY)['idle_power_mapping_provenance']['uuid_by_gpu'][str(a.gpu)]
    pci = {0:'0f',1:'10',2:'41',3:'44',4:'86',5:'87',7:'b7'}[a.gpu]
    jobs = []
    previous = list(a.after)
    sequence = [(fresh, tag) for fresh in range(a.rounds) for tag in
                (('reference', a.paired_tag) if a.paired_tag else (a.tag,))]
    for fresh, tag in sequence:
        name = f'{a.name}-{tag}-f{fresh}' if a.paired_tag else f'{a.name}-f{fresh}'
        command = ['numactl', f'--cpunodebind={node}', PYTHON,
                   str(ROOT / 'scripts/run_r16_with_device_identity.py'),
                   '--guard-policy', str(POLICY), '--out', '{attempt_dir}',
                   '--expected-uuid', uuid, '--expected-pci', f'0000:{pci}:00.0',
                   '--numa-node', str(node), '--allow-no-measurements', '--',
                   PYTHON, str(ROOT / 'scripts/run_sm90d1.py'), '--stage', a.stage,
                   '--tag', tag, '--round', str(fresh), '--timeout', str(a.timeout),
                   '--out', '{attempt_dir}/job']
        if a.config:
            command += ['--config', str(a.config.resolve())]
        command += a.extra
        if tag != 'reference':
            if a.candidate_loop is not None:
                command += ['--loop', str(a.candidate_loop)]
            if a.candidate_mode is not None:
                command += ['--mode', a.candidate_mode]
        jobs.append(dict(name=name, gpu=True, timing=a.stage not in ('correctness', 'resources', 'solver'),
            priority=a.priority, timeout_s=a.timeout + 60, needs_free_mib=32768,
            cwd=str(ROOT), after=previous,
            env=dict(CUDA_VISIBLE_DEVICES=str(a.gpu), TILEMEGA_DEVICE_INDEX=str(a.gpu),
                TILEMEGA_NUMA_NODE=str(node), TILEMEGA_GPU_PID_MODE='namespace_limited',
                PYTHONPATH=str(ROOT / 'python'), OMP_NUM_THREADS='8'), command=command))
        if a.stage == 'trace':
            jobs.append(dict(name=name+'-analyze', gpu=False, timing=False,
                priority=200, timeout_s=600, cwd=str(ROOT), after=[name],
                command=['python3', str(ROOT / 'scripts/collect_sm90d1_trace.py'),
                    '--state', str(RUN / f'scheduler/gpu{a.gpu}/state.json'), '--job', name,
                    '--config', str(a.config.resolve()), '--tag', tag]))
        previous = [name]
    write(RUN / f'queues/gpu{a.gpu}/queue_{a.name}.json', jobs)


if __name__ == '__main__':
    main()
