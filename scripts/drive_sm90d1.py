#!/usr/bin/env python3
"""Complete independent CPU builds, then publish prewritten guarded GPU jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import select
import subprocess

from build_sm90d1 import build
from sm90d1_common import ROOT, RUN, read, write

DENSE = ['qwen3_0_6b_b1', 'qwen3_0_6b_b16', 'llama3_2_1b_b1',
         'llama3_1_8b_b1', 'llama3_1_8b_b16']
B1 = ['qwen3_0_6b_b1', 'llama3_2_1b_b1', 'llama3_1_8b_b1']
COMPILER = RUN / 'compiler-v2/tilemega'


def config(cell):
    return ROOT / f'configs/serving/sm90d1/reference/{cell}.json'


def gpu(cell):
    return 0 if cell.startswith('llama3_1_8b') else 7


def queue(cell, name, stage, tag, rounds=1, after=(), priority=20, paired=None):
    command = ['python3', str(ROOT / 'scripts/queue_sm90d1.py'), '--gpu', str(gpu(cell)),
               '--name', name, '--config', str(config(cell)), '--stage', stage,
               '--tag', tag, '--rounds', str(rounds), '--priority', str(priority)]
    if after:
        command += ['--after', *after]
    if paired:
        command += ['--paired-tag', paired]
    subprocess.run(command, cwd=ROOT, check=True)


def derive(cell, tag, parent, definitions, stage='trace', rounds=1):
    source_tag = 'reference2' if parent == 'reference' else parent
    binary = RUN / 'build' / source_tag / f"{read(config(cell))['model']}-B{read(config(cell))['batch']}" / 'decode.so'
    output = RUN / 'build' / tag / binary.parent.name / 'decode.so'
    command = ['python3', str(ROOT / 'scripts/derive_sm90d1.py'),
               '--input', str(binary), '--out', str(output), '--lineinfo']
    for value in definitions:
        command += ['--define', value]
    with (RUN / f'{tag}-{cell}.log').open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        return dict(cell=cell, tag=tag, status='diagnostic_build_failed')
    priority = 5 if tag == 'full2' else 6 if tag == 'stage2' else 20
    queue(cell, f'{tag}-{cell}', stage, tag, rounds=rounds, priority=priority)
    return dict(cell=cell, tag=tag, status='queued')


def arm(cell, tag, changes, correctness=False):
    result = build(config(cell), COMPILER, tag, changes, 'decode')
    if result.get('status') in ('build_failed', 'build_timeout'):
        return dict(cell=cell, tag=tag, status=result['status'])
    after = []
    if correctness:
        name = f'{tag}-{cell}-corr'
        queue(cell, name, 'correctness', tag, priority=12)
        after = [name + '-f0']
    queue(cell, f'{tag}-{cell}-pair', 'measure', tag, rounds=3, after=after, paired=tag)
    # Stage envelopes are derived from precisely this arm's uninstrumented plan.
    return derive(cell, tag+'-stage', tag,
                  ['TILEMEGA_TRACE_STAGE=1', 'TILEMEGA_TRACE_STEP=1'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-pid', type=int)
    a = p.parse_args()
    if a.wait_pid:
        try:
            descriptor = os.pidfd_open(a.wait_pid)
        except ProcessLookupError:
            descriptor = None
        if descriptor is not None:
            select.select([descriptor], [], [])
            os.close(descriptor)
    jobs = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for cell in DENSE:
            jobs.append(pool.submit(derive, cell, 'full2', 'reference',
                ['TILEMEGA_TRACE_STAGE=1', 'TILEMEGA_TRACE_STEP=1', 'TILEMEGA_TRACE_TASK=1',
                 'TILEMEGA_TRACE_TASK_SAMPLE=1', 'TILEMEGA_TRACE_LOCAL_CLOCK=1']))
            jobs.append(pool.submit(derive, cell, 'stage2', 'reference',
                                    ['TILEMEGA_TRACE_STAGE=1', 'TILEMEGA_TRACE_STEP=1']))
        for cell in B1:
            jobs.append(pool.submit(arm, cell, 'P1-2',
                {'--pg':'l2', '--l2-prefetch-worker-bytes':65536, '--prefetch-history':0}))
            jobs.append(pool.submit(arm, cell, 'LA-2', {'--nonpaged-la':1}))
        for tag, mask in [('LAmerge-2',8), ('LAcombine-2',4), ('LAempty-2',0)]:
            jobs.append(pool.submit(arm, B1[0], tag,
                {'--nonpaged-la':1, '--diagnostic-la-classes':mask}, True))
        jobs.append(pool.submit(arm, B1[0], 'LArelease-2', {'--nonpaged-la':1, '--la-cta-release':1}))
        for cell in B1[:2]:
            jobs.append(pool.submit(arm, cell, 'PG-2', {'--pg':'pages', '--page-bytes':16384,
                '--lookahead-bytes':0, '--paged-la':1, '--paged-la-splitk':1}))
        for mode in (1,2,3):
            jobs.append(pool.submit(derive, B1[0], f'diag{mode}-2', 'reference',
                ['TILEMEGA_TRACE_STAGE=1', f'TILEMEGA_TRACE_STAGE_DIAGNOSTIC={mode}'], 'overhead'))
        for completed in as_completed(jobs):
            try:
                result = completed.result()
            except Exception as error:
                result = dict(status='cpu_driver_error', error=str(error))
            with (RUN / 'cpu-completions.jsonl').open('a') as stream:
                stream.write(json.dumps(result)+'\n')
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
