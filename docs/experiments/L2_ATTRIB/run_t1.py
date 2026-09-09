#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Fresh-process T1 controls. Retains per-round timing and stops on a failure."""
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--variants', default='base,load,lines,load_lines')
p.add_argument('--phases', default='build,correctness,attrib')
p.add_argument('--runs', type=int, default=25)
p.add_argument('--correctness-runs', type=int, default=50)
p.add_argument('--seqs', default='4,128')
p.add_argument('--kappa', type=int, default=1)
p.add_argument('--arms', default='full,nowait,neither,l1nosync')
p.add_argument('--arch', default='native')
p.add_argument('--cluster-dim', type=int, default=1)
p.add_argument('--cluster-reserve', action='store_true')
p.add_argument('--resume', action='store_true', help='verify and append an interrupted process prefix')
a = p.parse_args()
repo = Path(__file__).resolve().parents[3]
out = Path(a.out).resolve()
build = Path(os.environ.get('BUILD_DIR', repo / 'build-portable'))
variants = a.variants.split(',')
phases = a.phases.split(',')
if a.resume and any(phase in phases for phase in ('build','snapshot')):
    p.error('resume cannot rebuild or replace snapshots')
seqs = [int(s) for s in a.seqs.split(',')]
arms = {'full': [], 'nowait': ['-DTILEMEGA_UNSAFE_NO_EVENT_WAIT=1'],
        'neither': ['-DTILEMEGA_UNSAFE_NO_EVENT_WAIT=1',
                    '-DTILEMEGA_UNSAFE_NO_EVENT_NOTIFY=1'],
        'l1nosync': ['-DTILEMEGA_UNSAFE_NO_GRID_SYNC=1']}
arms = {name: arms[name] for name in a.arms.split(',')}
for sub in ['src', 'bin', 'log', 'ptxas']:
    (out / sub).mkdir(parents=True, exist_ok=True)

if 'build' in phases or 'snapshot' in phases:
    inputs = [repo / 'lib/Codegen/Codegen.cpp', build / 'tools/tilemega-compile',
              build / 'libtilemega.a']
    inputs += sorted((repo / 'include/tilemega/Codegen/tasks').glob('*'))
    inputs += sorted((out / 'bin').glob('*')) if 'snapshot' in phases else []
    hashes = {str(f.relative_to(repo)) if f.is_relative_to(repo) else str(f):
              hashlib.sha256(f.read_bytes()).hexdigest()
              for f in inputs if f.is_file()}
    (out / ('snapshot.json' if 'snapshot' in phases else 'build_manifest.json')).write_text(
        json.dumps({'arguments': vars(a), 'sha256': hashes,
                    'head': subprocess.check_output(['git', '-C', str(repo),
                                                     'rev-parse', 'HEAD'], text=True).strip()},
                   indent=2) + '\n')

def flags(variant):
    load = variant in ('load', 'load_lines', 'cluster') or variant.startswith('shard')
    lines = variant in ('lines', 'load_lines', 'cluster') or variant.startswith('shard')
    sharded = variant.startswith(('shard', 'fanin')) or variant == 'cluster'
    shard = int(variant[5:]) if variant.startswith(('shard', 'fanin')) else 0
    minimum = int(variant[3:]) if variant.startswith('occ') else 1
    return [f'-DTILEMEGA_MIN_BLOCKS_PER_SM={minimum}',
            f'-DTILEMEGA_COLD_START_TIMING={int(variant == "cold")}',
            f'-DTILEMEGA_EVENT_LOAD_POLL={int(load)}',
            f'-DTILEMEGA_EVENT_SPLIT_LINES={int(lines)}',
            f'-DTILEMEGA_EVENT_SHARDED={int(sharded)}',
            f'-DTILEMEGA_EVENT_SHARDS={shard}',
            f'-DTILEMEGA_EVENT_CLUSTER_FANIN={int(variant == "cluster")}',
            f'-DTILEMEGA_EVENT_CLUSTER_RESERVE={int(a.cluster_reserve or variant == "cluster")}',
            f'-DTILEMEGA_GENERATED_CLUSTER_DIM={a.cluster_dim}']

if 'build' in phases:
    for model in ('gqa2', 'mha4'):
        src = out / 'src' / f'{model}.cu'
        subprocess.run([str(build / 'tools/tilemega-compile'),
                        str(repo / f'docs/experiments/SEQSCAN/raw/export/{model}.json'),
                        str(src), '--variants',
                        str(repo / 'docs/experiments/OWNERSHIP/plan_structured.json')],
                       check=True)
        for variant in variants:
            for arm, extra in arms.items():
                cmd = [os.environ.get('CUDACXX', '/usr/local/cuda/bin/nvcc'),
                       '-std=c++17', '-O2', f'-arch={a.arch}', '-lineinfo', '-Xptxas=-v,--warn-on-spills',
                       f'-DTILEMEGA_EVENT_KAPPA={a.kappa}']
                cmd += [f'-I{repo / sub}' for sub in
                        ('include', 'third_party/cutlass/include',
                         'third_party/cutlass/tools/util/include', 'third_party/cutlass/test')]
                cmd += flags(variant) + extra + [str(src), str(build / 'libtilemega.a'),
                       f'-L{Path(os.environ.get("CUDACXX", "/usr/local/cuda/bin/nvcc")).parent.parent / "lib64"}', '-lcudart', '-o',
                       str(out / 'bin' / f'{model}_{variant}_{arm}')]
                with (out / 'ptxas' / f'{model}_{variant}_{arm}.build').open('w') as log:
                    subprocess.run(cmd, stdout=log, stderr=log, check=True)
                print('BUILT', model, variant, arm, flush=True)

resource_columns = ['reg', 'ctas_per_sm', 'grid', 'block', 'task_smem',
                    'occupancy_smem', 'static_smem', 'regs_per_sm', 'smem_per_sm',
                    'threads_per_sm', 'l1_reg', 'l05_reg', 'l1_ctas', 'l2_ctas', 'min_blocks', 'warp_size']
schedule_columns = ['variant_stages', 'task_refs', 'waits', 'max_worker_task_refs']
timing_columns = ['cold', 'warmup', 'repeat']

def ptxas_resources(model, variant, arm):
    current = None
    values = {}
    for line in (out / 'ptxas' / f'{model}_{variant}_{arm}.build').read_text().splitlines():
        if 'Compiling entry function' in line:
            current = next((k for k in ('tilemega_l2_kernel', 'tilemega_l1_kernel',
                                       'tilemega_stage_kernel') if k in line), None)
        if current:
            entry = values.setdefault(current, {})
            m = re.search(r'Used (\d+) registers', line)
            if m: entry['reg'] = int(m[1])
            m = re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads', line)
            if m: entry.update(stores=int(m[1]), loads=int(m[2]))
    return values

def run(model, variant, arm, seq, past, repeat, phase, replay=False):
    fixture = repo / f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p{past}'
    log = out / 'log' / f'{phase}_{model}_{variant}_{arm}_{seq}_{past}_{repeat}.txt'
    if replay:
        contents = log.read_text()
        result = subprocess.CompletedProcess([], 0 if 'RESULT status=PASS' in contents else 1, contents)
    else:
        if log.exists():
            raise RuntimeError(f'unindexed log must be inspected before resume: {log}')
        result = subprocess.run([str(out / 'bin' / f'{model}_{variant}_{arm}'), str(fixture)],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=120)
        log.write_text(result.stdout)
    passed = result.returncode == 0 and 'RESULT status=PASS' in result.stdout
    if arm == 'full' and not passed:
        raise RuntimeError(f'correctness failure; stop at {log}')
    timing = next((line for line in result.stdout.splitlines()
                   if line.startswith('E2E_TIME ')), None)
    if timing is None:
        raise RuntimeError(f'missing timing; stop at {log}')
    fields = dict(word.split('=', 1) for word in timing.split()[1:] if '=' in word)
    resource = next(line for line in result.stdout.splitlines() if line.startswith('E2E_RESOURCE '))
    resource = dict(word.split('=', 1) for word in resource.split()[1:] if '=' in word)
    schedule = next(line for line in result.stdout.splitlines() if line.startswith('E2E_SCHEDULE '))
    schedule = dict(word.split('=', 1) for word in schedule.split()[1:] if '=' in word)
    if not all(c in schedule for c in schedule_columns):
        raise RuntimeError(f'missing runtime queue features; rebuild binary: {log}')
    timing_policy = next(line for line in result.stdout.splitlines() if line.startswith('E2E_TIMING '))
    timing_policy = dict(word.split('=',1) for word in timing_policy.split()[1:] if '=' in word)
    compiled = ptxas_resources(model, variant, arm)
    l2 = compiled['tilemega_l2_kernel']
    if l2['reg'] != int(resource['reg']):
        raise RuntimeError(f'ptxas/runtime register mismatch: {log}')
    block, reg = int(resource['block']), int(resource['reg'])
    warp = int(resource['warp_size'])
    # F-40's tested allocation rule: registers round to 256 per warp. The
    # budgets and actual block/warp sizes come from TargetSpec/runtime.
    by_reg = int(resource['regs_per_sm']) // (((block+warp-1)//warp) * ((reg*warp+255)//256) * 256)
    by_smem = int(resource['smem_per_sm']) // (int(resource['occupancy_smem']) + int(resource['static_smem']))
    by_threads = int(resource['threads_per_sm']) // block
    predicted = min(by_reg, by_smem, by_threads)
    if int(resource['min_blocks']) == 2 and predicted == 2 and int(resource['l2_ctas']) == 1:
        raise RuntimeError(f'F-40 predicts 2, runtime L2 has 1; stop: {log}')
    binding = '+'.join(name for name, value in [('registers', by_reg), ('smem', by_smem),
                                               ('threads', by_threads)] if value == predicted)
    return [model, variant, arm, seq, past, repeat, int(passed),
            fields['l1_ms'], fields['l2_ms'], fields['l05_ms'],
            *[resource[c] for c in resource_columns],
            *[compiled[k][c] for k in ('tilemega_stage_kernel', 'tilemega_l1_kernel',
                                       'tilemega_l2_kernel') for c in ('stores', 'loads')],
            predicted, binding, *[schedule[c] for c in schedule_columns],
            *[timing_policy[c] for c in timing_columns]]

for phase in ('correctness', 'attrib', 'e2e', 'matrix'):
    if phase not in phases:
        continue
    rounds = a.correctness_runs if phase in ('correctness', 'matrix') else a.runs
    path = out / f'{phase}.tsv'
    prior = []
    if path.exists():
        if not a.resume:
            raise RuntimeError(f'refusing to overwrite evidence: {path}')
        with path.open() as stream:
            prior = list(csv.reader(stream, delimiter='\t'))
    with path.open('a' if prior else 'w') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        header = ['model', 'variant', 'arm', 'seq', 'past', 'round', 'pass',
                         'l1_ms', 'l2_ms', 'l05_ms', *resource_columns,
                         'l05_spill_stores', 'l05_spill_loads', 'l1_spill_stores',
                         'l1_spill_loads', 'l2_spill_stores', 'l2_spill_loads',
                         'f40_ctas', 'f40_binding', *schedule_columns,
                         *timing_columns, 'execution_index']
        if prior:
            if prior.pop(0) != header:
                raise RuntimeError('resume table schema differs')
        else:
            writer.writerow(header)
        execution_index = 0
        for model in ('gqa2', 'mha4'):
            for seq in ([1, 4, 128, 512, 2048] if phase == 'matrix' else seqs):
                for past in ([0, 3, 512] if phase == 'matrix' else [3]):
                    combinations = [(v, arm)
                                    for arm in (arms if phase == 'attrib' else ['full'])
                                    for v in variants]
                    for r in range(rounds):
                        order = combinations[r % len(combinations):] + combinations[:r % len(combinations)]
                        for variant, arm in order:
                            if execution_index < len(prior):
                                expected = [str(value) for value in
                                    [*run(model,variant,arm,seq,past,r,phase,replay=True),execution_index]]
                                if prior[execution_index] != expected:
                                    raise RuntimeError(f'resume prefix disagrees with raw log at {execution_index}')
                            else:
                                writer.writerow([*run(model, variant, arm, seq, past, r, phase),
                                                 execution_index])
                            execution_index += 1
                            handle.flush()
                    print('DONE', phase, model, seq, past, rounds, flush=True)
        if len(prior) > execution_index:
            raise RuntimeError('resume prefix contains rows outside the requested experiment')
