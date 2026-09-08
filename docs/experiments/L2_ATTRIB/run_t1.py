#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Fresh-process T1 controls. Retains per-round timing and stops on a failure."""
import argparse
import csv
import hashlib
import json
import os
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
a = p.parse_args()
repo = Path(__file__).resolve().parents[3]
out = Path(a.out).resolve()
build = Path(os.environ.get('BUILD_DIR', repo / 'build-portable'))
variants = a.variants.split(',')
phases = a.phases.split(',')
seqs = [int(s) for s in a.seqs.split(',')]
arms = {'full': [], 'nowait': ['-DTILEMEGA_UNSAFE_NO_EVENT_WAIT=1'],
        'neither': ['-DTILEMEGA_UNSAFE_NO_EVENT_WAIT=1',
                    '-DTILEMEGA_UNSAFE_NO_EVENT_NOTIFY=1'],
        'l1nosync': ['-DTILEMEGA_UNSAFE_NO_GRID_SYNC=1']}
arms = {name: arms[name] for name in a.arms.split(',')}
for sub in ['src', 'bin', 'log']:
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
    return [f'-DTILEMEGA_EVENT_LOAD_POLL={int(load)}',
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
                       '-std=c++17', '-O2', f'-arch={a.arch}', '-lineinfo',
                       f'-DTILEMEGA_EVENT_KAPPA={a.kappa}']
                cmd += [f'-I{repo / sub}' for sub in
                        ('include', 'third_party/cutlass/include',
                         'third_party/cutlass/tools/util/include', 'third_party/cutlass/test')]
                cmd += flags(variant) + extra + [str(src), str(build / 'libtilemega.a'),
                       '-L/usr/local/cuda-12.8/lib64', '-lcudart', '-o',
                       str(out / 'bin' / f'{model}_{variant}_{arm}')]
                with (out / 'log' / f'{model}_{variant}_{arm}.build').open('w') as log:
                    subprocess.run(cmd, stdout=log, stderr=log, check=True)
                print('BUILT', model, variant, arm, flush=True)

def run(model, variant, arm, seq, past, repeat, phase):
    fixture = repo / f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p{past}'
    result = subprocess.run([str(out / 'bin' / f'{model}_{variant}_{arm}'), str(fixture)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=120)
    log = out / 'log' / f'{phase}_{model}_{variant}_{arm}_{seq}_{past}_{repeat}.txt'
    log.write_text(result.stdout)
    passed = result.returncode == 0 and 'RESULT status=PASS' in result.stdout
    if arm == 'full' and not passed:
        raise RuntimeError(f'correctness failure; stop at {log}')
    timing = next((line for line in result.stdout.splitlines()
                   if line.startswith('E2E_TIME ')), None)
    if timing is None:
        raise RuntimeError(f'missing timing; stop at {log}')
    fields = dict(word.split('=', 1) for word in timing.split()[1:] if '=' in word)
    return [model, variant, arm, seq, past, repeat, int(passed),
            fields['l1_ms'], fields['l2_ms']]

for phase in ('correctness', 'attrib', 'e2e', 'matrix'):
    if phase not in phases:
        continue
    rounds = a.correctness_runs if phase in ('correctness', 'matrix') else a.runs
    with (out / f'{phase}.tsv').open('w') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(['model', 'variant', 'arm', 'seq', 'past', 'round', 'pass',
                         'l1_ms', 'l2_ms'])
        for model in ('gqa2', 'mha4'):
            for seq in ([1, 4, 128, 512, 2048] if phase == 'matrix' else seqs):
                for past in ([0, 3, 512] if phase == 'matrix' else [3]):
                    combinations = [(v, arm) for v in variants
                                    for arm in (arms if phase == 'attrib' else ['full'])]
                    for r in range(rounds):
                        order = combinations[r % len(combinations):] + combinations[:r % len(combinations)]
                        for variant, arm in order:
                            writer.writerow(run(model, variant, arm, seq, past, r, phase))
                            handle.flush()
                    print('DONE', phase, model, seq, past, rounds, flush=True)
