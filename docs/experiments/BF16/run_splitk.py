#!/usr/bin/env python3
"""T2.a independent before/after partial-storage controls, frozen generated inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--arch', default='native')
p.add_argument('--phases', default='build,correctness')
p.add_argument('--runs', type=int, default=50)
p.add_argument('--seq', type=int, default=128)
p.add_argument('--past', type=int, default=3)
p.add_argument('--jobs', type=int, default=4)
a = p.parse_args()
repo = Path(__file__).resolve().parents[3]
build = Path(os.environ.get('BUILD_DIR', repo / 'build-portable'))
out = Path(a.out).resolve()
nvcc = Path(os.environ.get('CUDACXX', '/usr/local/cuda/bin/nvcc'))
for sub in ('bin', 'src', 'plan', 'ptxas', 'logs'):
    (out / sub).mkdir(parents=True, exist_ok=True)
configs = [(model, split, precision) for model in ('gqa2', 'mha4')
           for split in (1, 2, 4, 8, 16) for precision in (0, 1)]

def tag(config):
    model, split, precision = config
    return f'{model}_k{split}_fp32{precision}'

if 'build' in a.phases.split(','):
    manifest = {'args': vars(a), 'sources': {str(f.relative_to(repo)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted((repo / 'include/tilemega/Codegen/tasks').glob('*')) if f.is_file()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for model in ('gqa2', 'mha4'):
        for split in (1, 2, 4, 8, 16):
            plan = json.loads((repo / 'docs/experiments/OWNERSHIP/plan_structured.json').read_text())
            plan['variants'][0]['uniform']['split_k'] = split
            plan_path = out / 'plan' / f'{model}_k{split}.json'
            plan_path.write_text(json.dumps(plan, indent=2) + '\n')
            subprocess.run([str(build / 'tools/tilemega-compile'),
                            str(repo / f'docs/experiments/SEQSCAN/raw/export/{model}.json'),
                            str(out / 'src' / f'{model}_k{split}.cu'), '--variants', str(plan_path)], check=True)
    def compile_one(config):
        model, split, precision = config
        cmd = [str(nvcc), '-std=c++17', '-O2', f'-arch={a.arch}', '-lineinfo',
               '-Xptxas=-v,--warn-on-spills', '--expt-relaxed-constexpr',
               f'-DTILEMEGA_FP32_PARTIALS={precision}',
               *[f'-I{repo / d}' for d in ('include', 'third_party/cutlass/include',
                                          'third_party/cutlass/tools/util/include', 'third_party/cutlass/test')],
               str(out / 'src' / f'{model}_k{split}.cu'), str(build / 'libtilemega.a'),
               f'-L{nvcc.parent.parent / "lib64"}', '-lcudart', '-o', str(out / 'bin' / tag(config))]
        with (out / 'ptxas' / f'{tag(config)}.txt').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=log, check=True)
        print('BUILT', tag(config), flush=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(compile_one, configs))

if 'correctness' in a.phases.split(','):
    (out / 'status.txt').write_text('RUNNING\n')
    try:
        with (out / 'correctness.tsv').open('w') as handle:
            writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
            writer.writerow(['model', 'split', 'fp32_partials', 'round', 'pass', 'mismatch',
                             'max_abs', 'partial_bytes', 'l05_ms', 'l1_ms', 'l2_ms'])
            for r in range(a.runs):
                for config in configs[r % len(configs):] + configs[:r % len(configs)]:
                    model, split, precision = config
                    result = subprocess.run([str(out / 'bin' / tag(config)),
                                             str(repo / f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{a.seq}_p{a.past}')],
                                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
                    log = out / 'logs' / f'{tag(config)}_r{r}.txt'
                    log.write_text(result.stdout)
                    passed = result.returncode == 0 and 'RESULT status=PASS' in result.stdout
                    match = re.search(r'l05_vs_l0_mismatch=(\d+) max_abs=(\S+)', result.stdout)
                    if not match: raise RuntimeError(f'no numerical result: {log}')
                    timing = next(x for x in result.stdout.splitlines() if x.startswith('E2E_TIME '))
                    timing = dict(x.split('=', 1) for x in timing.split()[1:] if '=' in x)
                    partials = re.search(r'E2E_PARTIALS .*total_bytes=(\d+)', result.stdout)
                    writer.writerow([model, split, precision, r, int(passed), match[1], match[2],
                                     partials[1], timing['l05_ms'], timing['l1_ms'], timing['l2_ms']])
                    handle.flush()
                    # BF16-partial baseline is the requested negative control.
                    # A failure of the *new* path is a gate, never discarded.
                    if precision and not passed:
                        raise RuntimeError(f'FP32 partial numerical gate failed; stop: {log}')
                print('ROUND', r + 1, flush=True)
        (out / 'status.txt').write_text('PASS\n')
    except BaseException:
        (out / 'status.txt').write_text('FAIL\n')
        raise
