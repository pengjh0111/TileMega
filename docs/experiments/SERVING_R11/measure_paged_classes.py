#!/usr/bin/env python3
"""Isolate selected paged GEMM classes with the exact generated page layout."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from measure_pg_ablation import LOCK, ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--model-config', type=Path, required=True,
                        help='HF config.json that defines the physical GEMM dimensions')
    parser.add_argument('--out', type=Path,
                        default=Path('/root/r11_work/paged_class_bench'))
    parser.add_argument('--classes', nargs='+',
                        choices=['qkv', 'gate_up', 'down', 'lm_head'],
                        default=['qkv', 'gate_up', 'down', 'lm_head'])
    args = parser.parse_args()
    plan = args.plan
    spec = json.loads(Path(str(plan) + '.plan.json').read_text())
    model = json.loads(args.model_config.read_text())
    page = spec['pages']
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    # Each tuple is (name, representative GEMM index, physical N, reduction K).
    # The config supplies physical dimensions; the plan supplies tiles and
    # shared memory. The standard and paged arms therefore use one geometry.
    hidden = model['hidden_size']
    intermediate = model['intermediate_size']
    head_dim = model['head_dim']
    qkv_width = (model['num_attention_heads'] + 2 * model['num_key_value_heads']) * head_dim
    stages = [('qkv', 0, qkv_width, hidden),
              ('gate_up', 2, 2 * intermediate, hidden),
              ('down', 3, hidden, intermediate),
              ('lm_head', len(spec['gemms']) - 1, model['vocab_size'], hidden)]
    results = []
    for name, variant, n, k in stages:
        if name not in args.classes:
            continue
        geometry = spec['gemms'][variant]
        executable = out / name
        definitions = {'TILE_N': geometry['tile_n'], 'TILE_K': geometry['tile_k'],
                       'STANDARD_STAGES': geometry['stages'],
                       'PAGE_BYTES': page['page_bytes'], 'PAGE_COUNT': page['pages'],
                       'WORKSPACE_OFFSET': page['workspace_offset'],
                       'POOL_OFFSET': page['pool_offset'],
                       'SHARED_BYTES': page['shared_bytes']}
        target = json.loads(Path(spec['runtime_target']).read_text())
        arch = f"sm_{target['sm_major']}{target['sm_minor']}"
        compiler = os.getenv('CUDACXX') or shutil.which('nvcc')
        if not compiler:
            raise RuntimeError('nvcc not found; set CUDACXX')
        command = [compiler, '-std=c++17', '-O3',
                   '--expt-relaxed-constexpr', f'-arch={arch}',
                   '-I' + str(ROOT / 'include'),
                   '-I' + str(ROOT / 'third_party/cutlass/include')]
        command += [f'-DBENCH_{key}={value}' for key, value in definitions.items()]
        command += [str(ROOT / 'docs/experiments/SERVING_R11/paged_class_bench.cu'),
                    '-o', str(executable)]
        with (out / f'{name}.build.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           check=True, cwd=ROOT)
        started = time.monotonic()
        with LOCK.open('a') as lock, (out / f'{name}.run.log').open('w') as log:
            fcntl.flock(lock, fcntl.LOCK_EX)
            subprocess.run([str(executable), str(n), str(k)], stdout=log,
                           stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        result = {'name': name, 'variant': variant, 'n': n, 'k': k,
                  'geometry': geometry, 'page': page, 'model_config': str(args.model_config),
                  'build_command': command, 'runtime_seconds': time.monotonic() - started,
                  'measurement': (out / f'{name}.run.log').read_text().strip()}
        results.append(result)
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
