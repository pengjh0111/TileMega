#!/usr/bin/env python3
"""Isolate selected paged GEMM classes with the exact generated page layout."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from measure_pg_ablation import LOCK, ROOT


def main() -> None:
    plan = Path('/root/r11_work/protocol/page_v2/llama_B1_pages/plan.so')
    spec = json.loads(Path(str(plan) + '.plan.json').read_text())
    page = spec['pages']
    out = Path('/root/r11_work/paged_class_bench')
    out.mkdir(exist_ok=True)
    # Each tuple is (name, variant id, physical output N, reduction K).
    # The dimensions come from the Llama 3.2 1B checkpoint, while tile shapes
    # and all shared-memory parameters are read from the generated plan.
    stages = [('qkv', 0, 3072, 2048),
              ('gate_up', 2, 16384, 2048),
              ('down', 3, 2048, 8192),
              ('lm_head', len(spec['gemms']) - 1, 128256, 2048)]
    results = []
    for name, variant, n, k in stages:
        geometry = spec['gemms'][variant]
        executable = out / name
        definitions = {'TILE_N': geometry['tile_n'], 'TILE_K': geometry['tile_k'],
                       'STANDARD_STAGES': geometry['stages'],
                       'PAGE_BYTES': page['page_bytes'], 'PAGE_COUNT': page['pages'],
                       'WORKSPACE_OFFSET': page['workspace_offset'],
                       'POOL_OFFSET': page['pool_offset'],
                       'SHARED_BYTES': page['shared_bytes']}
        command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O3',
                   '--expt-relaxed-constexpr', '-arch=sm_89',
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
                  'geometry': geometry, 'page': page,
                  'build_command': command, 'runtime_seconds': time.monotonic() - started,
                  'measurement': (out / f'{name}.run.log').read_text().strip()}
        results.append(result)
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
