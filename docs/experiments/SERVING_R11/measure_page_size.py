#!/usr/bin/env python3
"""Fixed-geometry 8/16 KiB page comparison after the protocol and PG arms."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from measure_pg_ablation import LOCK, PYTHON, ROOT, manifest, sha


def main() -> None:
    root = Path('/root/r11_work')
    baseline = root / 'protocol/page_v2/llama_B1_pages/plan.so'
    out = root / 'page_size_control'
    out.mkdir(parents=True, exist_ok=True)
    source = root / 'r10_control/plans/llama_decode_B1/plan.so.top1.mlir'
    target = root / 'prefetch_target.json'
    binary = out / 'llama_B1_pages16/plan.so'
    binary.parent.mkdir(parents=True, exist_ok=True)
    compile_command = [str(root / 'build/tools/tilemega'), 'compile', str(source), str(binary),
                       '--serving', 'decode', '--batch', '1', '--past-range', '64:1086',
                       '--capacity', '1088', '--emit', 'serving', '--pg', 'pages',
                       '--page-bytes', '16384', '--runtime-target', str(target)]
    (out / 'compile_command.json').write_text(json.dumps(compile_command, indent=2)+'\n')
    with (out / 'compile.log').open('w') as log:
        status = subprocess.run(compile_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    if status:
        raise RuntimeError(f'16 KiB compile failed ({status}): {out / "compile.log"}')
    a, b = manifest(baseline), manifest(binary)
    if b['pages']['page_bytes'] != 16384 or a['pages']['page_bytes'] != 8192:
        raise RuntimeError('page-size controls do not have 8 and 16 KiB pages')
    for field in ('gemms', 'grid', 'residency', 'kappa', 'attention_kv_block',
                  'attention_query_rows', 'sync', 'event_solo', 'event_red_publish',
                  'barrier_v2'):
        if a[field] != b[field]:
            raise RuntimeError(f'page-size controls differ in {field}')
    policy = root / 'pg_ablation/measurement_policy.json'
    if not policy.is_file():
        raise RuntimeError('PG control policy must be recorded before this experiment')
    results = []
    for name, plan in (('8k', baseline), ('16k', binary)):
        directory = out / name
        directory.mkdir(exist_ok=True)
        command = [PYTHON, '-m', 'tilemega.serving.measure', '--model',
                   '/root/models/llama3_2_1b', '--prefill-so',
                   str(root / 'r10_control/plans/llama_prefill_B1/plan.so'),
                   '--decode-so', str(plan), '--prompt-ids',
                   str(ROOT / 'docs/experiments/SERVING_R10/prompts/llama_ids.json'),
                   '--batch', '1', '--mode', 'L2', '--warmup', '1', '--repeats', '3',
                   '--policy', str(policy), '--out', str(directory)]
        started = time.monotonic()
        with LOCK.open('a') as lock, (directory / 'command.log').open('w') as log:
            fcntl.flock(lock, fcntl.LOCK_EX)
            status = subprocess.run(command, cwd=ROOT, env={**os.environ,
                'PYTHONPATH': str(ROOT / 'python')}, stdout=log,
                stderr=subprocess.STDOUT).returncode
        item = dict(name=name, binary=str(plan), sha256=sha(plan), exit_code=status,
                    seconds=time.monotonic()-started)
        if status:
            results.append(item)
            (out / 'results.json').write_text(json.dumps(results, indent=2)+'\n')
            raise RuntimeError(f'page-size measurement failed: {directory / "command.log"}')
        measurement = json.loads((directory / 'measurements.json').read_text())
        item.update(e2e_seconds=measurement['e2e_seconds'],
                    tpot_seconds=measurement['tpot_seconds'],
                    timed_tokens_identical=measurement['timed_tokens_identical'])
        results.append(item)
        (out / 'results.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
