#!/usr/bin/env python3
"""Same-session old/new 16 KiB page-loader control after fresh-process checks."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from measure_pg_ablation import LOCK, PYTHON, ROOT, manifest, sha


def main() -> None:
    work = Path('/root/r11_work/page_vector_once')
    original = Path('/root/r11_work/page_size_control/llama_B1_pages16/plan.so')
    candidate = work / 'llama_B1_pages16/plan.so'
    protocol = json.loads((work / 'fresh50/summary.json').read_text())
    if not protocol.get('complete') or protocol['passed'] != 50 or protocol['failed']:
        raise RuntimeError('50 new-process page-protocol checks must pass first')
    if protocol['cases'][0]['binary_sha256']['decode'] != sha(candidate):
        raise RuntimeError('protocol binary differs from performance candidate')
    a, b = manifest(original), manifest(candidate)
    for field in ('gemms', 'pages', 'grid', 'residency', 'kappa',
                  'attention_kv_block', 'attention_query_rows', 'sync',
                  'event_solo', 'event_red_publish', 'barrier_v2'):
        if a[field] != b[field]:
            raise RuntimeError(f'loader controls differ in {field}')
    policy = Path('/root/r11_work/pg_ablation/measurement_policy.json')
    if not policy.is_file():
        raise RuntimeError('the predeclared contamination policy is missing')
    results = []
    for name, plan in (('original', original), ('vector_once', candidate)):
        directory = work / name
        directory.mkdir(parents=True, exist_ok=True)
        command = [PYTHON, '-m', 'tilemega.serving.measure', '--model',
                   '/root/models/llama3_2_1b', '--prefill-so',
                   '/root/r11_work/r10_control/plans/llama_prefill_B1/plan.so',
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
        row = dict(name=name, binary=str(plan), sha256=sha(plan), exit_code=status,
                   seconds=time.monotonic() - started)
        results.append(row)
        if status:
            (work / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            raise RuntimeError(f'{name} measurement failed: {directory / "command.log"}')
        measurement = json.loads((directory / 'measurements.json').read_text())
        row.update(e2e_seconds=measurement['e2e_seconds'],
                   tpot_seconds=measurement['tpot_seconds'],
                   timed_tokens_identical=measurement['timed_tokens_identical'])
        (work / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
