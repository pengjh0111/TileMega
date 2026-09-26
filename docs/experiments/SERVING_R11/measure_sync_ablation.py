#!/usr/bin/env python3
"""Pair the calibrated serving publication protocol against its off control."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from measure_pg_ablation import LOCK, PYTHON, ROOT, cell_sources, manifest, power_policy, sha


def main() -> None:
    protocol = Path('/root/r11_work/protocol/sync_combo/fresh50/summary.json')
    evidence = json.loads(protocol.read_text())
    if not evidence.get('complete') or evidence.get('failed') or evidence.get('processes_per_case', 0) < 50:
        raise RuntimeError('SOLO+RED+BARRIER_V2 fresh-process check has not passed')
    output = Path('/root/r11_work/sync_ablation')
    output.mkdir(parents=True, exist_ok=True)
    policy=power_policy(output)
    rows = []
    for model, batch in (('llama', 1), ('qwen3', 16)):
        plans, prefill = cell_sources(model, batch)
        control = plans['pages']
        candidate = Path(f'/root/r11_work/sync_combo/{model}_B{batch}/plan.so')
        a, b = manifest(control), manifest(candidate)
        for key in ('batch_lo', 'batch_hi', 'grid', 'residency', 'kappa',
                    'attention_kv_block', 'attention_query_rows', 'gemms', 'pages'):
            if a[key] != b[key]:
                raise RuntimeError(f'{model} B{batch}: sync arms have unequal {key}')
        if not all((b['event_solo'], b['event_red_publish'], b['barrier_v2'])):
            raise RuntimeError('the synchronization candidate did not enable the full combination')
        checkpoint = Path('/root/models') / ('llama3_2_1b' if model == 'llama' else 'qwen3_1_7b')
        prompts = ROOT / f'docs/experiments/SERVING_R10/prompts/{model}_ids.json'
        for arm, binary in (('off', control), ('combo', candidate)):
            dest = output / f'{model}_B{batch}/{arm}'
            dest.mkdir(parents=True, exist_ok=True)
            command = [PYTHON, '-m', 'tilemega.serving.measure', '--model', str(checkpoint),
                       '--prefill-so', str(prefill), '--decode-so', str(binary),
                       '--prompt-ids', str(prompts), '--batch', str(batch), '--mode', 'L2',
                       '--warmup', '1', '--repeats', '3', '--policy', str(policy),
                       '--out', str(dest)]
            start = time.monotonic()
            with LOCK.open('a') as lock, (dest / 'command.log').open('w') as log:
                fcntl.flock(lock, fcntl.LOCK_EX)
                status = subprocess.run(command, cwd=ROOT,
                    env={**os.environ, 'PYTHONPATH': str(ROOT / 'python')},
                    stdout=log, stderr=subprocess.STDOUT).returncode
            row = dict(model=model, batch=batch, arm=arm, mode='L2', binary=str(binary),
                       sha256=sha(binary), exit_code=status, seconds=time.monotonic()-start)
            if not status:
                result = json.loads((dest / 'measurements.json').read_text())
                row.update(e2e_seconds=result['e2e_seconds'],
                           tpot_seconds=result['tpot_seconds'],
                           timed_tokens_identical=result['timed_tokens_identical'])
            rows.append(row)
            (output / 'results.json').write_text(json.dumps(rows, indent=2)+'\n')
            if status:
                raise RuntimeError(f'{model} B{batch} {arm} failed: {dest / "command.log"}')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
