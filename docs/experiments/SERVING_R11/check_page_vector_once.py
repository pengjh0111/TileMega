#!/usr/bin/env python3
"""Full-request C-1/C-2 check of the optimized multi-page Llama B1 plan."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess

from measure_pg_ablation import LOCK, PYTHON, ROOT


def main() -> None:
    work = Path('/root/r11_work/page_vector_once')
    out = work / 'correctness'
    out.mkdir(parents=True, exist_ok=True)
    model = '/root/models/llama3_2_1b'
    prompts = ROOT / 'docs/experiments/SERVING_R10/prompts/llama_ids.json'
    commands = [
        ('c2', [PYTHON, '-m', 'tilemega.serving.check_modes',
                '--model', model, '--prefill-so',
                '/root/r11_work/r10_control/plans/llama_prefill_B1/plan.so',
                '--decode-so', str(work / 'llama_B1_pages16/plan.so'),
                '--prompt-ids', str(prompts), '--batch', '1', '--steps', '1024',
                '--out', str(out)]),
        ('c1', [PYTHON, '-m', 'tilemega.serving.hf_check',
                '--model', model, '--prompt-ids', str(prompts),
                '--generated', str(out / 'tokens_L2.json'),
                '--out', str(out / 'hf_check.json'), '--skip-free-greedy']),
    ]
    results = []
    for label, command in commands:
        with LOCK.open('a') as lock, (out / f'{label}.log').open('w') as log:
            fcntl.flock(lock, fcntl.LOCK_EX)
            status = subprocess.run(command, cwd=ROOT,
                env={**os.environ, 'PYTHONPATH': str(ROOT / 'python')},
                stdout=log, stderr=subprocess.STDOUT).returncode
        results.append(dict(check=label, exit_code=status, command=command))
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        if status:
            raise RuntimeError(f'{label} failed: {out / f"{label}.log"}')


if __name__ == '__main__':
    main()
