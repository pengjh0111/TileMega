#!/usr/bin/env python3
"""Run full-request C-1/C-2 diagnostics on fixed-geometry page plans."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess

from measure_pg_ablation import LOCK, PYTHON, ROOT


def main() -> None:
    root = Path('/root/r11_work')
    output = root / 'fixed_page_correctness'
    output.mkdir(exist_ok=True)
    results = []
    for model in ('llama', 'qwen3'):
        checkpoint = Path('/root/models') / ('llama3_2_1b' if model == 'llama' else 'qwen3_1_7b')
        prompts = ROOT / f'docs/experiments/SERVING_R10/prompts/{model}_ids.json'
        for batch in (1, 16):
            cell = output / f'{model}_B{batch}'
            cell.mkdir(exist_ok=True)
            prefill = root / f'r10_control/plans/{model}_prefill_B{batch}/plan.so'
            decode = root / f'protocol/page_v2/{model}_B{batch}_pages/plan.so'
            commands = [
                ('c2', [PYTHON, '-m', 'tilemega.serving.check_modes',
                        '--model', str(checkpoint), '--prefill-so', str(prefill),
                        '--decode-so', str(decode), '--prompt-ids', str(prompts),
                        '--batch', str(batch), '--steps', '1024', '--out', str(cell)]),
                ('c1', [PYTHON, '-m', 'tilemega.serving.hf_check',
                        '--model', str(checkpoint), '--prompt-ids', str(prompts),
                        '--generated', str(cell / 'tokens_L2.json'),
                        '--out', str(cell / 'hf_check.json'), '--skip-free-greedy']),
            ]
            for label, command in commands:
                with LOCK.open('a') as lock, (cell / f'{label}.log').open('w') as log:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    status = subprocess.run(command, cwd=ROOT,
                        env={**os.environ, 'PYTHONPATH': str(ROOT / 'python')},
                        stdout=log, stderr=subprocess.STDOUT).returncode
                row = {'model': model, 'batch': batch, 'check': label,
                       'exit_code': status, 'command': command,
                       'result': str(cell / ('mode_check.json' if label == 'c2' else 'hf_check.json'))}
                results.append(row)
                (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
                if status:
                    raise RuntimeError(f'{model} B{batch} {label} failed: {cell / f"{label}.log"}')


if __name__ == '__main__':
    main()
