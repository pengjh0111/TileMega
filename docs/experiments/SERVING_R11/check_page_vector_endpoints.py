#!/usr/bin/env python3
"""Fresh-process page-protocol checks at the remaining B=1/16 endpoints."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from measure_pg_ablation import PYTHON, ROOT


def main() -> None:
    work = Path('/root/r11_work/page_vector_once')
    original = json.loads(Path('/root/r11_work/protocol/page_v2/cases.json').read_text())
    cases = []
    for source in original[1:]:
        model = 'llama' if 'llama' in source['model'] else 'qwen3'
        batch = source['batch']
        plan = work / f'{model}_B{batch}_pages16/plan.so'
        if not plan.is_file():
            raise RuntimeError(f'optimized plan missing: {plan}')
        case = dict(source)
        case['decode'] = str(plan)
        case['binary_sha256'] = dict(case['binary_sha256'])
        case['binary_sha256']['decode'] = hashlib.sha256(plan.read_bytes()).hexdigest()
        cases.append(case)
    case_file = work / 'endpoint_cases.json'
    case_file.write_text(json.dumps(cases, indent=2) + '\n')
    command = [PYTHON, str(ROOT / 'docs/experiments/SERVING_R11/check_protocol.py'),
               '--cases', str(case_file), '--out', str(work / 'endpoint_fresh50'),
               '--processes', '50', '--resume']
    return subprocess.run(command, cwd=ROOT, env={**os.environ,
        'PYTHONPATH': str(ROOT / 'python')}).returncode


if __name__ == '__main__':
    raise SystemExit(main())
