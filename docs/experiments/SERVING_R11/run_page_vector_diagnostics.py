#!/usr/bin/env python3
"""Queue current-source PG trace diagnostics after fresh-process checks."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from measure_pg_ablation import LOCK, PYTHON, ROOT


def main() -> None:
    work = Path('/root/r11_work/page_vector_once')
    first = json.loads((work / 'cases.json').read_text())
    rest = json.loads((work / 'endpoint_cases.json').read_text())
    cases = first + rest
    case_file = work / 'all_cases.json'
    case_file.write_text(json.dumps(cases, indent=2) + '\n')
    required = [work / 'fresh50/summary.json',
                work / 'endpoint_fresh50/summary.json']
    required += [work / 'diagnostics' / f'{model}_B{batch}' / 'trace.so'
                 for model in ('llama', 'qwen3') for batch in (1, 16)]
    deadline = time.monotonic() + 7200
    while not all(path.is_file() for path in required):
        progress = work / 'endpoint_fresh50/processes.json'
        if progress.is_file() and any(row['exit_code'] for row in json.loads(progress.read_text())):
            raise RuntimeError('endpoint fresh-process check failed; diagnostics not queued')
        if time.monotonic() > deadline:
            raise TimeoutError('fresh-process or diagnostic-build evidence was not produced')
        time.sleep(10)
    a, b = [json.loads(path.read_text()) for path in required[:2]]
    if not a.get('complete') or a['passed'] != 50 or not b.get('complete') or b['passed'] != 150:
        raise RuntimeError('the four 50-process protocol checks did not complete')
    command = [PYTHON, str(ROOT / 'docs/experiments/SERVING_R11/collect_page_diagnostics.py'),
               '--cases', str(case_file), '--diagnostics', str(work / 'diagnostics'),
               '--out', str(work / 'diagnostic_results')]
    with LOCK.open('a') as lock, (work / 'diagnostic_command.log').open('w') as log:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return subprocess.run(command, cwd=ROOT,
            env={**os.environ, 'PYTHONPATH': str(ROOT / 'python')},
            stdout=log, stderr=subprocess.STDOUT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
