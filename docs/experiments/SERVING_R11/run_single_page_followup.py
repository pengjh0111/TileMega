#!/usr/bin/env python3
"""Finish the current-source PG-1 control after its fresh-process gate."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORK = Path('/root/r11_work/pagecheck_once')
CELLS = ('llama_B1', 'llama_B16', 'qwen3_B1', 'qwen3_B16')
CHECKS = (WORK / 'fresh50_llama_B1/summary.json',
          WORK / 'fresh50_remaining/summary.json')


def run(argv: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        result = subprocess.run(argv, cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}): {argv}; see {log}')


def main() -> None:
    os.environ['PYTHONPATH'] = str(ROOT / 'python') + os.pathsep + os.getenv('PYTHONPATH', '')
    deadline = time.monotonic() + 4 * 60 * 60
    while not all(path.exists() for path in CHECKS):
        if time.monotonic() > deadline:
            raise TimeoutError('fresh-process protocol did not finish within four hours')
        time.sleep(30)
    checks = [json.loads(path.read_text()) for path in CHECKS]
    if not all(row['complete'] and row['failed'] == 0 for row in checks) or \
            sum(row['passed'] for row in checks) != 200:
        raise RuntimeError('current-source page protocol did not pass 200/200')
    for cell in CELLS:
        if not (WORK / 'diagnostics' / cell / 'trace.so').is_file():
            raise RuntimeError(f'trace library missing: {cell}')
    print('Protocol passed 200/200; measuring matched full requests', flush=True)
    run([sys.executable, str(HERE / 'measure_single_page_endpoints.py')],
        WORK / 'followup/e2e.log')
    run([sys.executable, str(HERE / 'archive_page_vector_endpoints.py'),
         '--work', str(WORK), '--results', str(WORK / 'matched_e2e'),
         '--policy', str(WORK / 'matched_e2e/measurement_policy.json'),
         '--cases', str(WORK / 'cases.json'),
         '--out', str(HERE / 'single_page_loader/e2e')],
        WORK / 'followup/e2e_archive.log')
    print('Full requests archived; collecting trace diagnostics', flush=True)
    lock = Path('/root/r10_work/serving_gpu.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        run([sys.executable, str(HERE / 'collect_page_diagnostics.py'),
             '--cases', str(WORK / 'cases.json'),
             '--diagnostics', str(WORK / 'diagnostics'),
             '--out', str(WORK / 'diagnostic_results')],
            WORK / 'followup/diagnostics.log')
    run([sys.executable, str(HERE / 'archive_page_vector_diagnostics.py'),
         '--work', str(WORK), '--cases', str(WORK / 'cases.json'),
         '--protocol', str(CHECKS[0]), '--protocol', str(CHECKS[1]),
         '--results', str(WORK / 'diagnostic_results'),
         '--out', str(HERE / 'single_page_loader/diagnostics')],
        WORK / 'followup/diagnostics_archive.log')
    print('Current-source diagnostics archived', flush=True)


if __name__ == '__main__':
    main()
