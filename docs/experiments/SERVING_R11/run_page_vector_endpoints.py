#!/usr/bin/env python3
"""Queue endpoint timing after all current-source trace diagnostics finish."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

from measure_pg_ablation import PYTHON, ROOT


def main() -> int:
    work = Path('/root/r11_work/page_vector_once')
    protocol = work / 'endpoint_fresh50/summary.json'
    reports = [work / 'diagnostic_results' / f'{model}_B{batch}' / 'report/page_chain_summary.json'
               for model in ('llama', 'qwen3') for batch in (1, 16)]
    deadline = time.monotonic() + 10800
    while not protocol.is_file() or not all(report.is_file() for report in reports):
        progress = work / 'endpoint_fresh50/processes.json'
        if progress.is_file() and any(row['exit_code'] for row in json.loads(progress.read_text())):
            raise RuntimeError('fresh-process check failed; no performance measurement queued')
        if (work / 'diagnostic_command.log').is_file():
            log = (work / 'diagnostic_command.log').read_text()
            if 'Traceback (most recent call last)' in log:
                raise RuntimeError('diagnostic queue failed; no performance measurement queued')
        if time.monotonic() > deadline:
            raise TimeoutError('protocol or current-source diagnostic evidence missing')
        time.sleep(10)
    return subprocess.run([PYTHON, str(ROOT / 'docs/experiments/SERVING_R11/measure_page_vector_endpoints.py')],
                          cwd=ROOT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
