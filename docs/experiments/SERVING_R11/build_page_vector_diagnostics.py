#!/usr/bin/env python3
"""Rebuild PG trace binaries after the physical-page routing change."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import subprocess

from measure_pg_ablation import ROOT


WORK = Path('/root/r11_work')


def build(name: str) -> tuple[str, int]:
    old = WORK / 'page_diagnostics' / name / 'build_command.json'
    command = json.loads(old.read_text())
    source = str(WORK / 'protocol/page_v2' / f'{name}_pages/plan.so.cu')
    replacement = str(WORK / 'page_vector_once' / f'{name}_pages16/plan.so.cu')
    output = WORK / 'page_vector_once/diagnostics' / name / 'trace.so'
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [replacement if arg == source else str(output) if arg.endswith('/trace.so') else arg
               for arg in command]
    if replacement not in command or str(output) not in command:
        raise RuntimeError(f'trace build command did not select {name}')
    (output.parent / 'build_command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (output.parent / 'build.log').open('w') as log:
        status = subprocess.run(command, cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT).returncode
    return name, status


def main() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name, status in pool.map(build, ('llama_B1', 'llama_B16',
                                            'qwen3_B1', 'qwen3_B16')):
            print(name, status, flush=True)
            if status:
                raise RuntimeError(f'{name} diagnostic build failed')


if __name__ == '__main__':
    main()
