#!/usr/bin/env python3
"""Build trace libraries for the fixed-geometry single-page loader control."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
WORK = Path('/root/r11_work')
CELLS = ('llama_B1', 'llama_B16', 'qwen3_B1', 'qwen3_B16')


def build(cell: str) -> tuple[str, int]:
    source = WORK / 'protocol/page_v2' / f'{cell}_pages/plan.so.cu'
    prior = WORK / 'page_diagnostics' / cell / 'build_command.json'
    destination = WORK / 'pagecheck_once/diagnostics' / cell
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / 'trace.so'
    command = json.loads(prior.read_text())
    old_output = str(WORK / 'page_diagnostics' / cell / 'trace.so')
    command = [str(output) if arg == old_output else arg for arg in command]
    if command.count(str(source)) != 1 or command.count(str(output)) != 1:
        raise ValueError(f'trace command does not match the measured plan: {cell}')
    (destination / 'build_command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (destination / 'build.log').open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT)
    return cell, result.returncode


def main() -> None:
    # nvcc compiles the large serving translation units; two jobs avoid
    # competing with the concurrent architecture audit for host memory.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for cell, status in pool.map(build, CELLS):
            print(cell, status, flush=True)
            if status:
                raise RuntimeError(f'{cell} trace build failed')


if __name__ == '__main__':
    main()
