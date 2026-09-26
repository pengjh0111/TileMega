#!/usr/bin/env python3
"""Measure the three remaining fixed-geometry page controls after protocol checks."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from measure_pg_ablation import LOCK, PYTHON, ROOT, cell_sources, manifest, sha


WORK = Path('/root/r11_work/page_vector_once')
CELLS = (('llama', 16), ('qwen3', 1), ('qwen3', 16))
FIELDS = ('gemms', 'grid', 'residency', 'kappa', 'attention_kv_block',
          'attention_query_rows', 'sync', 'event_solo', 'event_red_publish',
          'barrier_v2')


def matching_off_plan(model: str, batch: int, pages: Path, original: Path) -> Path:
    """Compile the same selected CG if the historical off arm used another rank."""
    if all(manifest(original)[field] == manifest(pages)[field] for field in FIELDS):
        return original
    source_command = json.loads((pages.parent / 'compile_command.json').read_text())
    if source_command[1] != 'compile' or '--pg' not in source_command:
        raise RuntimeError('cannot reconstruct the optimized plan source')
    selected = Path(source_command[2])
    if not selected.is_file():
        raise RuntimeError(f'selected CG missing: {selected}')
    destination = WORK / f'{model}_B{batch}_off_matched'
    destination.mkdir(parents=True, exist_ok=True)
    off = destination / 'plan.so'
    command = source_command.copy()
    command[3] = str(off)
    at = command.index('--pg')
    command[at + 1] = 'off'
    if '--page-bytes' in command:
        at = command.index('--page-bytes')
        del command[at:at + 2]
    source_hash = sha(selected)
    record = destination / 'build_record.json'
    if not (off.is_file() and record.is_file() and
            json.loads(record.read_text()).get('source_sha256') == source_hash):
        with (destination / 'build.log').open('w') as log:
            status = subprocess.run(command, cwd=ROOT,
                                    env={**os.environ, 'PYTHONPATH': str(ROOT / 'python')},
                                    stdout=log, stderr=subprocess.STDOUT).returncode
        if status:
            raise RuntimeError(f'matching PG-off build failed: {destination / "build.log"}')
        record.write_text(json.dumps(dict(argv=command, source_sha256=source_hash,
                                          binary_sha256=sha(off)), indent=2) + '\n')
    return off


def run_measure(model: str, batch: int, label: str, decode: Path,
                prefill: Path, policy: Path, out: Path) -> dict:
    model_dir = Path('/root/models') / ('llama3_2_1b' if model == 'llama'
                                       else 'qwen3_1_7b')
    prompts = ROOT / f'docs/experiments/SERVING_R10/prompts/{model}_ids.json'
    command = [PYTHON, '-m', 'tilemega.serving.measure', '--model', str(model_dir),
               '--prefill-so', str(prefill), '--decode-so', str(decode),
               '--prompt-ids', str(prompts), '--batch', str(batch),
               '--mode', 'L2', '--warmup', '1', '--repeats', '3',
               '--policy', str(policy), '--out', str(out)]
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (out / 'command.log').open('w') as log:
        status = subprocess.run(command, cwd=ROOT,
                                env={**os.environ, 'PYTHONPATH': str(ROOT / 'python')},
                                stdout=log, stderr=subprocess.STDOUT).returncode
    row = dict(model=model, batch=batch, label=label, binary=str(decode),
               sha256=sha(decode), argv=command, exit_code=status,
               seconds=time.monotonic() - started)
    if status:
        raise RuntimeError(f'{label} failed for {model} B{batch}: {out / "command.log"}')
    result = json.loads((out / 'measurements.json').read_text())
    row.update(e2e_seconds=result['e2e_seconds'],
               tpot_seconds=result['tpot_seconds'],
               timed_tokens_identical=result['timed_tokens_identical'])
    if not row['timed_tokens_identical']:
        raise RuntimeError(f'timed tokens differed for {model} B{batch} {label}')
    return row


def main() -> None:
    protocol = json.loads((WORK / 'endpoint_fresh50/summary.json').read_text())
    if not protocol.get('complete') or protocol.get('passed') != 150 or protocol.get('failed'):
        raise RuntimeError('all three 50-process protocol checks must pass first')
    cases = json.loads((WORK / 'endpoint_cases.json').read_text())
    if len(cases) != len(CELLS):
        raise RuntimeError('the endpoint case table is incomplete')
    policy = Path('/root/r11_work/pg_ablation/measurement_policy.json')
    if not policy.is_file():
        raise RuntimeError('the predeclared measurement policy is missing')
    out = WORK / 'endpoint_e2e'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    # Hold the device lock across both arms of every cell, preventing a
    # diagnostic or other TileMega measurement from splitting the comparison.
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index, (model, batch) in enumerate(CELLS):
            case = cases[index]
            if case['batch'] != batch or model not in case['model']:
                raise RuntimeError('endpoint protocol case ordering changed')
            source, prefill = cell_sources(model, batch)
            pages = Path(case['decode'])
            if sha(pages) != case['binary_sha256']['decode']:
                raise RuntimeError(f'optimized plan changed after protocol: {model} B{batch}')
            off = matching_off_plan(model, batch, pages, source['off'])
            a, b = manifest(off), manifest(pages)
            if any(a[field] != b[field] for field in FIELDS):
                raise RuntimeError(f'fixed geometry differs: {model} B{batch}')
            if a['pg'] != 'off' or b['pg'] != 'pages':
                raise RuntimeError(f'PG arm label differs: {model} B{batch}')
            order = (('off', off), ('pages', pages)) if index % 2 == 0 else (
                ('pages', pages), ('off', off))
            for label, decode in order:
                row = run_measure(model, batch, label, decode, prefill, policy,
                                  out / f'{model}_B{batch}' / label)
                rows.append(row)
                (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
