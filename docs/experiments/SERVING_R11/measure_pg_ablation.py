#!/usr/bin/env python3
"""Two-cell, fixed-geometry off/L2/page control after protocol acceptance."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PYTHON = '/root/venvs/tilemega-torch213-cu126/bin/python'
LOCK = Path('/root/r10_work/serving_gpu.lock')


def power_policy(out: Path) -> Path:
    """Declare one idle+30 W rule before any arm is timed."""
    policy = out / 'measurement_policy.json'
    if policy.is_file():
        return policy
    # The previous fresh-process check may leave the board warm.  No CUDA
    # context remains when the queue reaches this point; allow it to cool
    # before estimating idle power, then keep that estimate fixed for all arms.
    time.sleep(30)
    device=os.environ.get('TILEMEGA_DEVICE_INDEX','0')
    owners = subprocess.check_output(['nvidia-smi', '-i', device,
        '--query-compute-apps=pid',
        '--format=csv,noheader'], text=True).strip()
    if owners:
        raise RuntimeError(f'cannot predeclare idle power while GPU has owners: {owners}')
    samples = []
    for _ in range(5):
        raw = subprocess.check_output(['nvidia-smi', '-i', device,
            '--query-gpu=power.draw', '--format=csv,noheader,nounits'], text=True)
        samples.append(float(raw.splitlines()[0].strip()))
        time.sleep(2)
    if max(samples) - min(samples) > 10:
        raise RuntimeError(f'idle power did not settle: {samples}')
    policy.write_text(json.dumps(dict(guard=True,
        idle_power_w=statistics.median(samples), power_margin_w=30,
        cooldown_seconds=30, retries=3, idle_samples_w=samples), indent=2)+'\n')
    return policy


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(path: Path) -> dict:
    return json.loads(Path(str(path) + '.plan.json').read_text())


def cell_sources(model: str, batch: int):
    root = Path('/root/r11_work')
    base = root / f'protocol/page_v2/{model}_B{batch}'
    off = Path(str(base) + '_off/plan.so')
    pages = Path(str(base) + '_pages/plan.so')
    l2 = (root / 'pg0_smoke/plan.so' if (model, batch) == ('llama', 1)
          else root / f'protocol/pg0_cells/{model}_B{batch}_l2/plan.so')
    prefill = root / f'r10_control/plans/{model}_prefill_B{batch}/plan.so'
    return {'off': off, 'l2': l2, 'pages': pages}, prefill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocols = json.loads((ROOT / 'docs/experiments/SERVING_R11/protocol_results.json').read_text())
    if not protocols.get('complete'):
        raise RuntimeError('page protocol results are incomplete')
    policy=power_policy(args.out)
    selected = [('llama', 1), ('qwen3', 16)]
    records = []
    for index, (model, batch) in enumerate(selected):
        for protocol in ('pg0', 'pg1'):
            if not any(row['mode'] == protocol and row['model'] == model and
                       row['batch'] == batch and row['fresh_processes'] >= 50 and
                       row['mismatches'] == 0 for row in protocols['rows']):
                raise RuntimeError(f'{model} B{batch}: {protocol} protocol evidence missing')
        binaries, prefill = cell_sources(model, batch)
        control = manifest(binaries['off'])
        geometry = [(g['tile_m'], g['tile_n'], g['tile_k'], g['stages'], g['split_k'])
                    for g in control['gemms']]
        for mode, binary in binaries.items():
            plan = manifest(binary)
            if geometry != [(g['tile_m'], g['tile_n'], g['tile_k'], g['stages'], g['split_k'])
                            for g in plan['gemms']]:
                raise RuntimeError(f'{model} B{batch}: PG arms differ in GEMM geometry')
            matched = ('grid', 'residency', 'kappa', 'attention_kv_block',
                       'attention_query_rows', 'sync')
            if plan['pg'] != mode or any(plan[field] != control[field] for field in matched):
                raise RuntimeError(f'{model} B{batch}: PG arm does not match its label or control')
        model_dir = Path('/root/models') / ('llama3_2_1b' if model == 'llama' else 'qwen3_1_7b')
        prompts = ROOT / f'docs/experiments/SERVING_R10/prompts/{model}_ids.json'
        for mode in (('off', 'l2', 'pages') if index == 0 else ('pages', 'l2', 'off')):
            binary = binaries[mode]
            dest = args.out / f'{model}_B{batch}/{mode}'
            dest.mkdir(parents=True, exist_ok=True)
            argv = [PYTHON, '-m', 'tilemega.serving.measure', '--model', str(model_dir),
                    '--prefill-so', str(prefill), '--decode-so', str(binary),
                    '--prompt-ids', str(prompts), '--batch', str(batch), '--mode', 'L2',
                    '--warmup', '1', '--repeats', '3', '--policy', str(policy),
                    '--out', str(dest)]
            started = time.monotonic()
            with LOCK.open('a') as lock, (dest / 'command.log').open('w') as output:
                fcntl.flock(lock, fcntl.LOCK_EX)
                status = subprocess.run(argv, cwd=ROOT,
                    env={**os.environ, 'PYTHONPATH': str(ROOT / 'python')},
                    stdout=output, stderr=subprocess.STDOUT).returncode
            row = {'model': model, 'batch': batch, 'pg': mode, 'mode': 'L2',
                   'binary': str(binary), 'sha256': sha(binary), 'argv': argv,
                   'exit_code': status, 'seconds': time.monotonic()-started}
            if not status:
                result = json.loads((dest / 'measurements.json').read_text())
                row['e2e_seconds'] = result['e2e_seconds']
                row['tpot_seconds'] = result['tpot_seconds']
                row['timed_tokens_identical'] = result['timed_tokens_identical']
            records.append(row)
            (args.out / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
            if status:
                raise RuntimeError(f'{model} B{batch} PG={mode} failed; see {dest / "command.log"}')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
