#!/usr/bin/env python3
"""Publish the complete host-launch matrix and its conditional extension."""
import argparse
import csv
from pathlib import Path
import subprocess
import time

from sm90d1_common import ROOT, read, sha, write


def publish(root, push=False):
    while not (root / 'summary.json').exists() or not (root / 'llama3b-condition.json').exists():
        time.sleep(10)
    condition = read(root / 'llama3b-condition.json')
    extension = root / 'llama3b/summary.json'
    if condition['triggered']:
        while not extension.exists():
            time.sleep(10)
    main = read(root / 'summary.json')
    rows = main['rows']
    if condition['triggered']:
        rows += read(extension)['rows']
    out = ROOT / 'docs/results'
    old = list(csv.DictReader((out / 'SM90R6_ab4_comparison.csv').open()))
    extension_method = (
        'The user subsequently requested Llama-3.2-3B unconditionally, overriding the '
        'original batch-trend trigger. This extension retains the Llama-3.1-8B results. '
        if condition.get('unconditional') else
        'The extension rule was fixed before measurement: append Llama-3.2-3B when '
        'Llama-3.1-8B has greater hostlaunch/AB4b relative overhead at B16 than B1. '
        'The original Llama-3.1-8B rows remain in the dataset either way. ')
    report = ('# Stage-granular host-launch performance variant\n\n'
        'Verified implementation: opt-in `TILEMEGA_NATIVE_SEQUENCE_HOST_LAUNCH=1` '
        'bypasses CUDA Graph construction and replay. Each decode step enqueues the '
        'same per-stage kernels through ordinary CUDA host launches. The selected '
        'coordinates, stage bodies, grid and shared memory come from the existing '
        'AB4b/AB4c program at each model/batch. The new arm is labeled `AB4c_hostlaunch`; '
        'the original graph-based AB4c data remains separately labeled and retained. '
        'The runtime option defaults to zero.\n\n'
        'The user requested performance only. No new C1/C2 or standalone occupancy '
        'experiments were run. Each arm/cell uses three fresh processes, one warmup '
        'and three timed samples at N=1 and N=1024, and the existing timing driver '
        'and interference guard. Independent processes run on any idle GPU 0–3. '
        'The initial matrix contains 12 cells and 72 final timing processes. '
        'GPU assignments, all measured values and binary provenance are retained in CSV.\n\n'
        + extension_method + 'Extension enabled: `' + str(condition['triggered']).lower() + '`. '
        'When triggered, Llama-3.2-3B uses independent 600-second aggregate searches '
        'at B1/B4/B16, four compiled finalists and eight P-SEL processes per batch, '
        'then the same three-process performance pair; correctness is not rerun.\n\n'
        '| Model | Batch | AB4b TPOT ms | Host-launch TPOT ms | AB4b speedup vs host-launch | Original graph AB4c ms (historical) |\n'
        '|---|---:|---:|---:|---:|---:|\n')
    for value in sorted(rows, key=lambda r: (r['model'], r['batch'])):
        ref = next((r for r in old if r['model'] == value['model'] and int(r['batch']) == value['batch']), None)
        historical = f'{float(ref["ab4c_tpot_ms"]):.6f}' if ref else 'N/A'
        report += (f'| {value["model"]} | {value["batch"]} | {value["ab4b_tpot_ms"]:.6f} | '
            f'{value["hostlaunch_tpot_ms"]:.6f} | {value["hostlaunch_tpot_ms"] / value["ab4b_tpot_ms"]:.6f}× | '
            f'{historical} |\n')
    report += ('\nThe original graph AB4c column is historical data; its timing sessions differ '
        'from this collection. This dataset measures the explicit host-launch implementation '
        'and does not replace the graph-based baseline. Raw source, build receipts and all '
        'attempts: `' + str(root) + '`. CSV values retain full precision.\n')
    path = out / 'SM90R6_ab4_hostlaunch.md'
    path.write_text(report)
    names = ['SM90R6_ab4_hostlaunch.csv', 'SM90R6_ab4_hostlaunch_processes.csv', path.name]
    if condition['triggered']:
        names += ['SM90R6_ab4_hostlaunch_llama3b.csv', 'SM90R6_ab4_hostlaunch_llama3b_processes.csv']
    paths = [out / name for name in names]
    write(root / 'publication.json', dict(performance_only=True, cells=len(rows),
        correctness_runs=0, llama3b_added=condition['triggered'], files={str(p): sha(p) for p in paths}))
    if push:
        if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'sm90r6-ab4-hostlaunch':
            raise ValueError('unexpected publication branch')
        relative = [str(p.relative_to(ROOT)) for p in paths]
        subprocess.run(['git', 'add', '--', *relative], cwd=ROOT, check=True)
        subprocess.run(['git', 'commit', '--only', '-m', 'docs: publish host-launch AB4 performance matrix',
                        '--', *relative], cwd=ROOT, check=True)
        subprocess.run(['git', 'push', '-u', 'origin', 'sm90r6-ab4-hostlaunch'], cwd=ROOT, check=True, timeout=45)
        write(root / 'publication-git.json', dict(pushed=True, commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    publish(args.root.resolve(), args.push)
