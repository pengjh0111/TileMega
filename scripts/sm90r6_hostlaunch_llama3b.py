#!/usr/bin/env python3
"""Append Llama-3.2-3B conditionally or at the user's explicit request."""
import argparse
import concurrent.futures
import json
import lzma
import os
from pathlib import Path
import statistics as st
import subprocess
import time

from sm90d1_common import ROOT, PYTHON, read, sha, write
from sm90r3_llm import span
from sm90r6_hostlaunch_ab4 import LABEL, build, call, emit
from sm90r6_merge_analysis import write as csv_write

BASE = Path('/root/TileMega-r6')
COMPILER = BASE / 'runs/sm90r6/ablation/finalists-v1/compiler'
SEARCH = BASE / 'runs/sm90r6/ablation/offline-v1/search'


def wait_items(root, names):
    while True:
        path = root / 'queue/items.json'
        values = read(path) if path.exists() else {}
        if any(values.get(name, {}).get('status') == 'failed' for name in names):
            raise RuntimeError('Llama-3B performance item failed')
        if all(values.get(name, {}).get('status') == 'done' for name in names):
            return [values[name]['result'] for name in names]
        time.sleep(5)


def cell_run(batch, root):
    model = 'llama3_2_3b'
    directory = root / f'{model}-B{batch}'
    directory.mkdir()
    export = directory / 'decode.export.json'
    packed = ROOT / f'configs/serving/sm90d1/inputs/{model}/decode.export.json.xz'
    export.write_bytes(lzma.decompress(packed.read_bytes()))
    source = ROOT / f'configs/serving/sm90r3/{model}_b{batch}.json'
    seed = directory / 'seed.json'
    write(seed, read(source)['coordinates'])
    request = dict(id=directory.name, model=model, batch=batch, tp=1, arm='AB4b', budget_ms=600000,
        seed=str(seed), seed_sha256=sha(seed), seed_source=str(source), seed_source_sha256=sha(source),
        export=str(export), export_sha256=sha(export))
    write(directory / 'cells.json', [request])
    call([SEARCH, ROOT / 'configs/targets/sm_90_h800.json', ROOT,
          directory / 'cells.json', directory / 'search.jsonl'], directory / 'search.log')
    receipt = json.loads((directory / 'search.jsonl').read_text())
    write(directory / 'receipt.json', receipt)
    call(['/usr/bin/python3', ROOT / 'scripts/sm90r6_compile_finalists.py', '--receipt',
          directory / 'receipt.json', '--compiler', COMPILER, '--out', directory / 'finalists',
          '--compile-only'], directory / 'finalists.log')
    candidates = [v for v in read(directory / 'finalists/candidates.json') if v['status'] == 'compiled']
    if len(candidates) != 4:
        raise RuntimeError('Llama-3B needs four compiled finalists; inspect resource records')
    labels = ['c' + str(v['rank']) for v in candidates]
    definitions = {label: dict(decode=value['binary'], prefill='prompt', root=str(BASE))
                   for label, value in zip(labels, candidates)}
    names = []
    for slot, label in enumerate(labels + labels[::-1]):
        name = f'{model}-B{batch}-psel-{slot}-{label}'
        names.append(name)
        spec = dict(name=name, kind='sel', model=model, batch=batch, arms=definitions,
                    protocol='performance-only P-SEL for appended Llama-3B')
        write(root / 'requests' / (name + '.json'), dict(job=dict(id=name, kind='llm',
            model=model, batch=batch, slot=slot, priority=0, arm=label, spec=spec)))
    rows = wait_items(root, names)
    values = {label: [v['tpot_ms'] for v in rows if v['arm'] == label] for label in labels}
    ranked = sorted(labels, key=lambda label: st.median(values[label]))
    a, b = ranked[:2]
    gap = (st.median(values[b]) / st.median(values[a]) - 1) * 100
    threshold = max(.5, 2 * max(span(values[a]), span(values[b])))
    predictions = {label: value['predicted_ns'] for label, value in zip(labels, candidates)}
    winner = min((a, b), key=predictions.get) if gap <= threshold else a
    selected = candidates[int(winner[1:])]
    write(directory / 'selection.json', dict(winner=winner, gap_pct=gap, threshold_pct=threshold,
        predictions=predictions, rows=rows, correctness_runs=0))
    sequence = directory / 'graph-parent.so'
    call([COMPILER, 'compile', export, sequence, '--tmexec-config', selected['coordinates'],
          '--runtime-target', ROOT / 'configs/targets/sm_90_h800.json', '--batch', batch,
          '--serving', 'decode', '--emit', 'kernel-sequence', '--nvcc-split-compile', 4],
          directory / 'sequence-compile.log')
    cell = dict(model=model, batch=batch, arms={
        'AB4b': dict(binary=selected['binary']), 'AB4c': dict(binary=str(sequence))})
    host = build(cell, root / 'host-build')
    emit(cell, root, host)
    write(directory / 'prepared.json', dict(cell=cell, host=host))
    print(f'{model}-B{batch} performance pair queued', flush=True)
    return cell


def summarize(root, cells):
    q = read(root / 'queue/items.json')
    comparisons, processes = [], []
    for cell in cells:
        groups = {label: [v for v in q.values() if v['job']['batch'] == cell['batch'] and
                  v['job']['arm'] == label and v['status'] == 'done'] for label in ['AB4b', LABEL]}
        if any(len(v) != 3 for v in groups.values()):
            raise RuntimeError('incomplete appended performance pair')
        b, c = (st.median(v['result']['tpot_ms'] for v in groups[label]) for label in ['AB4b', LABEL])
        comparisons.append(dict(model=cell['model'], batch=cell['batch'], ab4b_tpot_ms=b,
            hostlaunch_tpot_ms=c, hostlaunch_slowdown_pct=(c / b - 1) * 100,
            ab4b_latency_reduction_vs_hostlaunch_pct=(1 - b / c) * 100,
            processes_per_arm=3, correctness_runs=0, evidence='verified performance only'))
        for label, group in groups.items():
            for value in group:
                result = value['result']
                processes.append(dict(result, batch=cell['batch'], arm=label))
    csv_write(ROOT / 'docs/results/SM90R6_ab4_hostlaunch_llama3b.csv', comparisons)
    csv_write(ROOT / 'docs/results/SM90R6_ab4_hostlaunch_llama3b_processes.csv', processes)
    write(root / 'summary.json', dict(rows=comparisons, psel_processes=24,
        final_processes=18, correctness_runs=0, compiler=str(COMPILER), compiler_sha256=sha(COMPILER)))


def run(parent, force=False):
    if force:
        request = parent / 'llama3b-request.json'
        if read(request)['mode'] != 'unconditional':
            raise ValueError('explicit unconditional request is required')
        condition = dict(triggered=True, unconditional=True,
            rule='User explicitly requested Llama-3.2-3B regardless of batch trend',
            request=str(request), request_sha256=sha(request))
    else:
        while not (parent / 'summary.json').exists():
            time.sleep(10)
        result = read(parent / 'summary.json')
        condition = dict(triggered=result['llama3_2_3b_required'], unconditional=False,
            rule=result['trigger_rule'], parent_summary_sha256=sha(parent / 'summary.json'))
    write(parent / 'llama3b-condition.json', condition)
    if not condition['triggered']:
        print('Llama-3B condition not triggered', flush=True)
        return
    root = parent / 'llama3b'
    root.mkdir()
    (root / 'requests').mkdir()
    command = ['/usr/bin/python3', '-u', str(ROOT / 'scripts/sm90r6_gpu_items.py'),
               '--out', str(root / 'queue'), '--requests', str(root / 'requests')]
    with (root / 'queue.log').open('w') as log:
        queue = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(root / 'queue-process.json', dict(pid=queue.pid, command=command))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        cells = list(pool.map(lambda batch: cell_run(batch, root), [1, 4, 16]))
    (root / 'requests/CLOSED').touch()
    while True:
        values = read(root / 'queue/items.json')
        if any(v['status'] == 'failed' for v in values.values()):
            raise RuntimeError('Llama-3B measurement failed')
        if len(values) == 42 and all(v['status'] == 'done' for v in values.values()):
            break
        time.sleep(10)
    summarize(root, cells)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--force', action='store_true', help='apply the recorded explicit user request')
    args = parser.parse_args()
    run(args.root.resolve(), args.force)
