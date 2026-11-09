#!/usr/bin/env python3
"""Collect an explicitly labeled AB4c host-launch performance variant."""
import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time
import traceback

from sm90d1_common import ROOT, PYTHON, read, sha, write
from sm90r3_llm import span
from sm90r6_merge_analysis import write as csv_write

LABEL = 'AB4c_hostlaunch'


def call(command, log):
    with log.open('w') as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT / 'python')),
            stdout=stream, stderr=subprocess.STDOUT, check=True)


def build(cell, out):
    directory = out / f'{cell["model"]}-B{cell["batch"]}'
    directory.mkdir(parents=True, exist_ok=True)
    old = cell['arms']['AB4c']['binary']
    binary = directory / 'decode.so'
    if (directory / 'built.json').exists():
        return str(binary)
    original = read(old + '.identity.json')
    generated = '#define TILEMEGA_NATIVE_SEQUENCE_HOST_LAUNCH 1\n' + Path(old + '.cu').read_text()
    Path(str(binary) + '.cu').write_text(generated)
    manifest = read(old + '.plan.json')
    manifest.update(sequence_launch='host_per_stage_no_graph', experimental_variant=LABEL)
    write(str(binary) + '.plan.json', manifest)
    snapshot = Path(str(binary) + '.source.json')
    call([PYTHON, '-m', 'tilemega.build.identity', '--root', ROOT, '--snapshot', snapshot,
          '--capture', '--compiler', original['source']['compiler']], directory / 'snapshot.log')
    oldroot = original['source']['root']
    command = []
    for argument in original['options']:
        if argument == old:
            argument = str(binary)
        elif argument == old + '.cu':
            argument = str(binary) + '.cu'
        elif argument == '-I' + oldroot + '/include':
            argument = '-I' + str(ROOT) + argument[len('-I' + oldroot):]
        command.append(argument)
    Path(str(binary) + '.build_command.txt').write_text(shlex.join(command) + '\n')
    limits = Path(str(binary) + '.build_limits.json')
    call(['/usr/bin/python3', ROOT / 'scripts/run_sm90r1_build.py', '--receipt', limits,
          '--max-gib', 48, '--timeout', 5400, '--lock', '/tmp/tilemega_sm90r2_native_build.lock',
          '--', *command], Path(str(binary) + '.ptxas.log'))
    call([PYTHON, '-m', 'tilemega.build.identity', '--root', ROOT, '--snapshot', snapshot,
          '--so', binary], directory / 'identity.log')
    identity = read(str(binary) + '.identity.json')
    write(directory / 'built.json', dict(binary=str(binary), binary_sha256=sha(binary),
        parent=old, parent_sha256=sha(old), parent_generated_source_sha256=sha(old + '.cu'),
        variant=LABEL, graph=False, body='unchanged stage kernels; host runtime dispatch changed',
        coordinates_sha256=identity['plan']['coordinates_sha256'],
        limits=str(limits), limits_sha256=sha(limits), correctness_runs=0))
    return str(binary)


def emit(cell, root, host=None):
    reference = cell['arms']['AB4b']['binary']
    base = Path(read(reference + '.identity.json')['source']['root'])
    definitions = {'AB4b': dict(decode=reference, prefill='prompt', root=str(base))}
    if host:
        definitions[LABEL] = dict(decode=host, prefill='prompt', root=str(ROOT))
    order = ['AB4b', LABEL, LABEL, 'AB4b', 'AB4b', LABEL]
    for slot, label in enumerate(order):
        if label not in definitions:
            continue
        name = f'{cell["model"]}-B{cell["batch"]}-{slot}-{label}'
        spec = dict(name=name, kind='sel', model=cell['model'], batch=cell['batch'],
            protocol='performance-only AB4c host launches without CUDA Graph',
            arms={label: definitions[label]})
        # Each process is independent; the measurement entry point has no arm-count restriction.
        job = dict(id=name, kind='llm', model=cell['model'], batch=cell['batch'],
                   slot=slot, priority=10, arm=label, spec=spec)
        path = root / 'requests' / (name + '.json')
        if path.exists() and read(path)['job'] != job:
            raise ValueError('queued definition changed')
        write(path, dict(job=job))


def summary(root, cells):
    items = read(root / 'queue/items.json')
    if any(v['status'] != 'done' for v in items.values()) or len(items) != len(cells) * 6:
        return None
    comparisons, processes = [], []
    for cell in cells:
        values = {}
        for label in ['AB4b', LABEL]:
            group = [v for v in items.values() if v['job']['model'] == cell['model'] and
                v['job']['batch'] == cell['batch'] and v['job']['arm'] == label]
            if len(group) != 3:
                raise ValueError('expected three fresh processes per arm')
            tpots = [v['result']['tpot_ms'] for v in group]
            values[label] = dict(tpot_ms=statistics.median(tpots), range_pct=span(tpots),
                                 gpus=[v['result']['gpu'] for v in group])
            for value in group:
                row = value['result']
                attempt = value['attempts'][-1]
                binary = value['job']['spec']['arms'][label]['decode']
                processes.append(dict(model=cell['model'], batch=cell['batch'], arm=label,
                    **{k: row[k] for k in ('tpot_ms', 'ttft_ms', 'e2e_ms', 'tokens_per_s', 'gpu', 'raw')},
                    item=value['job']['id'], started=attempt['started'], ended=attempt['ended'],
                    binary=binary, binary_sha256=sha(binary), correctness_runs=0))
        b, c = values['AB4b'], values[LABEL]
        comparisons.append(dict(model=cell['model'], batch=cell['batch'],
            ab4b_tpot_ms=b['tpot_ms'], hostlaunch_tpot_ms=c['tpot_ms'],
            hostlaunch_slowdown_pct=(c['tpot_ms'] / b['tpot_ms'] - 1) * 100,
            ab4b_latency_reduction_vs_hostlaunch_pct=(1 - b['tpot_ms'] / c['tpot_ms']) * 100,
            ab4b_range_pct=b['range_pct'], hostlaunch_range_pct=c['range_pct'],
            ab4b_gpus=json.dumps(b['gpus']), hostlaunch_gpus=json.dumps(c['gpus']),
            processes_per_arm=3, variant=LABEL, cuda_graph=False, correctness_runs=0,
            evidence='verified performance only'))
    out = ROOT / 'docs/results'
    csv_write(out / 'SM90R6_ab4_hostlaunch.csv', comparisons)
    csv_write(out / 'SM90R6_ab4_hostlaunch_processes.csv', processes)
    llama = {v['batch']: v for v in comparisons if v['model'] == 'llama3_1_8b'}
    trigger = llama[16]['hostlaunch_slowdown_pct'] > llama[1]['hostlaunch_slowdown_pct']
    write(root / 'summary.json', dict(rows=comparisons, accepted_processes=len(processes),
        correctness_runs=0, llama3_2_3b_required=trigger,
        trigger_rule='B16 hostlaunch/AB4b relative overhead exceeds B1; fixed before measurement'))
    return comparisons


def run(root):
    inventory = read(root / 'existing-inputs.json')
    cells = inventory['cells']
    (root / 'requests').mkdir(exist_ok=True)
    write(root / 'method.json', dict(variant=LABEL, cuda_graph=False, correctness_runs=0,
        worktree=str(ROOT), runtime_sha256=sha(ROOT / 'include/tilemega/Codegen/tasks/Sm90DenseRuntime.cuh'),
        conditional_llama3_2_3b='If B16 hostlaunch/AB4b overhead exceeds B1, append Llama-3.2-3B',
        graph_dataset='retained unchanged', expected_cells=len(cells), expected_processes=len(cells) * 6))
    for cell in cells:
        emit(cell, root)
    command = ['/usr/bin/python3', '-u', str(ROOT / 'scripts/sm90r6_gpu_items.py'),
               '--out', str(root / 'queue'), '--requests', str(root / 'requests')]
    previous = root / 'queue-process.json'
    alive = previous.exists() and Path('/proc', str(read(previous)['pid'])).exists()
    if not alive:
        with (root / 'queue.log').open('a') as log:
            queue = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(previous, dict(pid=queue.pid, command=command))
    failures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(build, cell, root / 'build-v2'): cell for cell in cells}
        for future in concurrent.futures.as_completed(futures):
            cell = futures[future]
            try:
                emit(cell, root, future.result())
                print(cell['model'] + '-B' + str(cell['batch']) + ' host variant compiled', flush=True)
            except Exception:
                failures[cell['model'] + '-B' + str(cell['batch'])] = traceback.format_exc()
                write(root / 'build-failures.json', failures)
    if failures:
        raise RuntimeError('host variants failed to compile; other queued performance jobs continue')
    (root / 'requests/CLOSED').touch()
    while True:
        items = read(root / 'queue/items.json')
        write(root / 'status.json', dict(at=time.time(), total=72,
            done=sum(v['status'] == 'done' for v in items.values()),
            running=[dict(item=k, gpu=v['gpu']) for k, v in items.items() if v['status'] == 'running'],
            failed=[k for k, v in items.items() if v['status'] == 'failed']))
        if any(v['status'] == 'failed' for v in items.values()):
            raise RuntimeError('performance job failed; inspect raw queue')
        result = summary(root, cells)
        if result is not None:
            print(json.dumps(result), flush=True)
            return
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())
