#!/usr/bin/env python3
"""Publish accepted AIT cells from immutable P5 receipts, preserving partial status."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import dnn1_aitemplate as runner
import dnn1_baselines as base
import dnn1_aitemplate_failure as failure


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def checked(path, digest):
    path = Path(path)
    if base.sha(path) != digest:
        raise ValueError('published evidence changed: ' + str(path))
    return base.read(path)


def accepted_row(value, fields):
    if (value['status'] != 'pass' or value['issues'] or value['processes'] != 5 or
            value['correctness_passes'] != 5 or value['precision'] != 'bf16' or
            value['graph_mode'] is not True):
        raise ValueError('publication requires a complete accepted BF16 graph P5 cell')
    receipts = [checked(p['path'], p['sha256']) for p in value['source']]
    first = receipts[0]
    session_path = Path(value['source'][0]['path']).parents[2] / 'cell-result.json'
    session = checked(session_path, value['session_sha256'])
    guard = session_path.parents[2] / 'guard.jsonl'
    if base.sha(guard) != value['outer_guard_sha256']:
        raise ValueError('outer guard changed')
    identity = first['identity']
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    compiled_path = runner.AIT_ROOT / 'engines' / key / 'compiled.json'
    compiled = checked(compiled_path, first['compiled_receipt_sha256'])
    if base.sha(compiled['library']) != first['library_sha256']:
        raise ValueError('compiled library changed')
    if any(r['identity'] != identity or r['library_sha256'] != first['library_sha256']
           for r in receipts):
        raise ValueError('publication cannot mix identities or libraries')
    clocks = [[], []]
    for receipt, window in zip(receipts, value['windows']):
        if base.sha(window['snapshots']) != window['snapshots_sha256'] or window['issues']:
            raise ValueError('PID-window audit changed or failed')
        if receipt['process_role'] == 'measurement':
            for index, label in enumerate(('before_measurement', 'after_measurement')):
                clocks[index].append(float(window[label][0]['telemetry']['clocks.sm']))
    if len(clocks[0]) != 3 or any(len(r['samples_ms']) != 200 for r in receipts[:3]):
        raise ValueError('publication requires three complete 200-event receipts')
    row = {field: '' for field in fields}
    row.update(model=value['model'], batch=value['batch'], system='aitemplate',
        arm=f"aitemplate-bf16-graph-v{identity['recipe']['schema']}", precision='bf16',
        gpu=value['gpu'], status='pass', sm_clock_start_mhz=compact(clocks[0]),
        sm_clock_end_mhz=compact(clocks[1]), processes=3,
        correctness_processes=5, correctness_passes='5/5',
        correctness_metrics_identical=len({compact(r['correctness']) for r in receipts}) == 1,
        correctness=compact([r['correctness'] for r in receipts]), versions=compact(identity),
        engine_or_library_sha256=compact(dict(library_sha256=first['library_sha256'],
            compiled_receipt_sha256=first['compiled_receipt_sha256'])),
        source=compact([p['path'] for p in value['source']]),
        correctness_policy=value['correctness_policy'], strict_v1_status='pass',
        sm_clock_source='three fresh before/after measurement snapshots',
        exported_program_sha256=identity['exported_program_sha256'],
        source_checkpoint_sha256=identity['source_checkpoint_sha256'],
        inputs_sha256=first['inputs_sha256'], p_cell_complete=True, cell_session=str(session_path))
    for name in ('median_ms', 'p90_ms', 'process_range_ms', 'process_range_pct'):
        row[name] = value[name]
    diag = dict(model=value['model'], batch=value['batch'], status='pass', physical_gpu=value['gpu'],
        gpu_uuid=identity['gpu_uuid'], precision='bf16', graph_requested=True,
        graph_observations=compact(value['graph_observations']),
        recipe=compact(identity['recipe']), source_identity_sha256=hashlib.sha256(compact(identity).encode()).hexdigest(),
        installed_sources_sha256=identity['installed_sources_sha256'],
        library=str(compiled['library']), library_sha256=first['library_sha256'],
        compiled_receipt=str(compiled_path), compiled_receipt_sha256=first['compiled_receipt_sha256'],
        sm_clock_start_mhz=compact(clocks[0]), sm_clock_end_mhz=compact(clocks[1]),
        session=str(session_path), session_sha256=value['session_sha256'],
        outer_guard=str(guard), outer_guard_sha256=value['outer_guard_sha256'],
        profile_cache_before=compact(session['profile_cache_before']),
        profile_cache_after=compact(session['profile_cache_after']), error_chain='')
    return row, diag


def write_csv(path, rows, fields=None):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def publish(args):
    canonical = runner.ROOT / 'docs/results/DNN1_baselines.csv'
    with canonical.open() as stream:
        fields = csv.DictReader(stream).fieldnames
    values = [value for epoch in args.epoch for value in base.read(epoch / 'accepted.json')]
    declarations = [(path, base.read(path)) for path in args.failure_declaration]
    keys = [(v['model'], v['batch']) for v in values] + [
        (value['model'], value['batch']) for _, value in declarations]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('accepted cells must be nonempty and distinct across epochs')
    if args.complete and set(keys) != {(model, batch) for model in base.MODELS for batch in base.BATCHES}:
        raise ValueError('complete publication requires exactly25 distinct terminal cells')
    order = {m: i for i, m in enumerate(base.MODELS)}
    values.sort(key=lambda r: (order[r['model']], r['batch']))
    pairs = [accepted_row(value, fields) for value in values]
    pairs.extend(failure.row(value, fields, path) for path, value in declarations)
    pairs.sort(key=lambda pair: (order[pair[0]['model']], pair[0]['batch']))
    args.out.mkdir(parents=True, exist_ok=True)
    name = 'DNN1_aitemplate.csv' if args.complete else 'DNN1_aitemplate_partial.csv'
    write_csv(args.out / name, [p[0] for p in pairs], fields)
    write_csv(args.out / 'DNN1_aitemplate_diag.csv', [p[1] for p in pairs])
    print(compact(dict(status='complete' if args.complete else 'partial', accepted=len(values),
        terminal=len(pairs), requested=25, csv=str(args.out / name), canonical150_unchanged=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--complete', action='store_true')
    parser.add_argument('--failure-declaration', type=Path, action='append', default=[],
                        help='explicit final N/A declaration with SHA-bound original failure evidence')
    publish(parser.parse_args())
