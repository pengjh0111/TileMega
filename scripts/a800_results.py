#!/usr/bin/env python3
"""Collect accepted independent requests from one A800 UUID."""
import argparse
import json
from pathlib import Path
import statistics

from a100_baselines import BATCHES, ENGINES, MODELS, ROOT, inspect_inputs
from a100_results import measurement, read, write_csv
from a800_baselines import INVENTORY, RUN


METRICS = ('tpot_ms', 't1_s', 't1024_s', 'tok_s')
OOM_MARKERS = ('CUDA out of memory', 'torch.OutOfMemoryError', 'CUDA_ERROR_OUT_OF_MEMORY')


def metrics(rows, count):
    if len(rows) != count:
        return dict.fromkeys(METRICS, '')
    return {name: statistics.median(row[name] for row in rows) for name in METRICS}


def summarize(rows, key):
    values = sorted({row[key] for row in rows})
    if len(values) <= 1:
        return values[0] if values else ''
    return json.dumps([json.loads(value) for value in values])


def baseline_cell(run, inputs, state, scope, model, engine, batch):
    cell = model, engine, batch
    jobs = scope.get(cell, [])
    rows = [row for job in jobs if (row := measurement(job, state.get(job['name'], {}), inputs))]
    names = [f'{model}_{engine}_b{batch:02d}_r{i}' for i in range(3)]
    def confirmed_oom(name):
        receipt = read(run/f'{name}.oom.json')
        if receipt:
            return receipt['status'] == 'OOM' and receipt['device_uuid'] == inputs['device']['uuid']
        if state.get(name, {}).get('status') != 'failed':
            return False
        log = run/'baseline'/f'{name}.log'
        latest = log.read_text(errors='replace').rsplit('\nATTEMPT ', 1)[-1] if log.is_file() else ''
        return any(mark in latest for mark in OOM_MARKERS)

    oom = all(confirmed_oom(name) for name in names)
    needed = 3
    status = 'measured' if len(rows) == needed else 'OOM' if oom else 'incomplete' if rows else 'pending'
    return rows, dict(model=model, framework=engine, batch=batch, status=status, processes=len(rows),
        required_processes=needed, correctness='exact token counts' if rows else 'pending',
        **metrics(rows, needed), backend=summarize(rows, 'backend'), versions=summarize(rows, 'versions'),
        gpu_uuid=inputs['device']['uuid'], config_sha256=inputs['models'][model]['config_sha256'],
        prompt_sha256=inputs['models'][model]['prompt_sha256'], checkpoint_sha256='',
        harness_sha256=' '.join(sorted({job['harness_sha256'] for job in jobs})),
        receipt_sha256=' '.join(row['receipt'] for row in rows), evidence=' '.join(names))


def native_cell(inputs, model, batch):
    uuid = inputs['device']['uuid']
    run = RUN/'native'/uuid/model/f'B{batch}'
    receipts = [read(path) for path in sorted(run.glob('measure-*.json'))]
    rows, artifact_ids = [], set()
    for receipt in receipts:
        if not receipt.get('pass') or receipt['device_uuid'] != uuid:
            raise ValueError('native measurement receipt did not pass on the cohort UUID')
        attempt = Path(receipt['evidence']).parent.parent
        job = dict(name=f'{model}-native-r{receipt["round"]}', model=model, batch=batch,
                   round=receipt['round'])
        rows.append(measurement(job, dict(status='done', out=str(attempt)), inputs))
        artifact_ids.add((receipt['prefill_sha256'], receipt['decode_sha256'], receipt['target_sha256']))
    if len(artifact_ids) > 1:
        raise ValueError(f'{model} B{batch} contains more than one native plan pair')
    needed = 3 if model in tuple(MODELS)[:4] else 5
    if len(rows) > needed:
        raise ValueError('choose an explicit cohort without extra formal rounds')
    status = 'measured' if len(rows) == needed else 'incomplete' if rows else 'pending_correctness'
    artifacts = next(iter(artifact_ids), ('', '', ''))
    return rows, dict(model=model, batch=batch, status=status, processes=len(rows), required_processes=needed,
        correctness='five fresh checks passed' if rows else 'pending',
        **metrics(rows, needed), backend='SM80 native' if rows else '', fastest_baseline='',
        fastest_baseline_tpot_ms='', speedup='', config_sha256=inputs['models'][model]['config_sha256'],
        prompt_sha256=inputs['models'][model]['prompt_sha256'], checkpoint_sha256='',
        prefill_sha256=artifacts[0], decode_sha256=artifacts[1], target_sha256=artifacts[2],
        gpu_uuid=uuid, versions=summarize(rows, 'versions'),
        receipt_sha256=' '.join(row['receipt'] for row in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT/'docs/results')
    args = p.parse_args()
    if not args.cohort or not all(c.isascii() and (c.isalnum() or c in '-_') for c in args.cohort):
        p.error('invalid cohort name')
    dev = read(args.target)['device_provenance']
    if 'A800' not in dev['name']:
        p.error('supply the guarded A800 target')
    run = RUN/'cohorts'/args.cohort
    recorded = read(run/'baseline_inputs.json')
    inputs = dict(device=dev, models=inspect_inputs(INVENTORY))
    if recorded and (recorded['device']['uuid'] != dev['uuid'] or recorded['models'] != inputs['models']):
        raise ValueError('baseline and native inputs differ')
    jobs = read(run/'queue/queue_baselines.json')
    state, scope = read(run/'baseline/state.json'), {}
    for job in jobs:
        if 'model' in job:
            scope.setdefault((job['model'], job['engine'], job['batch']), []).append(job)
    baseline, native = [], []
    for model in MODELS:
        all_rows, checkpoint_hashes = [], set()
        for batch in BATCHES:
            measured = []
            for engine in ENGINES:
                rows, cell = baseline_cell(run, inputs, state, scope, model, engine, batch)
                checkpoint_hashes.update(row['checkpoint_sha256'] for row in rows)
                measured.append(cell)
                baseline.append(cell)
                all_rows.append(cell)
            rows, cell = native_cell(inputs, model, batch)
            checkpoint_hashes.update(row['checkpoint_sha256'] for row in rows)
            if all(row['status'] in {'measured', 'OOM', 'N/A'} for row in measured):
                valid = [r for r in measured if r['status'] == 'measured']
                if valid:
                    fastest = min(valid, key=lambda row: row['tpot_ms'])
                    cell.update(fastest_baseline=fastest['framework'],
                                fastest_baseline_tpot_ms=fastest['tpot_ms'])
                    if cell['status'] == 'measured':
                        cell['speedup'] = fastest['tpot_ms']/cell['tpot_ms']
            native.append(cell)
            all_rows.append(cell)
        if len(checkpoint_hashes) > 1:
            raise ValueError(f'shared checkpoint changed for {model}')
        for row in all_rows:
            row['checkpoint_sha256'] = next(iter(checkpoint_hashes), '')
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/'A800_baselines.csv', baseline)
    write_csv(args.out/'A800_llm.csv', native)


if __name__ == '__main__':
    main()
