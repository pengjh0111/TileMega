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


def baseline_scope(cohorts, inputs):
    scope = {}
    for run in cohorts:
        recorded = read(run/'baseline_inputs.json')
        if not recorded:
            continue
        if recorded['device']['uuid'] != inputs['device']['uuid'] or recorded['models'] != inputs['models']:
            raise ValueError(f'baseline and native inputs differ for {run.name}')
        state = read(run/'baseline/state.json')
        for job in read(run/'queue/queue_baselines.json'):
            if 'model' not in job:
                continue
            name = job['name']
            step = state.get(name, {})
            capacity = run/f'{name}.oom.json'
            if step.get('out') or capacity.is_file():
                key = job['model'], job['engine'], job['batch']
                scope.setdefault(key, []).append((run, job, step, capacity))
    return scope


def confirmed_oom(run, job, step, capacity, inputs):
    receipt = read(capacity)
    if receipt:
        return receipt['status'] == 'OOM' and receipt['device_uuid'] == inputs['device']['uuid']
    if step.get('status') != 'failed':
        return False
    log = run/'baseline'/f'{job["name"]}.log'
    latest = log.read_text(errors='replace').rsplit('\nATTEMPT ', 1)[-1] if log.is_file() else ''
    return any(mark in latest for mark in OOM_MARKERS)


def baseline_cell(inputs, jobs, model, engine, batch):
    accepted, oom_rounds, evidence, harnesses = {}, set(), set(), set()
    for run, job, step, capacity in jobs:
        repeat = job['round']
        harnesses.add(job['harness_sha256'])
        if step.get('out'):
            evidence.add(str(Path(step['out']).relative_to(RUN)))
        if row := measurement(job, step, inputs):
            if repeat in accepted and accepted[repeat] != row:
                raise ValueError(f'duplicate accepted round for {job["name"]}; select one cohort')
            accepted[repeat] = row
        elif confirmed_oom(run, job, step, capacity, inputs):
            oom_rounds.add(repeat)
            if capacity.is_file():
                evidence.add(str(capacity.relative_to(RUN)))
    rows = [accepted[key] for key in sorted(accepted)]
    needed = 3
    status = ('measured' if len(rows) == needed else 'incomplete' if rows else
              'OOM' if len(oom_rounds) == needed else 'attempted' if evidence else 'pending')
    return rows, dict(model=model, framework=engine, batch=batch, status=status, processes=len(rows),
        required_processes=needed, correctness='exact token counts' if rows else 'pending',
        **metrics(rows, needed), backend=summarize(rows, 'backend'), versions=summarize(rows, 'versions'),
        gpu_uuid=inputs['device']['uuid'], config_sha256=inputs['models'][model]['config_sha256'],
        prompt_sha256=inputs['models'][model]['prompt_sha256'], checkpoint_sha256='',
        harness_sha256=' '.join(sorted(harnesses)),
        receipt_sha256=' '.join(row['receipt'] for row in rows), evidence=' '.join(sorted(evidence)))


def native_cell(inputs, model, batch):
    uuid = inputs['device']['uuid']
    run = RUN/'native'/uuid/model/f'B{batch}'
    receipts = [read(path) for path in sorted(run.glob('*/measure-*.json'))]
    checks = [read(path) for pattern in ('correctness-*.json', '*/correctness-*.json')
              for path in run.glob(pattern)]
    accepted_checks = sum(bool(row.get('pass')) for row in checks)
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
    if len(rows) > needed or len({r['round'] for r in receipts}) != len(rows):
        raise ValueError('choose an explicit cohort without extra formal rounds')
    status = 'measured' if len(rows) == needed else 'incomplete' if rows else 'pending_correctness'
    artifacts = next(iter(artifact_ids), ('', '', ''))
    return rows, dict(model=model, batch=batch, status=status, processes=len(rows), required_processes=needed,
        correctness='five fresh checks passed' if rows else 'pilot passed' if accepted_checks else 'pending',
        accepted_correctness_processes=accepted_checks,
        **metrics(rows, needed), backend='SM80 native' if rows else '', fastest_baseline='',
        fastest_baseline_tpot_ms='', speedup='', config_sha256=inputs['models'][model]['config_sha256'],
        prompt_sha256=inputs['models'][model]['prompt_sha256'], checkpoint_sha256='',
        prefill_sha256=artifacts[0], decode_sha256=artifacts[1], target_sha256=artifacts[2],
        gpu_uuid=uuid, versions=summarize(rows, 'versions'),
        receipt_sha256=' '.join(row['receipt'] for row in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', action='append', required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT/'docs/results')
    args = p.parse_args()
    if any(not name or not all(c.isascii() and (c.isalnum() or c in '-_') for c in name)
           for name in args.cohort):
        p.error('invalid cohort name')
    dev = read(args.target)['device_provenance']
    if 'A800' not in dev['name']:
        p.error('supply the guarded A800 target')
    inputs = dict(device=dev, models=inspect_inputs(INVENTORY))
    scope = baseline_scope((RUN/'cohorts'/name for name in args.cohort), inputs)
    baseline, native = [], []
    for model in MODELS:
        all_rows, checkpoint_hashes = [], set()
        for batch in BATCHES:
            measured = []
            for engine in ENGINES:
                rows, cell = baseline_cell(inputs, scope.get((model, engine, batch), []), model, engine, batch)
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
