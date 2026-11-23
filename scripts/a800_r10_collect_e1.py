#!/usr/bin/env python3
"""Select R10 cache arms using only completed, same-allocation pair receipts."""
import argparse
import csv
import json
import math
from pathlib import Path
import re

from a100_baselines import sha
from tilemega.bench.scheduler import write_json

from a800_r10_prepare import read


TERMINAL = {'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'PREEMPTED', 'REVOKED', 'TIMEOUT'}
ARMS = ('control', 'e1-device', *(f'e1-host-{n}' for n in (1, 2, 4)),
        *(f'e1-both-{n}' for n in (1, 2, 4)))


def collect(ledger, accounting, job, runs, destination):
    tasks = {}
    for line in accounting.read_text().splitlines():
        name, raw, state, exit_status = line.split('|')
        index = re.fullmatch(rf'{job}_(\d+)', name)
        if index:
            tasks[index.group(1)] = (raw, state.split()[0], exit_status)
    scope = read(ledger)
    if any(index not in tasks or tasks[index][1] not in TERMINAL for index in scope['items']):
        return False

    rows, selection = [], {}
    for index, entry in scope['rejected'].items():
        for arm in ARMS:
            rows.append(dict(model=entry['model'], batch=entry['batch'], arm=arm,
                             status='prepare_rejected', evidence=entry['error']))
        selection[f"{entry['model']}/B{entry['batch']}"] = {}

    for index, entry in scope['items'].items():
        plan = Path(entry['plan'])
        if sha(plan) != entry['sha256']:
            raise ValueError('E1 preparation changed after submission')
        options = read(plan)
        raw, state, exit_status = tasks[index]
        out = runs/'optimization/decode-pairs'/raw
        arms = {}
        source = None
        if state == 'COMPLETED' and exit_status == '0:0':
            source = out/'result.json'
            summary, guard = read(source), read(out/'planned.json')
            if guard['protocol_version'] != 'a800-single-v2':
                raise ValueError('cache trials used a different timing protocol')
            measurements = read(out/'rounds.json')
            variants = {row['arm']: row for row in measurements}
            control = variants['control']
            arms['control'] = control
            winner = ('control', control['tpot_seconds'])
            comparisons = summary.get('arms')
            if comparisons is None:
                if (summary['clears_tpot_screen'] and len(variants) != 2):
                    raise ValueError('two-arm E1 summary does not match its rounds')
                comparisons = {'trial': {'clears_tpot_screen': summary['clears_tpot_screen']}}
            for label, outcome in comparisons.items():
                name = options['labels']['trial'] if label == 'trial' else label
                sample = variants[label]
                checks = ('gpu_uuid', 'tokens_sha256', 'checkpoint_sha256', 'config_sha256', 'prompt_sha256')
                if any(sample[field] != control[field] for field in checks):
                    raise ValueError('E1 comparison changed its physical GPU, inputs or output')
                after = sample['tpot_seconds']
                if not math.isfinite(after) or after <= 0:
                    raise ValueError('invalid cache-arm TPOT')
                if outcome['clears_tpot_screen'] and after < min(winner[1], .995*control['tpot_seconds']):
                    winner = (name, after)
                arms[name] = sample
        else:
            winner = ('control', math.inf)
        label = winner[0]
        policy = {'weight_evict_first': label == 'e1-device' or label.startswith('e1-both')}
        policy['persist_l2_eighths'] = int(label[-1]) if label.startswith(('e1-host', 'e1-both')) else 0
        model, batch = options['model'], options['batch']
        selection[f'{model}/B{batch}'] = policy if any(policy.values()) else {}
        for arm in ARMS:
            sample = arms.get(arm, {})
            rows.append(dict(model=model, batch=batch, arm=arm, job=raw,
                             status='measured' if sample else state.lower() + '_no_measurement',
                             selected=bool(sample and arm == label),
                             tpot_ms=sample['tpot_seconds']*1000 if sample else '',
                             gpu_uuid=sample.get('gpu_uuid', ''),
                             evidence=sha(source) if source else exit_status))

    rows.sort(key=lambda row: (row['model'], int(row['batch']), ARMS.index(row['arm'])))
    destination.mkdir(parents=True, exist_ok=True)
    path = destination/'A800_R10_e1.csv'
    pending = path.with_suffix('.tmp')
    with pending.open('w', newline='') as output:
        writer = csv.DictWriter(output, ('model', 'batch', 'arm', 'job', 'status', 'selected',
                                         'tpot_ms', 'gpu_uuid', 'evidence'))
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)
    write_json(destination/'e1.selection.json', selection)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', required=True, type=Path)
    parser.add_argument('--job', required=True, type=int)
    parser.add_argument('--accounting', required=True, type=Path)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    return 0 if collect(args.ledger, args.accounting, args.job, args.runs, args.out) else 3


if __name__ == '__main__':
    raise SystemExit(main())
