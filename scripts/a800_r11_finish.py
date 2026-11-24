#!/usr/bin/env python3
"""Bind eligible phase winners for the final per-cell comparison against R10."""
import argparse
from collections import defaultdict
from pathlib import Path

from a100_baselines import inspect_inputs, sha
from a800_baselines import INVENTORY
from a800_r11_collect_round import collect
from a800_r11_gate import read
from tilemega.bench.scheduler import write_json


def completed(path):
    accounting = read(path/'accounting.json')
    for ledger in sorted(path.glob('*.rounds.json')):
        ticket = ledger.with_name(ledger.name.replace('.rounds.json', '.id'))
        if not ticket.exists():
            continue
        parent = ticket.read_text().strip()
        for index, item in enumerate(read(ledger)['cells']):
            outcome = accounting.get(parent, {}).get(str(index), {})
            if outcome.get('state') not in ('COMPLETED', 'FAILED'):
                continue
            if sha(item['plan']) != item['sha256']:
                raise ValueError('a measured cell changed its allocation plan')
            yield ledger.name.split('-')[0], read(item['plan']), outcome['job']


def select(waves, runs, inputs):
    cells = defaultdict(dict)
    for phase, plan, job in completed(waves):
        if phase == 'final':
            continue
        folder = runs/'optimization/decode-pairs'/job
        if not (folder/'result.json').exists():
            continue
        manifest = read(plan['finalists'])
        if sha(plan['finalists']) != plan['finalists_sha256']:
            raise ValueError('the measured shortlist changed')
        named = {('trial' if row['rank'] == 1 else f'rank-{row["rank"]}'): row for row in manifest['candidates']}
        observations = [(fact, data) for fact, data, _ in collect(runs, job, inputs) if data is not None]
        if not observations:
            continue
        # Ranking here only compares arms in this allocation. The surviving
        # arms from different phases will meet in the final single-GPU job.
        fact, _ = min(observations, key=lambda pair: pair[0]['tpot_ms'])
        chosen = named[fact['arm']]
        if chosen['binary_sha256'] != fact['decode_sha256']:
            raise ValueError('the surviving trial differs from the measured binary')
        cell = fact['model'], fact['batch']
        if phase in cells[cell]:
            raise ValueError(f'multiple terminal jobs for {cell} {phase}; supply one accepted allocation')
        cells[cell][phase] = dict(chosen, job=job, evidence_sha256=fact['evidence'],
            target=chosen.get('target', manifest['target']),
            target_sha256=chosen.get('target_sha256', manifest['target_sha256']),
            passed_screen=fact['status'] == 'faster')
    return cells


def bind(out, incumbent, phases):
    control = read(incumbent)
    chosen = [row for _, row in sorted(phases.items()) if row['passed_screen']]
    if not chosen and 'pipeline' in phases:
        # Retest the best eligible solver finalist against the incumbent even
        # when the first timing found a tie or loss.
        chosen = [phases['pipeline']]
    rows, seen = [], {control['decode_sha256']}
    for row in chosen:
        digest = row['binary_sha256']
        if digest not in seen:
            seen.add(digest)
            rows.append(dict(row, rank=len(rows)+1))
    if not rows:
        raise ValueError('no distinct numeric-gate trial is available for the final paired test')
    out.mkdir(parents=True, exist_ok=True)
    scope = out/'prior-allocations.json'
    write_json(scope, phases)
    write_json(out/'finalists.json', dict(model=control['model'], batch=control['batch'],
        control=str(incumbent), control_sha256=sha(incumbent), prepared=str(scope),
        prepared_sha256=sha(scope), target=control['target'], target_sha256=control['target_sha256'],
        candidates=rows, routing_export=control['export']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('waves', 'runs', 'incumbents', 'out'):
        parser.add_argument(f'--{option}', required=True, type=Path)
    args = parser.parse_args()
    cells = select(args.waves, args.runs, inspect_inputs(INVENTORY))
    for (model, batch), phases in sorted(cells.items()):
        bind(args.out/model/f'B{batch}', args.incumbents/model/f'B{batch}'/'incumbent.json', phases)


if __name__ == '__main__':
    main()
