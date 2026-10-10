#!/usr/bin/env python3
"""DM-1 §8.G: seal solver winners before the comparison matrix."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def series(values):
    if len(values) != 3 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('expected three positive finite round medians')
    return statistics.median(values), max(values) - min(values)


def distinguishably_faster(candidate, base):
    c, cr = series(candidate)
    b, br = series(base)
    return b - c > max(cr, br)


def seal(search):
    choices = {}
    for cell, result in sorted(search.items()):
        if result.get('source') != 'solver_second_level':
            raise ValueError(cell + ': selection must come from solver second level')
        candidates = result['candidates']
        ids = [r['identity_id'] for r in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError(cell + ': duplicate candidate identity')
        winner = next((r for r in candidates if r['identity_id'] == result['winner']), None)
        if winner is None or winner.get('role') != 'solver_candidate':
            raise ValueError(cell + ': winner is absent or is a control arm')
        if winner.get('correctness_passed') is not True:
            raise ValueError(cell + ': winner lacks its correctness gate')
        identity = winner['identity_id']
        if len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
            raise ValueError(cell + ': winner needs a SHA256 identity')
        choices[cell] = dict(arm='TM', identity_id=identity,
                             source='solver_second_level', winner=winner)
    if not choices:
        raise ValueError('no solver winners')
    return dict(rule='DM-1 §8.G', choices=choices,
                selection_sha256=hashlib.sha256(json.dumps(
                    choices, sort_keys=True, separators=(',', ':')).encode()).hexdigest())


def audit(selection, matrix):
    expected = seal({cell: dict(source='solver_second_level',
                              candidates=[r['winner']], winner=r['identity_id'])
                     for cell, r in selection['choices'].items()})
    if selection != expected:
        raise ValueError('sealed solver selection was changed')
    reports = {}
    for cell, choice in selection['choices'].items():
        arms = matrix.get(cell, {})
        if 'TM' not in arms:
            reports[cell] = dict(status='missing', missing=['TM'])
            continue
        tm = arms['TM']
        if tm['identity_id'] != choice['identity_id']:
            raise ValueError(cell + ': matrix TM differs from sealed winner')
        if tm.get('correctness_passed') is not True:
            raise ValueError(cell + ': TM correctness gate failed')
        rounds = tm['rounds']
        if len(rounds) != 3 or len(set(rounds)) != 3:
            raise ValueError(cell + ': expected three distinct rounds')
        series(tm['latency_ms'])
        faster = []
        for arm, row in sorted(arms.items()):
            if row.get('role') != 'control':
                continue
            if row.get('correctness_passed') is not True or row['rounds'] != rounds:
                raise ValueError(cell + ': controls require matching rounds and gates')
            if distinguishably_faster(row['latency_ms'], tm['latency_ms']):
                faster.append(dict(arm=arm, identity_id=row['identity_id'],
                                   analysis_required=True,
                                   possible_causes=['solver omission', 'measurement noise']))
        reports[cell] = dict(status='checked', identity_id=choice['identity_id'],
                             faster_controls=faster, selection_changed=False)
    return dict(selection_sha256=selection['selection_sha256'], cells=reports)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('seal', 'audit'), required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if args.stage == 'seal':
        result = seal(data)
    else:
        if args.selection is None:
            parser.error('audit requires --selection')
        result = audit(json.loads(args.selection.read_text()), data)
    if args.out.exists():
        raise FileExistsError('refuse to overwrite preregistered selection or audit')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
