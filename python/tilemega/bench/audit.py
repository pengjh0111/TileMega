"""Audit accepted scheduler attempts without mixing rejected measurements."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .arms import sha256
from .report import discernible, metrics, summarize


def accepted_rounds(state_path, prefix, rounds=3):
    state = json.loads(Path(state_path).read_text())
    selected = {name: row for name, row in state.items() if name.startswith(prefix)}
    if not selected or any(row['status'] != 'done' for row in selected.values()):
        raise ValueError('comparison steps have not all completed successfully')
    groups, receipts = defaultdict(list), []
    for name, row in sorted(selected.items()):
        folder = Path(row['out'])
        guard_path, round_path = folder / 'guard.jsonl', folder / 'round.json'
        guard = [json.loads(line) for line in guard_path.read_text().splitlines()]
        if not guard or guard[-1].get('phase') != 'after' or guard[-1].get('exit_code') != 0:
            raise ValueError('attempt lacks successful guard completion: ' + name)
        data = json.loads(round_path.read_text())
        if data.get('invalidated') or any(arm['exit_code'] for arm in data['arms'].values()):
            raise ValueError('attempt contains an invalid measurement: ' + name)
        receipts.append(dict(step=name, attempt=row['attempts'], round=str(round_path),
            round_sha256=sha256(round_path), guard=str(guard_path), guard_sha256=sha256(guard_path)))
        for label, arm in data['arms'].items():
            command_path = folder / label / 'command.json'
            command = json.loads(command_path.read_text())
            binaries = {phase: item['sha256'] for phase, item in command['binaries'].items()}
            tokens = folder / label / f'tokens_N{data["workload"]["osl"]}_run1.json'
            groups[data['cell'], label].append(dict(round=data['round'],
                **metrics(arm, data['workload']), binaries=binaries,
                tokens_sha256=sha256(tokens), command_sha256=sha256(command_path)))
    cells = defaultdict(dict)
    for (cell, label), samples in sorted(groups.items()):
        if len(samples) != rounds or len({item['round'] for item in samples}) != rounds:
            raise ValueError('missing or duplicated comparison rounds: ' + cell + '/' + label)
        if any(item['binaries'] != samples[0]['binaries'] for item in samples):
            raise ValueError('binary identity changed between rounds: ' + cell + '/' + label)
        cells[cell][label] = dict(samples=samples,
            tpot_ms=summarize([item['tpot_seconds'] * 1000 for item in samples]),
            request_seconds=summarize([item['request_seconds'] for item in samples]))
    for cell, arms in cells.items():
        rounds_seen = {label: {item['round'] for item in arm['samples']} for label, arm in arms.items()}
        if any(value != next(iter(rounds_seen.values())) for value in rounds_seen.values()):
            raise ValueError('unpaired comparison rounds: ' + cell)
    return dict(counts=dict(Counter(row['status'] for row in state.values())),
                state_sha256=sha256(state_path), accepted_attempts=receipts, cells=dict(cells))


def compare(result, reference, candidate):
    rows = []
    for cell, arms in sorted(result['cells'].items()):
        a, b = arms[reference]['tpot_ms'], arms[candidate]['tpot_ms']
        rows.append(dict(cell=cell, reference=reference, candidate=candidate,
            reference_ms=a['median'], candidate_ms=b['median'],
            relative=b['median'] / a['median'] - 1,
            discernible=discernible(b['samples'], a['samples']),
            tokens_identical=len({sample['tokens_sha256'] for label in (reference, candidate)
                for sample in arms[label]['samples']}) == 1))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--compare', nargs=2, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = accepted_rounds(args.state, args.prefix)
    result['comparisons'] = [row for pair in args.compare for row in compare(result, *pair)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(counts=result['counts'], comparisons=result['comparisons'])))


if __name__ == '__main__':
    main()
