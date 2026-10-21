#!/usr/bin/env python3
"""Collect accepted same-card process pairs without substituting historical times."""
import argparse
import csv
import json
from pathlib import Path

from sm90d1_common import read


def sample(row):
    folder = Path(row['out'])
    guard = read(folder / 'guard_result.json')
    if guard.get('code') != 0:
        raise ValueError(f'outer guard rejected {folder}')
    selected = read(folder / 'job/selected-config.json')
    if selected['stage'] != 'measure':
        raise ValueError('correctness and diagnostic times are ineligible')
    paths = list((folder / 'job/measurements').glob('B*/measurements.json'))
    if len(paths) != 1:
        raise ValueError('one batch measurement is required')
    measurement = read(paths[0])
    identity = folder / 'job/physical-device-identity.json'
    if not identity.exists():
        identity = folder / identity.name
    return dict(folder=folder, selected=selected, measurement=measurement,
                identity=read(identity), tokens=read(folder / 'job/measurements/reference_tokens.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    state = read(a.state)
    rows = []
    for name, value in state.items():
        marker = '-reference-f'
        if marker not in name or value['status'] != 'done':
            continue
        selected=read(Path(value['out']) / 'job/selected-config.json')
        if selected['stage']!='measure':
            continue
        prefix, fresh = name.rsplit(marker, 1)
        reference = sample(value)
        for candidate_name, candidate_value in state.items():
            if (candidate_name == name or candidate_value['status'] != 'done' or
                    not candidate_name.startswith(prefix + '-') or
                    not candidate_name.endswith('-f' + fresh)):
                continue
            candidate = sample(candidate_value)
            if reference['identity']['cuda_uuid'] != candidate['identity']['cuda_uuid']:
                raise ValueError('a paired comparison must use the same physical card')
            if reference['measurement']['checkpoint'] != candidate['measurement']['checkpoint']:
                raise ValueError('paired checkpoint identity changed')
            before = reference['measurement']['tpot_seconds'] * 1000
            after = candidate['measurement']['tpot_seconds'] * 1000
            def tag(record):
                command=record['selected']['command']
                return command[command.index('--decode')+1].split('/build/')[-1].split('/')[0]
            rows.append(dict(pair=prefix, fresh_process=int(fresh),
                baseline_ms=before, candidate_ms=after, change_pct=(after / before - 1) * 100,
                token_equality=reference['tokens'] == candidate['tokens'],
                uuid=reference['identity']['cuda_uuid'],
                baseline_sha256=reference['selected']['decode_sha256'],
                candidate_sha256=candidate['selected']['decode_sha256'],
                baseline_tag=tag(reference),candidate_tag=tag(candidate),
                baseline_run=str(reference['folder']), candidate_run=str(candidate['folder']),
                ownership_verified=bool(reference['measurement']['ownership_verified'] and
                                        candidate['measurement']['ownership_verified'])))
    if not rows:
        raise ValueError('no complete accepted pairs')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
