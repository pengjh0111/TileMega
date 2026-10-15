#!/usr/bin/env python3
"""Collect completed guarded R16 runs without treating failed attempts as data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def collect(root):
    records, rejected = [], []
    for state_file in sorted((root/'runs/r16/scheduler').glob('gpu*/state.json')):
        state = json.loads(state_file.read_text())
        for name, job in state.items():
            if job.get('status') != 'done' or not name.startswith(('m1_', 'anchors_', 'm0_')):
                continue
            directory = Path(job['out'])
            if not directory.is_absolute():
                directory = root/directory
            guard_path = directory/'guard_result.json'
            guard = json.loads(guard_path.read_text()) if guard_path.exists() else {}
            if guard.get('code') != 0:
                rejected.append({'job': name, 'reason': 'guard did not accept completed run'})
                continue
            for path in sorted((directory/'measurements').glob('B*/measurements.json')):
                value = json.loads(path.read_text())
                if value.get('max_tokens') != 1024 or value.get('prompt_length') != 64:
                    rejected.append({'job': name, 'reason': 'not the final 64/1024 protocol'})
                    continue
                rows = value.get('generation_runs', [])
                valid = (all(sum(row['N'] == count and not row['warmup'] for row in rows) == 3
                             for count in (1, 1024)) and value.get('tpot_seconds', 0) > 0)
                if not valid:
                    rejected.append({'job': name, 'reason': 'incomplete measured repetitions'})
                    continue
                records.append({'job': name, 'phase': name.split('_')[0],
                    'model': Path(value['model']).name, 'system': value['system'],
                    'batch': value['batch'], 'round': value['round_index'],
                    'gpu': value['gpu_index'], 'e2e_ms': value['e2e_seconds']*1000,
                    'ttft_ms': value['ttft_seconds']*1000,
                    'tpot_ms': value['tpot_seconds']*1000,
                    'tokens_per_second': value['output_tokens_per_second'],
                    'acceptance': guard.get('acceptance'), 'checkpoint_identity_shared': False,
                    'measurement': str(path.relative_to(root))})
    return records, rejected


def aggregate(records):
    groups = {}
    for row in records:
        if row['phase'] != 'm1':
            continue
        key = row['model'], row['system'], row['batch']
        groups.setdefault(key, []).append(row)
    result = []
    for (model, system, batch), rows in sorted(groups.items()):
        rounds = {row['round'] for row in rows}
        if len(rounds) != len(rows):
            raise ValueError(f'duplicate completed process round for {model}/{system}/B{batch}')
        result.append({'model': model, 'system': system, 'batch': batch,
            'fresh_processes': len(rounds), 'protocol_complete': rounds == {0, 1, 2},
            'gpu': ','.join(map(str, sorted({row['gpu'] for row in rows}))),
            **{field: statistics.median(row[field] for row in rows)
               for field in ('e2e_ms', 'ttft_ms', 'tpot_ms', 'tokens_per_second')},
            'acceptance': 'strict' if all(row['acceptance'] == 'strict' for row in rows)
                           else 'provisional_container_pid_namespace',
            'checkpoint_identity_shared': False})
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out or root/'docs/results'
    out.mkdir(parents=True, exist_ok=True)
    records, rejected = collect(root)
    cells = aggregate(records)
    write_csv(out/'R16_llm_processes.csv', records)
    write_csv(out/'R16_llm_summary.csv', cells)
    print(json.dumps({'completed_process_cells': len(records), 'm1_cells': len(cells),
        'three_process_cells': sum(row['protocol_complete'] for row in cells),
        'strict_cells': sum(row['acceptance'] == 'strict' for row in cells),
        'rejected': rejected}, indent=2))


if __name__ == '__main__':
    main()
