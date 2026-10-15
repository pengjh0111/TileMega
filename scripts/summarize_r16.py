#!/usr/bin/env python3
"""Collect completed guarded R16 runs without treating failed attempts as data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from r16_measurement_validation import completed_guard, validate_llm, validate_dnn, canonical_model


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
            try:
                guard = completed_guard(directory, job)
            except (OSError, ValueError, KeyError, TypeError) as error:
                rejected.append({'job': name, 'reason': str(error)})
                continue
            for path in sorted((directory/'measurements').glob('B*/measurements.json')):
                try:
                    value, _, own = validate_llm(path, guard, root)
                    if value['gpu_index'] != int(state_file.parent.name.removeprefix('gpu')):
                        raise ValueError('receipt GPU differs from scheduler card')
                except (OSError, ValueError, KeyError, TypeError) as error:
                    rejected.append({'job': name, 'reason': str(error), 'measurement': str(path)})
                    continue
                records.append({'job': name, 'phase': name.split('_')[0],
                    'model': value.get('_r16_native', {}).get('canonical_model', canonical_model(root, value['model_config_sha256'], Path(value['model']).name)), 'system': value['system'],
                    'batch': value['batch'], 'round': value['round_index'],
                    'gpu': value['gpu_index'], 'e2e_ms': value['e2e_seconds']*1000,
                    'ttft_ms': value['ttft_seconds']*1000,
                    'tpot_ms': value['tpot_seconds']*1000,
                    'tokens_per_second': value['output_tokens_per_second'],
                    'acceptance': own['acceptance'], 'ownership_verified': own['ownership_verified'],
                    'unresolved_host_pids': ','.join(map(str, own['unresolved_host_pids'])),
                    'checkpoint_identity_shared': value.get('_r16_native', {}).get('checkpoint_identity_shared', False),
                    'source_checkpoint_sha256': value.get('_r16_native', {}).get('source_checkpoint_sha256', ''),
                    'source_checkpoint_label': value.get('_r16_native', {}).get('source_checkpoint_label',
                        'backend-native dummy' if value.get('load_format') == 'dummy' else 'local checkpoint identity unavailable'),
                    'native_artifact_ids': value.get('_r16_native', {}).get('native_artifact_ids', ''),
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
            'checkpoint_identity_shared': all(row['checkpoint_identity_shared'] for row in rows),
            'source_checkpoint_sha256': ','.join(sorted({row['source_checkpoint_sha256'] for row in rows if row['source_checkpoint_sha256']})),
            'native_artifact_ids': ','.join(sorted({row['native_artifact_ids'] for row in rows if row['native_artifact_ids']}))})
    return result


def collect_dnn(root):
    records, rejected = [], []
    for state_file in sorted((root/'runs/r16/scheduler').glob('gpu*/state.json')):
        for name, job in json.loads(state_file.read_text()).items():
            if job.get('status') != 'done' or not name.startswith(('dnn-', 'tm-dnn-', 'm3_')) or name.startswith('dnn-smoke-'):
                continue
            directory = Path(job['out'])
            if not directory.is_absolute():
                directory = root/directory
            path = directory/'measurements/measurements.json'
            try:
                guard = completed_guard(directory, job)
                value, own = validate_dnn(path, guard)
                if value['gpu_index'] != int(state_file.parent.name.removeprefix('gpu')):
                    raise ValueError('receipt GPU differs from scheduler card')
            except (OSError, ValueError, KeyError, TypeError) as error:
                rejected.append({'job': name, 'reason': str(error)})
                continue
            records.append(dict(job=name, model=value['model'], system=value['system'], batch=value['batch'],
                precision=value['precision'], compile_mode=value.get('configuration', {}).get('compile_mode', ''),
                round=value['round_index'], gpu=value['gpu_index'], median_ms=value['median_ms'], p90_ms=value['p90_ms'],
                acceptance=own['acceptance'], ownership_verified=own['ownership_verified'],
                unresolved_host_pids=','.join(map(str, own['unresolved_host_pids'])),
                ownership_metadata_source='completed guard and per-measurement telemetry',
                random_weights=value.get('random_weights'), exported_program_sha256=value.get('exported_program_sha256', ''),
                native_artifact_ids=value.get('_r16_native', {}).get('native_artifact_ids', ''),
                source_checkpoint_label=value.get('_r16_native', {}).get('source_checkpoint_label', 'stored export checkpoint'),
                measurement=str(path.relative_to(root))))
    return records, rejected


def write_csv(path, rows):
    if not rows:
        path.write_text('')
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
    dnn_records, dnn_rejected = collect_dnn(root)
    rejected.extend(dnn_rejected)
    cells = aggregate(records)
    write_csv(out/'R16_llm_processes.csv', records)
    write_csv(out/'R16_llm_summary.csv', cells)
    write_csv(out/'R16_dnn_processes.csv', dnn_records)
    print(json.dumps({'completed_process_cells': len(records), 'm1_cells': len(cells),
        'three_process_cells': sum(row['protocol_complete'] for row in cells),
        'strict_cells': sum(row['acceptance'] == 'strict' for row in cells),
        'completed_dnn_processes': len(dnn_records),
        'rejected': rejected}, indent=2))


if __name__ == '__main__':
    main()
