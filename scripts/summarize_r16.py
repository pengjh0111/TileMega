#!/usr/bin/env python3
"""Collect completed guarded R16 runs without treating failed attempts as data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from r16_measurement_validation import completed_guard, validate_llm, validate_dnn, canonical_model


def record_directory(root, name, record):
    path=Path(record['out'])
    directory=path if path.is_absolute() else root/path
    if (directory/'guard_result.json').exists() or path.is_absolute():
        return directory
    provenance=root/'runs/r16/analysis/scheduler-cwd-provenance.json'
    if provenance.is_file():
        original=json.loads(provenance.read_text()).get('jobs',{}).get(name)
        if original:
            actual=(Path(original['cwd'])/path).resolve()
            if root/'runs/r16' not in actual.parents:
                raise ValueError('recorded job cwd resolves outside R16 evidence')
            if (actual/'guard_result.json').exists():
                return actual
    return directory


def collect(root):
    records, rejected = [], []
    for state_file in sorted((root/'runs/r16/scheduler').glob('gpu*/state.json')):
        state = json.loads(state_file.read_text())
        for name, job in state.items():
            if job.get('status') != 'done' or not name.startswith(('m1_', 'anchors_', 'm0_')):
                continue
            directory = record_directory(root, name, job)
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
                    'pid': value['pid'],
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
        key = (row['model'], row['system'], row['batch'],
               row.get('native_artifact_ids', ''), row.get('source_checkpoint_sha256', ''))
        groups.setdefault(key, []).append(row)
    result = []
    for (model, system, batch, artifacts, checkpoint), rows in sorted(groups.items()):
        rounds = {row['round'] for row in rows}
        if len(rounds) != len(rows):
            raise ValueError(f'duplicate completed process round for {model}/{system}/B{batch}')
        pids = {row['pid'] for row in rows if row.get('pid') is not None}
        if len(pids) != len(rows):
            raise ValueError(f'LLM rounds lack distinct measured process identities for {model}/{system}/B{batch}')
        result.append({'model': model, 'system': system, 'batch': batch,
            'fresh_processes': len(rounds), 'protocol_complete': rounds == {0, 1, 2},
            'gpu': ','.join(map(str, sorted({row['gpu'] for row in rows}))),
            **{field: statistics.median(row[field] for row in rows)
               for field in ('e2e_ms', 'ttft_ms', 'tpot_ms', 'tokens_per_second')},
            'acceptance': 'strict' if all(row['acceptance'] == 'strict' for row in rows)
                           else 'provisional_container_pid_namespace',
            'checkpoint_identity_shared': all(row['checkpoint_identity_shared'] for row in rows),
            'source_checkpoint_sha256': checkpoint, 'native_artifact_ids': artifacts})
    return result


def collect_dnn(root):
    records, rejected = [], []
    definitions={job['name']:job for queue in (root/'runs/r16/queue').glob('gpu*/queue_*.json')
                 for job in json.loads(queue.read_text())}
    for state_file in sorted((root/'runs/r16/scheduler').glob('gpu*/state.json')):
        for name, job in json.loads(state_file.read_text()).items():
            if job.get('status') != 'done' or not name.startswith(('dnn-', 'tm-dnn-', 'm3_', 'h7a-')) or name.startswith('dnn-smoke-'):
                continue
            definition=definitions.get(name)
            if definition is not None and not definition.get('timing',definition.get('gpu',False)):
                continue
            directory = record_directory(root, name, job)
            path = directory/'measurements/measurements.json'
            try:
                guard = completed_guard(directory, job)
                value, own = validate_dnn(path, guard)
                adjustment = dnn_runtime_adjustment(value)
                device_identity = dnn_device_identity(value, directory)
                if value['gpu_index'] != int(state_file.parent.name.removeprefix('gpu')):
                    raise ValueError('receipt GPU differs from scheduler card')
            except (OSError, ValueError, KeyError, TypeError) as error:
                rejected.append({'job': name, 'reason': str(error)})
                continue
            records.append(dict(job=name, model=value['model'], system=value['system'], batch=value['batch'],
                precision=value['precision'], compile_mode=value.get('configuration', {}).get('compile_mode', ''),
                candidate_id=value.get('candidate_id', value.get('configuration', {}).get('candidate_id', '')),
                candidate_receipt_sha256=value.get('candidate_receipt_sha256', ''), pid=value.get('pid'),
                temporary_workaround=bool(adjustment), runtime_adjustment=adjustment,
                **device_identity,
                source_checkpoint_sha256=value.get('source_checkpoint_sha256', ''),
                round=value['round_index'], gpu=value['gpu_index'], median_ms=value['median_ms'], p90_ms=value['p90_ms'],
                acceptance=own['acceptance'], ownership_verified=own['ownership_verified'],
                unresolved_host_pids=','.join(map(str, own['unresolved_host_pids'])),
                ownership_metadata_source='completed guard and per-measurement telemetry',
                random_weights=value.get('random_weights'), exported_program_sha256=value.get('exported_program_sha256', ''),
                native_artifact_ids=value.get('_r16_native', {}).get('native_artifact_ids', ''),
                source_checkpoint_label=value.get('_r16_native', {}).get('source_checkpoint_label', 'stored export checkpoint'),
                measurement=str(path.relative_to(root))))
    return records, rejected


def dnn_device_identity(value, directory):
    """Only identities measured by this process establish a physical card."""
    candidates = []
    cache = value.get('configuration', {}).get('engine_cache', {}).get('identity', {})
    if cache.get('gpu_uuid'):
        candidates.append(str(cache['gpu_uuid']).removeprefix('GPU-').lower())
    proof_path = directory/'hardware-epoch.json'
    boot = ''
    if proof_path.exists():
        proof = json.loads(proof_path.read_text())
        if proof.get('pid') != value.get('pid') or not proof.get('cuda_uuid'):
            raise ValueError('hardware epoch proof does not identify measured process')
        candidates.append(str(proof['cuda_uuid']).removeprefix('GPU-').lower())
        boot = proof.get('boot_id', '')
    if len(set(candidates)) > 1:
        raise ValueError('measured physical GPU identities disagree')
    return dict(physical_gpu_uuid=candidates[0] if candidates else '',
                physical_gpu_identity_status='verified' if candidates else 'unknown',
                hardware_boot_id=boot)


def dnn_runtime_adjustment(value):
    key = 'runtime_shared_memory_optin_workaround'
    top = value.get(key)
    nested = value.get('configuration', {}).get(key)
    if top is None and nested is None:
        return ''
    if not isinstance(top, dict) or top != nested:
        raise ValueError('runtime workaround metadata is absent or contradictory')
    if (top.get('enabled') is not True or top.get('temporary_workaround') is not True or
            top.get('artifact_id') != value.get('artifact_id') or
            top.get('library_sha256') != value.get('identity', {}).get('binary_sha256') or
            not top.get('driver_sha256') or not top.get('production_fix_commit')):
        raise ValueError('runtime workaround does not identify the measured artifact and repair')
    return json.dumps(top, sort_keys=True, separators=(',', ':'))


def aggregate_dnn(records):
    groups = {}
    for row in records:
        key = tuple(row[field] for field in ('model', 'system', 'batch', 'precision',
            'compile_mode', 'candidate_id', 'candidate_receipt_sha256', 'native_artifact_ids',
            'exported_program_sha256'))
        # An index is not a physical identity across container reconnects.
        device_key = (row.get('physical_gpu_uuid', ''), row.get('hardware_boot_id', ''))
        if not device_key[0]:
            device_key = ('unknown-index-' + str(row['gpu']), device_key[1])
        groups.setdefault((key, device_key), []).append(row)
    result = []
    for (key, device_key), rows in sorted(groups.items()):
        rounds = {row['round'] for row in rows}
        pids = {row['pid'] for row in rows if row['pid'] is not None}
        if len(rounds) != len(rows):
            raise ValueError(f'duplicate DNN process round for {key}')
        if len(pids) != len(rows):
            raise ValueError(f'DNN rounds lack distinct measured process identities for {key}')
        checkpoint_hashes = {row.get('source_checkpoint_sha256', '') for row in rows}
        checkpoint_hashes.discard('')
        if len(checkpoint_hashes) > 1:
            raise ValueError(f'DNN checkpoint hashes disagree for the same export for {key}')
        adjustments = {row.get('runtime_adjustment', '') for row in rows}
        if len(adjustments) != 1:
            raise ValueError(f'DNN runtime workaround differs between fresh rounds for {key}')
        result.append(dict(zip(('model', 'system', 'batch', 'precision', 'compile_mode',
            'candidate_id', 'candidate_receipt_sha256', 'native_artifact_ids', 'exported_program_sha256'), key),
            source_checkpoint_sha256=next(iter(checkpoint_hashes), ''),
            runtime_adjustment=next(iter(adjustments)),
            temporary_workaround=bool(next(iter(adjustments))),
            checkpoint_hash_processes=sum(bool(row.get('source_checkpoint_sha256')) for row in rows),
            fresh_processes=len(rows), protocol_complete=rounds == {0, 1, 2},
            physical_gpu_uuid=rows[0].get('physical_gpu_uuid', ''),
            physical_gpu_identity_status=rows[0].get('physical_gpu_identity_status', 'unknown'),
            hardware_boot_id=rows[0].get('hardware_boot_id', ''),
            same_physical_gpu_verified=bool(rows[0].get('physical_gpu_uuid')),
            same_physical_gpu_protocol_complete=rounds == {0, 1, 2} and bool(rows[0].get('physical_gpu_uuid')),
            gpu=','.join(map(str, sorted({row['gpu'] for row in rows}))),
            median_ms=statistics.median(row['median_ms'] for row in rows),
            p90_ms=statistics.median(row['p90_ms'] for row in rows),
            acceptance='strict' if all(row['acceptance'] == 'strict' for row in rows)
                else 'provisional_container_pid_namespace'))
    return result


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
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
    dnn_cells = aggregate_dnn(dnn_records)
    write_csv(out/'R16_llm_processes.csv', records)
    write_csv(out/'R16_llm_summary.csv', cells)
    write_csv(out/'R16_dnn_processes.csv', dnn_records)
    write_csv(out/'R16_dnn_summary.csv', dnn_cells)
    print(json.dumps({'completed_process_cells': len(records), 'm1_cells': len(cells),
        'three_process_cells': sum(row['protocol_complete'] for row in cells),
        'strict_cells': sum(row['acceptance'] == 'strict' for row in cells),
        'completed_dnn_processes': len(dnn_records),
        'complete_dnn_cells': sum(row['protocol_complete'] for row in dnn_cells),
        'rejected': rejected}, indent=2))


if __name__ == '__main__':
    main()
