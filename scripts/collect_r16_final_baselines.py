#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Merge frozen baselines with explicitly declared, independently checked supplements."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from collect_r16_candidates import read_states
from collect_r16_final_matrix import EXPECTED
from r16_measurement_validation import completed_guard, file_sha, validate_llm
from summarize_r16 import aggregate, collect, dnn_device_identity, record_directory, write_csv


SYSTEMS = ('vllm', 'sglang', 'tensorrt-llm', 'torch.compile')
CHECKPOINT = '2ce4ccb3bac783cba132292ceb8a1851b68e4df172a9b32000533829681475b9'
CONFIG = '2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297'
METRICS = ('e2e_ms', 'ttft_ms', 'tpot_ms', 'tokens_per_second')


def methodology_ready(report):
    return (report.get('requires_methodology_ready') is not True or
            report.get('methodology_ready') is True)


def declared_jobs(report):
    cells = report['cells']
    coordinates = {(cell.get('system', 'torch.compile'), cell['batch']) for cell in cells}
    # Cross-card diagnostics can coexist in records but do not constitute a
    # second baseline process for an already declared primary round.
    records = [row for row in report.get('records', [])
               if (row['system'], row['batch']) in coordinates]
    records += [row for cell in cells for row in cell.get('rounds', [])]
    return [row['job'] for row in records]


def supplemental_process(root, name, event, expected_gpu):
    directory = record_directory(root, name, event)
    guard = completed_guard(directory, event)
    paths = list((directory/'measurements').glob('B*/measurements.json'))
    if len(paths) != 1:
        raise ValueError('supplement must contain exactly one complete baseline cell')
    path = paths[0]
    value, _, ownership = validate_llm(path, guard, root)
    if value['system'] not in SYSTEMS or value['gpu_index'] != expected_gpu:
        raise ValueError('supplement system or guarded GPU differs')
    checkpoint = value['checkpoint']
    if (checkpoint['checkpoint_sha256'] != CHECKPOINT or
            checkpoint['config_sha256'] != CONFIG or value['model_config_sha256'] != CONFIG or
            checkpoint['total_bytes'] != 61064245248 or checkpoint['tensor_count'] != 18867):
        raise ValueError('supplement does not use the paired full MoE checkpoint')
    if value['system'] == 'torch.compile' and value['versions']['transformers'] != '5.20.1':
        raise ValueError('upgraded MoE supplement does not use its frozen Transformers version')
    device = dnn_device_identity(value, directory, path)
    if device['physical_gpu_identity_status'] != 'verified':
        raise ValueError('new supplemental measurement lacks physical CUDA/NVML identity')
    command = json.loads((directory/'physical-device-command.json').read_text())
    if command['actual_returncode'] != 0 or str(path.resolve()) not in {
            str(Path(p).resolve()) for p in command['measurements']}:
        raise ValueError('supplement measurement was not produced by its successful command')
    return dict(job=name, model='qwen3_30b_a3b', system=value['system'],
        batch=value['batch'], round=value['round_index'], pid=value['pid'],
        gpu=value['gpu_index'], **device,
        e2e_ms=value['e2e_seconds']*1000, ttft_ms=value['ttft_seconds']*1000,
        tpot_ms=value['tpot_seconds']*1000, tokens_per_second=value['output_tokens_per_second'],
        acceptance=ownership['acceptance'], versions=value['versions'],
        configuration=value['configuration'], source_checkpoint_sha256=CHECKPOINT,
        measurement=str(path.resolve()), measurement_sha256=file_sha(path))


def supplemental_cells(records):
    groups = {}
    for row in records:
        groups.setdefault((row['model'], row['system'], row['batch']), []).append(row)
    output = []
    for key, rows in sorted(groups.items()):
        if len({row['round'] for row in rows}) != len(rows):
            raise ValueError('supplement declares multiple processes for the same round')
        if {row['round'] for row in rows} != {0, 1, 2}:
            continue
        if len({row['pid'] for row in rows}) != 3:
            raise ValueError('supplement lacks three independent processes')
        for field in ('gpu', 'physical_gpu_uuid', 'hardware_boot_id', 'versions',
                      'source_checkpoint_sha256'):
            if len({json.dumps(row[field], sort_keys=True) for row in rows}) != 1:
                raise ValueError('supplement processes disagree on '+field)
        output.append(dict(model=key[0], system=key[1], batch=key[2],
            fresh_processes=3, protocol_complete=True,
            **{field: statistics.median(row[field] for row in rows) for field in METRICS},
            **{field: rows[0][field] for field in ('physical_gpu_uuid',
                'physical_gpu_identity_status', 'hardware_boot_id', 'versions',
                'source_checkpoint_sha256')},
            gpu=str(rows[0]['gpu']), same_physical_gpu_protocol_complete=True,
            acceptance='strict' if all(row['acceptance']=='strict' for row in rows)
                else 'provisional_container_pid_namespace',
            checkpoint_identity_shared=True, native_artifact_ids='',
            measurements=[row['measurement'] for row in rows],
            source='explicit paired full-MoE supplement'))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--scheduler-root', type=Path, action='append', required=True)
    parser.add_argument('--supplement-report', type=Path, action='append', default=[],
                        help='report declaring records[].job or cells[].rounds[].job')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    original, original_rejections = collect(root)
    original = [row for row in original if row['system'] in SYSTEMS]
    cells = aggregate(original)
    selected = {}
    for row in cells:
        if not row['protocol_complete']:
            continue
        key = (row['model'], row['system'], row['batch'])
        if key in selected:
            raise ValueError('ambiguous original baseline identity groups')
        processes = [item for item in original if item['phase']=='m1' and
                     (item['model'],item['system'],item['batch'])==key]
        if len(processes) != 3:
            raise ValueError('original baseline provenance is ambiguous')
        measurements = [str(root/item['measurement']) for item in processes]
        versions = [json.loads(Path(path).read_text())['versions'] for path in measurements]
        if len({json.dumps(version,sort_keys=True) for version in versions}) != 1:
            raise ValueError('original baseline versions differ across fresh processes')
        row = dict(row, versions=versions[0], measurements=measurements,
                   source='original frozen baseline')
        selected[key] = row
    state, cards = read_states(root, args.scheduler_root)
    supplemental, rejected, names, checks = [], [], set(), []
    for path in args.supplement_report:
        snapshot = path.read_bytes()
        report = json.loads(snapshot)
        checks.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(snapshot).hexdigest(),
            required=report.get('requires_methodology_ready') is True,
            passed=methodology_ready(report)))
        for name in declared_jobs(report):
            if name in names:
                raise ValueError('duplicate explicit supplemental job')
            names.add(name)
            try:
                supplemental.append(supplemental_process(root, name, state[name], cards[name]))
            except (OSError, ValueError, KeyError, TypeError) as error:
                rejected.append(dict(job=name, reason=str(error)))
    for row in supplemental_cells(supplemental):
        selected[(row['model'], row['system'], row['batch'])] = row
    expected = {(model, system, batch) for model, batch in EXPECTED for system in SYSTEMS}
    if set(selected)-expected:
        raise ValueError('baseline cell outside the final 120-cell matrix')
    missing = [dict(model=m, system=s, batch=b) for m,s,b in sorted(expected-set(selected))]
    rows = [selected[key] for key in sorted(selected)]
    for check in checks:
        if file_sha(check['path']) != check['sha256']:
            check.update(passed=False, reason='supplement report changed during independent validation')
    ready = not missing and all(check['passed'] for check in checks)
    result = dict(complete=ready, completed_cells=len(rows), expected_cells=120,
        missing=missing, cells=rows, supplemental_processes=supplemental,
        methodology_checks=checks,
        supplemental_rejections=rejected, original_rejections=original_rejections,
        default_promoted=False, scope='frozen original cells, replaced only by valid explicitly paired MoE supplements')
    args.out.mkdir(parents=True, exist_ok=True)
    temporary = args.out/'final-baselines.pending'
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(args.out/'final-baselines.json')
    # Structured provenance stays JSON; CSV remains directly usable by comparison tools.
    fields = ('model','system','batch','fresh_processes','protocol_complete','gpu',
              'physical_gpu_uuid','physical_gpu_identity_status','hardware_boot_id',
              'e2e_ms','ttft_ms','tpot_ms','tokens_per_second','acceptance',
              'source_checkpoint_sha256','checkpoint_identity_shared','native_artifact_ids','source')
    write_csv(args.out/'R16_final_baselines.csv', [{key:row[key] for key in fields} for row in rows])
    print(json.dumps(dict(completed_cells=len(rows), expected_cells=120,
        complete=ready, supplemental_processes=len(supplemental),
        supplemental_rejections=len(rejected))))
    return int(args.require_complete and not ready)


if __name__ == '__main__':
    raise SystemExit(main())
