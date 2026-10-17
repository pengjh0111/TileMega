#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Select the measured R16 B1 DNN policy after all declared candidates settle."""
import argparse
import json
from pathlib import Path
import statistics

from analyze_r16_dnn_pair import comparison, evidence, matching_devices
from compare_r16_results import compare, geomeans
from r16_measurement_validation import completed_guard, file_sha
from summarize_r16 import aggregate_dnn, collect_dnn, write_csv


PAIRS = {
    'resnet18': 'uuid-epoch-gpu7-v3/resnet18-ab-evidence.json',
    'nafnet': 'uuid-epoch-gpu7-v3/nafnet-ab-evidence.json',
    'mbv1': 'p2-source19-gpu7-v1/mbv1-ab-evidence.json',
    'mbv2': 'p2-source19-native-shape-v1/mbv2-ab-evidence.json',
    'bert': 'p2-source19-native-shape-v1/bert-ab-evidence.json',
}
TERMINAL = {'done', 'failed', 'skipped', 'cancelled', 'not_run'}


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def settled(jobs, states):
    return all(states.get(name, {}).get('status') in TERMINAL for name in jobs)


def fixed_evidence(candidate, proof, state):
    attempt = Path(state['out'])
    completed_guard(attempt, state)
    path = attempt/'fixed-bf16-gate.json'
    fixed = read(path)
    config = read(candidate)
    device = read(attempt/'physical-device-identity.json')
    command = read(attempt/'physical-device-command.json')
    if (fixed.get('passed') is not True or fixed['gate'].get('passed') is not True or
            fixed['candidate_sha256'] != file_sha(candidate) or
            fixed['binary_sha256'] != file_sha(proof['library']) or
            any(fixed[a] != proof[b] for a, b in (
                ('artifact_id', 'artifact_id'), ('checkpoint_sha256', 'checkpoint_sha256'),
                ('input_sha256', 'input_sha256')))):
        raise ValueError('fixed BF16 gate does not certify the selected binary and inputs')
    if (fixed['gate']['thresholds'] != dict(agreement_margin=.005,
            bf16_error_factor=1.5, psnr_margin_db=.1) or
            fixed['gate_source_sha256'] != file_sha(Path(__file__).resolve().parents[1]/
                'python/tilemega/dnn/gates.py')):
        raise ValueError('fixed BF16 criteria changed')
    if (device['probe_exit_code'] != 0 or command['actual_returncode'] != 0 or
            device['cuda_uuid'] != device['nvml_uuid'].removeprefix('GPU-') or
            device['gpu_index'] != config['gpu']):
        raise ValueError('fixed BF16 physical device attribution differs')
    return dict(path=str(path), sha256=file_sha(path), candidate=str(candidate),
                candidate_sha256=file_sha(candidate), scope=fixed['scope'],
                physical_device=device, metrics=fixed['metrics'])


def choose(options):
    """Keep the incumbent unless a certified alternative improves the E2E gate."""
    selected = options[0]
    decisions = []
    for option in options[1:]:
        for key in ('checkpoint_sha256', 'archive_sha256', 'input_sha256'):
            if selected['proof'][key] != option['proof'][key]:
                raise ValueError('DNN comparison changed ' + key)
        if selected['device'] != option['device']:
            raise ValueError('DNN comparison changed physical card or host epoch')
        result = comparison(selected['rows'], option['rows'])
        decisions.append(dict(reference=selected['proof']['candidate_id'],
            candidate=option['proof']['candidate_id'], **result))
        if result['eligible_in_measured_grid']:
            selected = option
    return selected, decisions


def baseline_policy(groups, cells):
    expected = {(model, system, precision, mode) for model in PAIRS
        for system, precision, mode in [('tensorrt','bf16',''), ('tensorrt','fp16',''),
            ('torchcompile','bf16','max-autotune'), ('torchcompile','bf16','reduce-overhead')]}
    chosen, excluded = [], []
    for key in sorted(expected):
        cell = next(cell for cell in cells if cell['model'] == key[0])
        proof = cell['selected']
        candidates = []
        for row in groups:
            if (row['model'], row['system'], row['precision'], row['compile_mode']) != key:
                continue
            if (not row['protocol_complete'] or row['fresh_processes'] != 3 or
                    row['exported_program_sha256'] != proof['archive_sha256'] or
                    row['source_checkpoint_sha256'] not in ('', proof['checkpoint_sha256'])):
                excluded.append(dict(coordinate=key, reason='incomplete or different exported program/checkpoint', group=row))
                continue
            candidates.append(row)
        def rank(row):
            return (row['physical_gpu_uuid'] == cell['physical_device']['uuid'],
                    bool(row['physical_gpu_uuid']),
                    bool(row['hardware_boot_id']) and row['hardware_boot_id'] == cell['physical_device']['boot_id'])
        candidates.sort(key=rank, reverse=True)
        if not candidates or (len(candidates) > 1 and rank(candidates[0]) == rank(candidates[1])):
            raise ValueError('external DNN baseline is missing or ambiguous: '+str(key))
        chosen.append(dict(candidates[0],
            selection_reason='prefer accepted current physical card/epoch; never select by speed',
            checkpoint_scope='identical exported program; '+('direct checkpoint hash verified'
                if candidates[0]['source_checkpoint_sha256'] else 'historical direct checkpoint hash absent')))
        excluded.extend(dict(coordinate=key, reason='superseded by stronger physical device attribution', group=row)
                        for row in candidates[1:])
    return chosen, excluded


def collect(root):
    base = root/'runs/r16/dnn'
    records, rejected = collect_dnn(root)
    old = read(root/'runs/r16/scheduler/gpu7/state.json')
    clean = read(root/'runs/r16/scheduler-rf2-final-v1/gpu2/state.json')
    plan_path = base/'final-combinations-v1/functional-plan.json'
    plan = read(plan_path)
    cells, compact = [], []
    for model, relative in PAIRS.items():
        pair_path = base/relative
        pair = read(pair_path)
        options, exclusions, pending = [], [], []
        for arm in ('reference', 'candidate'):
            path = Path(pair[arm]['candidate_path'])
            rows, proof = evidence(path, records)
            options.append(dict(rows=rows, proof=proof, device=matching_devices(root, rows)))
        extra = [case for case in plan['cases'] if case['model'] == model]
        for case in extra:
            jobs = [job['name'] for job in case['timing_jobs']]
            if not settled(jobs, old):
                pending.extend(name for name in jobs
                    if old.get(name, {}).get('status') not in TERMINAL)
                continue
            try:
                rows, proof = evidence(Path(case['candidate']), records)
                bridge_path = Path(case['candidate']).parent/'cross-scheduler-bridge.json'
                bridge = read(bridge_path)
                if (not bridge['passed'] or bridge['timing_candidate_sha256'] != file_sha(case['candidate']) or
                        bridge['functional_candidate_sha256'] != file_sha(case['functional_candidate']) or
                        bridge['gate_5fresh_sha256'] != proof['gate_sha256'] or
                        bridge['artifact_id'] != proof['artifact_id']):
                    raise ValueError('functional/timing scheduler bridge changed')
                fixed = fixed_evidence(Path(case['functional_candidate']), proof, clean[case['fixed_job']])
                options.append(dict(rows=rows, proof=proof, device=matching_devices(root, rows),
                    fixed_bf16=fixed, bridge=dict(path=str(bridge_path), sha256=file_sha(bridge_path))))
            except (ValueError, KeyError, OSError, TypeError) as error:
                exclusions.append(dict(candidate=case['candidate'], status='rejected',
                    reason=str(error), jobs={name:old.get(name, {}) for name in jobs}))
        selected, decisions = choose(options)
        if 'fixed_bf16' not in selected:
            fixed_candidate = base/f'final-fixed-bf16-v1/{model}-candidate.json'
            selected['fixed_bf16'] = fixed_evidence(fixed_candidate, selected['proof'],
                clean[f'dnn_final_fixed_bf16_v1_{model}_gpu2'])
        cell = dict(model=model, batch=1, precision='bf16',
            status='pending_candidates' if pending else 'complete', pending_jobs=pending,
            selected=selected['proof'], fixed_bf16=selected['fixed_bf16'],
            physical_device=selected['device'], decisions=decisions, rejected=exclusions,
            median_ms=statistics.median(row['median_ms'] for row in selected['rows']),
            p90_ms=statistics.median(row['p90_ms'] for row in selected['rows']))
        if 'bridge' in selected:
            cell['functional_timing_bridge'] = selected['bridge']
        cells.append(cell)
        compact.append(dict(model=model, system='TileMega', batch=1, precision='bf16', status=cell['status'],
            candidate_id=selected['proof']['candidate_id'], median_ms=cell['median_ms'],
            p90_ms=cell['p90_ms'], artifact_id=selected['proof']['artifact_id'],
            native_artifact_ids=selected['proof']['artifact_id'],
            comparison_arm='R16-final-selected-DNN-policy-v1',
            exported_program_sha256=selected['proof']['archive_sha256'],
            source_checkpoint_sha256=selected['proof']['checkpoint_sha256'],
            fresh_processes=3, protocol_complete=True, gpu=selected['rows'][0]['gpu'],
            physical_gpu_identity_status='verified', hardware_boot_id=selected['device']['boot_id'],
            acceptance='provisional_container_pid_namespace',
            physical_gpu_uuid=selected['device']['uuid'],
            coordinates=json.dumps(selected['proof']['selected_coordinates'], sort_keys=True),
            fixed_bf16_sha256=selected['fixed_bf16']['sha256']))
    external, external_excluded = baseline_policy(aggregate_dnn([row for row in records
        if not row['system'].lower().startswith(('tilemega', 'tm-'))]), cells)
    completed = sum(cell['status'] == 'complete' for cell in cells)
    return dict(evidence='verified_independent_receipt_analysis', complete=completed == 5,
        completed_models=completed, expected_models=5, cells=cells,
        scope='B1 fixed-input random checkpoints; unchanged BF16 criteria; not dataset accuracy',
        default_promoted=False, acceptance='provisional_container_pid_namespace',
        combination_plan=dict(path=str(plan_path), sha256=file_sha(plan_path)),
        external_baseline_cells=20, external_baseline_complete=True, external_baselines=external,
        excluded_external_groups=external_excluded,
        unrelated_rejected_measurements=rejected), compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result, rows = collect(args.root.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    atomic(args.out/'selection.json', result)
    write_csv(args.out/'R16_final_dnn.csv', rows)
    write_csv(args.out/'R16_final_dnn_baselines.csv', result['external_baselines'])
    ratios = compare(rows+result['external_baselines'], 'dnn')
    write_csv(args.out/'R16_final_dnn_ratios.csv', ratios)
    write_csv(args.out/'R16_final_dnn_geomeans.csv', geomeans(ratios, 'dnn'))
    print(json.dumps({key:result[key] for key in ('complete', 'completed_models', 'expected_models')}))
    return int(args.require_complete and not result['complete'])


if __name__ == '__main__':
    raise SystemExit(main())
