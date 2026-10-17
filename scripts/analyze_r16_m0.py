#!/usr/bin/env python3
"""Verify fresh-process M0 concurrency results without launching GPU work."""
import argparse
import json
from pathlib import Path
import statistics

from r16_measurement_validation import request_intervals
from summarize_r16 import collect


def overlap(left, right):
    return max(0.0, min(left['inner_end'], right['inner_end'])-max(left['inner_start'], right['inner_start']))


def analyze(root, run_id=None, solo_phase=None, *, rows=None, rejected=None):
    root = Path(root).resolve()
    if rows is None:
        rows, rejected = collect(root)
    rejected = [] if rejected is None else rejected
    groups, errors = {}, []
    selected = []
    for row in rows:
        name = row['job']
        phase = next((phase for phase in ('solo', 'concurrent') if name.startswith(f'm0_{phase}_')), None)
        if phase is None:
            continue
        if phase == 'solo' and solo_phase:
            if not name.startswith(f"{solo_phase}_{row['model']}_"):
                continue
        elif run_id and not name.startswith(f'm0_{phase}_{run_id}_'):
            continue
        value = json.loads((root/row['measurement']).read_text())
        samples = [json.loads(line) for line in Path(value['guard_log']).read_text().splitlines() if line.strip()]
        guard = json.loads((root/row['measurement']).parents[2].joinpath('guard_result.json').read_text())
        if phase == 'concurrent' and guard.get('numa_concurrency_trial') is not True:
            errors.append(f'{name}: no explicit NUMA concurrency trial receipt')
        item = dict(row, phase=phase, value=value, intervals=request_intervals(value, samples))
        selected.append(item)
        groups.setdefault((phase, row['gpu']), []).append(item)
    for item in rejected:
        if item['job'].startswith('m0_') and (not run_id or f'_{run_id}_' in item['job']):
            errors.append(f"{item['job']}: {item['reason']}")
    cards = sorted({row['gpu'] for row in selected})
    complete = len(cards) == 2
    for phase in ('solo', 'concurrent'):
        for card in cards:
            processes = groups.get((phase, card), [])
            if (len(processes) != 3 or {row['round'] for row in processes} != {0, 1, 2} or
                    len({row['value'].get('pid') for row in processes}) != 3):
                complete = False
                errors.append(f'{phase}/gpu{card}: need exactly three distinct fresh processes, rounds 0/1/2')
    identities = []
    for item in selected:
        value = item['value']
        config = dict(value['configuration'])
        config.pop('engine_config', None)
        identities.append(json.dumps({key: value.get(key) for key in
            ('system', 'model_config_sha256', 'prompt_ids_sha256', 'batch', 'max_tokens',
             'versions', 'load_format', 'checkpoint_sha256')} | {'configuration': config}, sort_keys=True))
    identity_consistent = bool(identities) and len(set(identities)) == 1
    if identities and not identity_consistent:
        errors.append('solo/concurrent model configuration, initialization, or versions differ')
    weights = {'evidence': 'unavailable', 'checkpoint_hash_verified': False}
    if selected:
        value = selected[0]['value']
        if value.get('checkpoint_sha256'):
            weights = {'evidence': 'verified_receipt_hash', 'checkpoint_hash_verified': True,
                       'checkpoint_sha256': value['checkpoint_sha256']}
        elif value.get('load_format') == 'dummy' and value['configuration'].get('seed') is not None:
            weights = {'evidence': 'inferred', 'checkpoint_hash_verified': False,
                       'semantics': 'same official backend/version dummy initializer and engine seed; tensor hashes were not captured',
                       'engine_seed': value['configuration']['seed']}
        else:
            errors.append('weight identity cannot be established from the receipts')
    coverage, comparisons = [], []
    if complete:
        for card in cards:
            solo, concurrent = groups[('solo', card)], groups[('concurrent', card)]
            comparison = dict(gpu=card, fresh_processes_solo=3, fresh_processes_concurrent=3)
            for metric in ('e2e_ms', 'tpot_ms'):
                baseline = statistics.median(row[metric] for row in solo)
                candidate = statistics.median(row[metric] for row in concurrent)
                difference = abs(candidate/baseline-1)
                comparison[metric] = dict(solo=baseline, concurrent=candidate,
                                          relative_difference=difference, passed=difference <= .01+1e-12)
                if difference > .01+1e-12:
                    errors.append(f'gpu{card}: {metric} difference exceeds 1%')
            comparisons.append(comparison)
        for card in cards:
            other = next(gpu for gpu in cards if gpu != card)
            other_intervals = [interval for item in groups[('concurrent', other)] for interval in item['intervals']]
            for item in groups[('concurrent', card)]:
                for interval in item['intervals']:
                    intersections = [overlap(interval, candidate) for candidate in other_intervals]
                    amount = max(intersections, default=0)
                    coverage.append(dict(gpu=card, round=item['round'], run=interval['run'],
                        guaranteed_overlap_seconds=amount, request_duration_seconds=interval['duration'],
                        passed=amount > 0))
                    if amount <= 0:
                        errors.append(f"{item['job']}/run{interval['run']}: no guaranteed actual N1024 timing overlap")
        # Solo must also be solo on the other trial card, rather than merely
        # having two independently guarded jobs with a misleading phase name.
        solo_intervals = {card: [interval for item in groups[('solo', card)] for interval in item['intervals']]
                          for card in cards}
        for left in solo_intervals[cards[0]]:
            for right in solo_intervals[cards[1]]:
                if min(left['outer_end'], right['outer_end']) > max(left['outer_start'], right['outer_start']):
                    errors.append('solo trial cards have overlapping request windows')
                    break
    passed = complete and identity_consistent and not errors
    ownership_verified = bool(selected) and all(row['ownership_verified'] for row in selected)
    return dict(evidence='verified_read_only_receipt_analysis', status='passed' if passed else 'failed' if complete else 'incomplete',
        passed=passed, can_enable_numa_concurrency=passed, run_id=run_id, solo_phase=solo_phase, threshold=.01,
        process_count=len(selected), cards=cards, identity_consistent=identity_consistent,
        weight_identity=weights, ownership_verified=ownership_verified,
        acceptance=('strict' if ownership_verified else 'provisional_container_pid_namespace') if selected else 'not_run',
        thermal='all selected receipts passed per-request throttle checks' if selected else 'not_run',
        versions=selected[0]['value']['versions'] if selected else {},
        comparisons=comparisons, interval_coverage=coverage, errors=errors,
        interval_method='guaranteed inner interval [after_timestamp-duration,before_timestamp+duration]; outer telemetry overlap alone cannot pass',
        measurements=[row['measurement'] for row in selected])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run-id')
    parser.add_argument('--solo-phase', help='exact solo phase, e.g. m0_solo, when reusing legacy solo receipts')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root, args.run_id, args.solo_phase)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'passed', 'process_count', 'ownership_verified', 'errors')}))
    raise SystemExit(0 if report['passed'] else 2 if report['status'] == 'incomplete' else 3)


if __name__ == '__main__':
    main()
