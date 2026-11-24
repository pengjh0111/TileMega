#!/usr/bin/env python3
"""Rebuild the A800 native serving tables from guarded calibration receipts."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from a100_results import write_csv
from collect_a800_attention_fresh import collect as fresh, collect_compact
from collect_a800_dense_service import collect as dense, read, sha
from collect_a800_fused_service import append as append_fused
from collect_a800_registry import CONSTANTS, merge, service_profiles
from tilemega.bench.scheduler import write_json


def accepted(profile):
    return sum(len(profile.get(key, ())) for key in ('measurements', 'epilogue_measurements',
            'additional_observations')) + ('measurement' in profile)


def rebuild(reference, observations, out):
    base_sha = sha(reference)
    current = read(reference)
    out.mkdir(parents=True, exist_ok=True)
    staging = out/'service-supplement.json'
    rows = []
    for observation in observations:
        table = observation['table']
        run, binary = Path(observation['run']), Path(observation['binary'])
        sample = dict(table=table, job=observation['job'], binary_sha256='', gpu_uuid='',
                      profiles=0, measurements=0, max_perturbation_pct='', status='rejected', reason='')
        prior = current[CONSTANTS].get(table)
        prior_count = len(service_profiles(prior)) if prior is not None else 0
        reported = []
        try:
            identity = read(binary.with_suffix('.identity.json'))
            sample['binary_sha256'] = identity['binary_sha256']
            if not (run/'frozen.json').is_file():
                raise ValueError('allocation produced no frozen guard receipt')
            sample['gpu_uuid'] = read(run/'job/physical-device-identity.json')['nvml_uuid']
            for case in identity['cases']:
                if Path(case).name != case:
                    raise ValueError('measurement case is not a local path')
                files = run/f'{case}.json'
                if files.is_file():
                    reported.append(read(files))
            if table in ('dense_gemm_service', 'dense_gate_service'):
                write_json(staging, current)
                updated = dense(run, staging, binary, reference, .05, table == 'dense_gate_service')
            elif table == 'fused_expert_service':
                updated = deepcopy(current)
                anchor = observation.get('anchor')
                append_fused(updated, base_sha, run, binary, Path(anchor) if anchor else None)
            elif table in ('attention_fresh_service', 'compact_attention_service'):
                collector = fresh if table == 'attention_fresh_service' else collect_compact
                if prior is not None:
                    raise ValueError('retain the previous matching attention measurement')
                updated = deepcopy(current)
                updated[CONSTANTS][table] = collector(reference, run, binary)[CONSTANTS][table]
            else:
                raise ValueError(f'unknown service table {table}')
            new_profiles = service_profiles(updated[CONSTANTS][table])[prior_count:]
            notes = [f'{name}: {reason}' for profile in new_profiles
                     for name, reason in profile.get('rejected', {}).items()]
            notes.extend(f'compact d{item["dim"]}-q{item["queries"]}-norm{item["norm"]}: '
                         f'{item["dynamic_shared"]} bytes, residency {item["residency"]}'
                         for item in reported if table == 'compact_attention_service'
                         and item['status'] == 'resource_rejected')
            count = sum(map(accepted, new_profiles))
            if not count:
                raise ValueError('; '.join(notes) or 'no new valid service measurements')
            current = updated
            sample.update(status='verified', profiles=len(new_profiles), measurements=count, reason='; '.join(notes))
        except (ValueError, KeyError, OSError) as error:
            sample['reason'] = str(error)
        perturbations = []
        for item in reported:
            if item.get('traced_span_ns') and item.get('untraced_span_ns'):
                perturbations.append(abs(item['traced_span_ns']/item['untraced_span_ns']-1))
            elif item.get('plain_kernel_ns'):
                for role in ('producer', 'consumer'):
                    dispatch = item[f'{role}_off_kernel_ns']
                    perturbations.extend((abs(dispatch/item['plain_kernel_ns']-1),
                                          abs(item[f'{role}_kernel_ns']/dispatch-1)))
        if perturbations:
            sample['max_perturbation_pct'] = 100*max(perturbations)
        rows.append(sample)
    write_json(staging, current)
    write_json(out/'registry.json', merge([staging]))
    write_csv(out/'A800_R11_calibration.csv', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('reference', 'ledger', 'out'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.reference, json.loads(args.ledger.read_text())['observations'], args.out)


if __name__ == '__main__':
    main()
