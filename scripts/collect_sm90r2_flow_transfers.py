#!/usr/bin/env python3
"""Collect guarded finite-ring observations; no hardware admission inference."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from sm90d1_common import read, sha, write


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.floor(fraction * len(ordered)))]


def collect(folder, fresh):
    guard = read(folder / 'guard_result.json')
    unit = folder / 'unit'
    command = read(unit / 'physical-device-command.json')
    process = read(unit / 'probe-process.json')
    identity = read(unit / 'physical-device-identity.json')
    metadata = read(unit / 'probe/metadata.json')
    if guard['code'] or command['actual_returncode'] or metadata['numerical_failures']:
        raise ValueError('guard, child or transfer sentinel rejected the process')
    if (process['fresh'] != fresh or metadata['fresh'] != fresh
            or process['ppid'] != identity['wrapper_pid']
            or process['boot_id'] != identity['boot_id']
            or metadata['frequency_drift_pct'] > 2):
        raise ValueError('probe identity/frequency mismatch')
    if (metadata['slots'], metadata['weight_bytes'], metadata['activation_bytes']) != (2, 32768, 16384):
        raise ValueError('not the declared Qwen B16 transfer packet')
    groups = defaultdict(lambda: defaultdict(list))
    trials = defaultdict(list)
    raw = unit / 'probe/raw.csv'
    with raw.open() as stream:
        for row in csv.DictReader(stream):
            if int(row['fresh']) != fresh:
                raise ValueError('mixed process observations')
            key = (row['cache_preparation'], int(row['activation']),
                   int(row['active_ctas']), int(row['operations']))
            wait, issue, packet, first, ready, retired, joined = (
                int(row[k]) for k in ('wait_begin', 'issue_begin', 'packet_end',
                                     'first_warp_ready', 'all_warps_ready',
                                     'all_warps_retired', 'joined'))
            if not (wait <= issue <= packet and issue <= first <= ready <= retired <= joined):
                raise ValueError('timestamp order failed')
            # Integer deltas precede all float/statistical conversions.
            values = dict(empty_wait_observed_ns=issue - wait,
                          source_issue_packet_ns=packet - issue,
                          source_issue_to_first_warp_ready_ns=first - issue,
                          source_issue_to_all_warps_ready_ns=ready - issue,
                          ready_observation_skew_ns=ready - first,
                          readers_retire_envelope_ns=retired - ready)
            for name, value in values.items():
                groups[key][name].append(value)
            trials[(key, int(row['trial']))].append((int(row['cta']),
                int(row['sequence']), issue, ready))
    expected = {(cache, a, active, ops) for cache in ('warm_touch', 'scrubbed_surrogate')
                for a in (0, 1) for active in (1, 32, 64, 128) for ops in (1, 2, 8)}
    if set(groups) != expected:
        raise ValueError('incomplete transfer geometry coverage')
    for (key, _), rows in trials.items():
        _, with_a, active, operations = key
        if {(r[0], r[1]) for r in rows} != {(c, s) for c in range(active) for s in range(operations)} or len(rows) != active * operations:
            raise ValueError('missing/duplicate CTA or sequence')
        envelope = max(r[3] for r in rows) - min(r[2] for r in rows)
        if envelope <= 0:
            raise ValueError('nonpositive population envelope')
        groups[key]['population_issue_to_ready_envelope_ns'].append(envelope)
        groups[key]['logical_payload_effective_bytes_per_ns'].append(
            active * operations * (32768 + with_a * 16384) / envelope)
    for key, values in groups.items():
        if len(values['population_issue_to_ready_envelope_ns']) != 16:
            raise ValueError('expected 16 measured trials after discarded warmup')
    receipts = {str(p.relative_to(folder)): sha(p) for p in (
        folder / 'guard_result.json', unit / 'physical-device-command.json',
        unit / 'physical-device-identity.json', unit / 'probe-process.json',
        unit / 'probe/metadata.json', raw)}
    return groups, dict(fresh=fresh, pid=process['pid'], boot_id=process['boot_id'],
        binary_sha256=process['binary_sha256'], ownership_verified=guard['ownership_verified'],
        attempt=str(folder), receipts=receipts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('preserve prior collection')
    binding = read(args.bindings)
    for path, digest in binding['files'].items():
        if sha(Path(path)) != digest:
            raise ValueError('frozen probe input changed: ' + path)
    state = read(args.state)
    aggregate = defaultdict(list)
    evidence = []
    for fresh in range(5):
        row = state[f'flow-transfer-v2-f{fresh}']
        if row['status'] != 'done':
            raise ValueError('five accepted fresh processes are required')
        groups, receipt = collect(Path(row['out']), fresh)
        evidence.append(receipt)
        for key, values in groups.items():
            for metric, samples in values.items():
                aggregate[key + (metric,)].append((median(samples), quantile(samples, .99)))
    if len({(v['boot_id'], v['pid']) for v in evidence}) != 5:
        raise ValueError('processes are not distinct')
    binary = next(Path(p) for p in binding['files'] if Path(p).name == 'probe')
    if {e['binary_sha256'] for e in evidence} != {sha(binary)}:
        raise ValueError('processes used another probe')
    args.out.mkdir(parents=True)
    with (args.out / 'metrics.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['cache_preparation', 'activation', 'active_ctas', 'operations',
                         'metric', 'processes', 'median_of_process_medians',
                         'median_of_process_p99', 'aggregation'])
        for key, values in sorted(aggregate.items()):
            writer.writerow([*key, len(values), median(v[0] for v in values),
                             median(v[1] for v in values), 'process_statistics_not_pooled'])
    write(args.out / 'receipt.json', dict(evidence='verified diagnostic observations only',
        bindings_sha256=sha(args.bindings), state_sha256=sha(args.state), processes=evidence,
        numerical_passes=5, numerical_processes=5,
        hardware_admission_observed=False, first_byte_observed=False,
        cache_hit_ratio_measured=False, production_binary_equivalent=False,
        physical_capacity_or_latency_fit='not_performed', performance_acceptance=False))


if __name__ == '__main__':
    main()
