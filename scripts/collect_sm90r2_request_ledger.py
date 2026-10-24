#!/usr/bin/env python3
"""Summarize guarded request-history traces without inventing class overhead."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import median

from sm90d1_common import read, sha, write


def window(state, name):
    row = state[name]
    if row['status'] != 'done':
        raise ValueError('completed history window required: ' + name)
    attempt = Path(row['out'])
    guard, job = read(attempt / 'guard_result.json'), attempt / 'job'
    selected = read(job / 'selected-config.json')
    config = read(selected['config'])
    if guard['code'] or read(job / 'execution.json')['exit_code']:
        raise ValueError('rejected window')
    if selected['config_sha256'] != sha(selected['config']):
        raise ValueError('configuration changed')
    folders = list((job / 'trace').glob('past*'))
    if len(folders) != 1:
        raise ValueError('one past window required')
    folder = folders[0]
    meta = read(folder / 'trace.json')
    if (meta['protocol_version'] != 3 or meta['measured_steps'] != 8
            or meta['history_decode_steps'] != meta['past'] - 64
            or not meta['diagnostic_only'] or meta['performance_eligible']):
        raise ValueError('not the declared request-history protocol')
    command = selected['command']
    binary = Path(command[command.index('--decode') + 1])
    if sha(binary) != meta['trace_binary_sha256'] or sha(binary) != selected['decode_sha256']:
        raise ValueError('window binary changed')
    physical = read(job / 'physical-device-identity.json')
    return dict(attempt=attempt, folder=folder, meta=meta, config=config,
                selected=selected, uuid=physical['cuda_uuid'], guard=guard)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--trace-job', required=True)
    p.add_argument('--control-job', required=True)
    p.add_argument('--body-review', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('preserve previous collection')
    state, review = read(a.state), read(a.body_review)
    trace, control = (window(state, name) for name in (a.trace_job, a.control_job))
    if (review['status'] != 'passed' or review['core_count'] != review['core_equal_bytes']
            or not all(review['checks'].values())):
        raise ValueError('exact function/resource isolation required')
    if trace['uuid'] != control['uuid'] or trace['config'] != control['config']:
        raise ValueError('history/control physical device or input differs')
    for key in ('past', 'measured_steps', 'warmup_steps', 'history_decode_steps',
                'prompt_ids_sha256', 'plain_binary_sha256', 'driver_sha256', 'grid', 'residency'):
        if trace['meta'][key] != control['meta'][key]:
            raise ValueError('history/control mismatch: ' + key)
    if (trace['meta']['matched_window_control'] or not control['meta']['matched_window_control']
            or review['binaries']['plain']['binary_sha256'] != control['meta']['trace_binary_sha256']
            or review['binaries']['trace']['binary_sha256'] != trace['meta']['trace_binary_sha256']):
        raise ValueError('compiled body review belongs to another pair')
    ledger = read(trace['folder'] / 'ledger.json')
    steps = ledger['steps']
    if len(steps) != 8 or any(abs(v['closure_error_fraction']) > .03 for v in steps):
        raise ValueError('ledger does not close')
    a.out.mkdir(parents=True)
    with (a.out / 'stages.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'batch', 'past', 'kind', 'stage_instances',
                         'median_instance_envelope_ns', 'median_instance_floor_ns',
                         'median_step_class_envelope_ns', 'median_step_class_floor_ns',
                         'class_overhead_identified', 'performance_eligible'])
        for kind in sorted({v['kind'] for v in ledger['stages']}):
            rows = [v for v in ledger['stages'] if v['kind'] == kind]
            totals = defaultdict(lambda: [0, 0])
            for row in rows:
                totals[row['iteration']][0] += row['envelope_ns']
                totals[row['iteration']][1] += row['floor_ns']
            writer.writerow([trace['config']['model'], trace['config']['batch'], trace['meta']['past'],
                kind, len(rows), median(v['envelope_ns'] for v in rows), median(v['floor_ns'] for v in rows),
                median(v[0] for v in totals.values()), median(v[1] for v in totals.values()), False, False])
    write(a.out / 'receipt.json', dict(evidence='verified diagnostic observations',
        trace_job=a.trace_job, control_job=a.control_job, physical_uuid=trace['uuid'],
        ownership_verified=trace['guard']['ownership_verified'] and control['guard']['ownership_verified'],
        trace_window_mean_step_ms=trace['meta']['mean_step_ms'],
        control_window_mean_step_ms=control['meta']['mean_step_ms'],
        observed_window_overhead_ms=trace['meta']['mean_step_ms'] - control['meta']['mean_step_ms'],
        class_overhead_identified=False, performance_eligible=False,
        max_closure_error_fraction=max(abs(v['closure_error_fraction']) for v in steps),
        inputs={str(path): sha(path) for path in (a.state, a.body_review,
            trace['folder'] / 'trace.json', control['folder'] / 'trace.json', trace['folder'] / 'ledger.json')}))


if __name__ == '__main__':
    main()
