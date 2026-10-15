#!/usr/bin/env python3
"""Summarize L1 stage envelopes; these are diagnostics, not additive DAG costs."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def category(stage):
    if stage['kind_name'] != 'kGemm':
        return stage['kind_name']
    for pattern, name in (('.qkv.weight', 'qkv'), ('.gate_up.weight', 'gate_up'),
                          ('.o_proj.weight', 'o'), ('.down_proj.weight', 'down'),
                          ('lm_head.weight', 'lm_head')):
        if pattern in stage['name']:
            return 'gemm-'+name
    return 'gemm-other'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    identity = json.loads((args.trace/'trace.json').read_text())
    if not identity['execution_identity']['trace'] or identity['mode'] != 'L1':
        raise ValueError('expected an explicitly instrumented L1 stage trace')
    stages = {int(row['stage']): row for row in table(args.trace/'runtime_stages.tsv')}
    grouped = collections.defaultdict(list)
    for row in table(args.trace/'stage_trace.tsv'):
        grouped[(int(row['iteration']), int(row['stage']))].append(
            {key: int(value) for key, value in row.items()})
    samples = collections.defaultdict(list)
    for (iteration, stage), rows in grouped.items():
        active = [row for row in rows if row['tasks'] > 0]
        start = min(row['t_begin'] for row in rows)
        end = max(row['t_release'] for row in rows)
        task_end = max(row['t_tasks_end'] for row in rows)
        if end < start or any(row['t_tasks_end'] < row['t_begin'] for row in rows):
            raise ValueError('invalid stage clock ordering')
        samples[category(stages[stage])].append(dict(iteration=iteration,
            envelope_ns=end-start, release_tail_ns=max(0, end-task_end),
            weight_bytes=int(stages[stage]['weight_bytes']),
            active_sms=len({row['smid'] for row in active}),
            tasks=sum(row['tasks'] for row in rows)))
    categories = []
    for kind, rows in samples.items():
        iterations = collections.defaultdict(lambda: collections.defaultdict(int))
        for row in rows:
            for field in ('envelope_ns', 'release_tail_ns', 'weight_bytes'):
                iterations[row['iteration']][field] += row[field]
        medians = {field: statistics.median(row[field] for row in iterations.values())
                   for field in ('envelope_ns', 'release_tail_ns', 'weight_bytes')}
        categories.append(dict(kind=kind, **medians,
            median_active_sms_per_stage=statistics.median(row['active_sms'] for row in rows),
            median_tasks_per_stage=statistics.median(row['tasks'] for row in rows),
            logical_weight_gbps=medians['weight_bytes']/medians['envelope_ns']
                if medians['envelope_ns'] else None))
    categories.sort(key=lambda row: row['envelope_ns'], reverse=True)
    report = dict(evidence='verified trace arithmetic; logical bandwidth inferred',
        performance_eligible=False, trace=str(args.trace.resolve()),
        execution_identity=identity['execution_identity'],
        limitations=['stage envelopes can overlap; not an additive critical-path decomposition',
                     'logical weight bytes are not measured DRAM traffic',
                     'zero KV diagnostic is not a generated request'], categories=categories)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(categories))


if __name__ == '__main__':
    main()
