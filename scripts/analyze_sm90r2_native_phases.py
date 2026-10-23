#!/usr/bin/env python3
"""Summarize native phase probes; never treat instrumented times as TPOT."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics

from sm90d1_common import write
from analyze_sm90d1_ledger import category


def analyze(folder, first_iteration=None):
    def rows(name):
        with (folder / name).open() as stream:
            return list(csv.DictReader(stream, delimiter='\t'))
    clocks = rows('native_clock.tsv')
    frequency = statistics.median(int(r['cycles']) / int(r['global_ns']) for r in clocks)
    if frequency <= 0:
        raise ValueError('missing local clock calibration')
    stages = {int(r['stage']): r for r in rows('runtime_stages.tsv')}
    observations = rows('native_phases.tsv')
    if first_iteration is None:
        import json
        metadata = folder / 'trace.json'
        keep = json.loads(metadata.read_text())['retain_latest_iterations'] if metadata.exists() else None
        first_iteration = max(0,max(int(r['iteration']) for r in observations)-keep+1) if keep else 0
    groups = defaultdict(list)
    for raw in observations:
        r = {k: int(v) for k, v in raw.items()}
        if r['iteration'] < first_iteration:
            continue
        if r['clock_end'] < r['clock_begin'] or r['global_end'] < r['global_begin']:
            raise ValueError('invalid phase timestamps')
        mma = [r[f'first_mma_{i}'] for i in range(4) if r[f'first_mma_{i}']]
        releases = [r[f'previous_release_{i}'] for i in range(4)]
        start = min(mma) - min(releases) if mma and r['task']==r['worker'] else None
        if start is not None and start < 0:
            raise ValueError('first MMA precedes this CTA release')
        values = dict(query_ns=max(r[f'query_{i}'] for i in range(4)) / frequency,
            new_row_ns=r['new_row_cycles'] / frequency, wait_ns=r['wait_cycles'] / frequency,
            compute_ns=r['compute_cycles'] / frequency, partial_ns=r['partial_cycles'] / frequency,
            publish_ns=r['publish_cycles'] / frequency, merge_ns=r['merge_cycles'] / frequency,
            task_ns=(r['clock_end'] - r['clock_begin']) / frequency,
            first_mma_from_release_ns=start / frequency if start is not None else None)
        key = category(stages[r['stage']])
        groups[key].append(values)
    result = []
    for kind, values in sorted(groups.items()):
        row = dict(kind=kind, tasks=len(values), merge_tasks=sum(v['merge_ns'] > 0 for v in values),
                   new_row_tasks=sum(v['new_row_ns'] > 0 for v in values))
        for field in values[0]:
            selected = [v[field] for v in values if v[field] is not None]
            row[field] = statistics.median(selected) if selected else None
            if field in ('merge_ns', 'new_row_ns'):
                active = [v for v in selected if v > 0]
                row[field + '_active_median'] = statistics.median(active) if active else None
        result.append(row)
    return dict(cycles_per_ns=frequency, classes=result, diagnostic_only=True,
        clock_scope='same CTA/SM intervals; integer subtraction before conversion',
        overlap='query/new-row warps overlap; phase medians do not form an additive envelope',
        instrumentation_overhead='requires a matched uninstrumented control; not inferred here')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--first-iteration', type=int)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    write(a.out, analyze(a.trace, a.first_iteration))


if __name__ == '__main__':
    main()
