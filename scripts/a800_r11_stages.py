#!/usr/bin/env python3
"""Measure adjacent-stage overlap from matched production CUDA intervals."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics

from a100_baselines import sha
from a100_results import write_csv
from a800_r9_fused_trace import accepted_trace
from sm90r4_ledger import kind


def active_by_sm(rows):
    by_sm = defaultdict(list)
    for worker, sm, begin, arrive, release, tasks in rows:
        if tasks and arrive > begin:
            by_sm[sm].append((worker, begin, arrive, release))
    return by_sm


def overlap(a, b):
    return sum(any(worker != other and max(begin, start) < min(end, finish)
                   for worker, begin, end, _ in intervals
                   for other, start, finish, _ in b[sm]) for sm, intervals in a.items() if sm in b)


def stage_rows(args):
    plan, report, device = accepted_trace(args)
    if plan['trace_kind'] != 'overlap' or not report.get('reference_pid'):
        raise ValueError('need independent, paired production stage trace processes')
    ledger = args.run/'ledger'
    with (ledger/'runtime_stages.tsv').open() as stream:
        stages = {int(row['stage']): row for row in csv.DictReader(stream, delimiter='\t')}
    moe = any('router' in row['name'] for row in stages.values())
    labels = {index: kind(row, moe) for index, row in stages.items()}
    by_stage = defaultdict(list)
    columns = ('worker', 'smid', 't_begin', 't_tasks_end', 't_release', 'tasks')
    path = ledger/'stage_trace.tsv'
    evidence = sha(path)
    with path.open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            step, stage = int(row['iteration']), int(row['stage'])
            if stage in stages and labels[stage] not in ('embedding', 'other'):
                record = tuple(int(row[field]) for field in columns)
                _, _, begin, arrive, release, _ = record
                if not 0 < begin <= arrive <= release:
                    raise ValueError(f'incomplete CUDA timestamps at {step=}, {stage=}')
                by_stage[step, stage].append(record)
    if not by_stage:
        raise ValueError('production trace has no active stage samples')
    if any(len(rows) != plan['grid'] or {row[0] for row in rows} != set(range(plan['grid']))
           for rows in by_stage.values()):
        raise ValueError('each sampled boundary must cover every persistent CTA')
    active = {key: active_by_sm(rows) for key, rows in by_stage.items()}
    summaries = defaultdict(list)
    steps = {step for step, _ in by_stage}
    for (step, stage), rows in sorted(by_stage.items()):
        label = labels[stage]
        working = active[step, stage]
        if not working:
            raise ValueError(f'no active workers for {label} on iteration {step}')
        spans = [item for group in working.values() for item in group]
        body = max(row[2] for row in spans)-min(row[1] for row in spans)
        total = max(row[4] for row in rows)-min(row[2] for row in rows)
        critical_barrier = max(row[4] for row in rows)-max(row[3] for row in rows)
        fractions = [(release-end)/(release-begin) for worker, begin, end, release in spans
                     if worker % 16 == 0 and release > begin]
        summaries[label].append(dict(body=body, span=total, critical_barrier=critical_barrier,
            fraction=statistics.median(fractions) if fractions else None,
            own=overlap(working, working), previous=overlap(active.get((step, stage-1), {}), working),
            following=overlap(working, active.get((step, stage+1), {})),
            weight=int(stages[stage]['weight_bytes'])))
    for label, rows in summaries.items():
        fractions = [row['fraction'] for row in rows if row['fraction'] is not None]
        # Router and dense projections have fixed logical weights. Fused MoE
        # stage bytes need observed active experts, and attention also reads KV.
        fixed = label in ('qkv', 'o', 'gate_up', 'down', 'router', 'lm_head')
        measurement = dict(model=plan['model'], batch=plan['batch'], stage=label,
            body_span_ns=statistics.median(row['body'] for row in rows),
            stage_span_ns=statistics.median(row['span'] for row in rows),
            critical_barrier_ns=statistics.median(row['critical_barrier'] for row in rows),
            grid_barrier_fraction=statistics.median(fractions) if fractions else '',
            dual_active_sms=max(row['own'] for row in rows),
            overlaps_previous_sms=max(row['previous'] for row in rows),
            overlaps_next_sms=max(row['following'] for row in rows),
            effective_gbps=(sum(row['weight'] for row in rows)/sum(row['span'] for row in rows)) if fixed else '',
            sampled_boundaries=len(rows), sampled_steps=len(steps), trace_perturbation_pct=report['overhead_pct'],
            gpu_uuid=device['nvml_uuid'], trace_sha256=evidence)
        yield measurement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run', 'plan', 'out'):
        parser.add_argument(f'--{flag}', required=True, type=Path)
    args = parser.parse_args()
    write_csv(args.out, list(stage_rows(args)))


if __name__ == '__main__':
    main()
