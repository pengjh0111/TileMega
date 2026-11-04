#!/usr/bin/env python3
"""Merge isolated, in-program D0, and P-ABL evidence for a fixed-TN sweep."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as st

from sm90d1_common import read, sha
from sm90r5_marks import kind, tsv
from sm90r6_merge_analysis import write


def critical_segments(group, family):
    critical = max(group, key=lambda r: int(r['t_tasks_end']))
    begin, arrive, release = (int(critical[k]) for k in ('t_begin', 't_tasks_end', 't_release'))
    if not 0 < begin <= arrive <= release or int(critical['past']) < 256:
        raise ValueError('invalid steady-state D0 marks')
    result = dict(stage=(release - begin) / 1000, barrier_tail=(release - arrive) / 1000)
    if family == 'expert_gate_up':
        gate_end = int(critical['m2'])
        if not begin <= gate_end <= arrive:
            raise ValueError('critical CTA lacks a valid gate-up/down split')
        result.update(gate_up=(gate_end - begin) / 1000, down=(arrive - gate_end) / 1000)
    else:
        result['qkv'] = (arrive - begin) / 1000
    return result


def d0(directory, model, family):
    accepted = read(directory / 'result.json')
    if not accepted['passed'] or len(accepted['processes']) != 3:
        raise ValueError('I3 requires three accepted D0 processes')
    medians = defaultdict(list); overheads = []; instances = []
    for process in accepted['processes']:
        trial = Path(process['directory']); trace = trial / 'trace'
        run = read(trace / 'run.json')
        if (read(trial / 'guard/guard_result.json')['code'] or
                sha(trace / 'stage_trace.tsv') != process['raw_sha256'] or
                not run['full_token_equivalence'] or
                read(trace / 'trace-tokens.json') != read(trace / 'reference-tokens.json')):
            raise ValueError('changed or unaccepted I3 D0 process')
        types = {int(r['stage']): kind(r, family == 'expert_gate_up', True)
                 for r in tsv(trace / 'runtime_stages.tsv')}
        target = 'expert_fused' if family == 'expert_gate_up' else 'qkv'
        steps = {int(r['iteration']) for r in tsv(trace / 'step_trace.tsv') if int(r['kernel_end'])}
        if len(steps) != 16:
            raise ValueError('I3 D0 needs sixteen retained steady-state iterations')
        groups = defaultdict(list)
        for row in tsv(trace / 'stage_trace.tsv'):
            if (int(row['iteration']) in steps and types[int(row['stage'])] == target
                    and int(row['t_release'])):
                groups[int(row['iteration']), int(row['stage'])].append(row)
        samples = defaultdict(list)
        for group in groups.values():
            # Split the same CTA instance before taking any medians.
            for key, value in critical_segments(group, family).items():
                samples[key].append(value)
        if not groups:
            raise ValueError('missing target D0 stage')
        instances.append(len(groups)); overheads.append(run['overhead_pct'])
        for key, values in samples.items():
            medians[key].append(st.median(values))
    return {k: st.median(v) for k, v in medians.items()}, instances, st.median(overheads)


def merge(directory):
    cell = read(directory / 'cells.json')[0]
    family = 'expert_gate_up' if cell['family'] == 6 else 'qkv'
    timing = read(directory / 'timing.json')
    if (any(a['processes'] != 3 for a in timing['arms'].values()) or
            len({r['gpu'] for r in timing['rows']}) != 1 or
            any(r['gpu'] == 6 for r in timing['rows'])):
        raise ValueError('I3 P-ABL process count or card mismatch')
    check = read(directory / 'isolated-measurements/result.json')
    if not check['passed'] or len(check['processes']) != 3 or not check['same_card']:
        raise ValueError('incomplete isolated TN sweep')
    isolated = defaultdict(dict)
    for line in (directory / 'isolated-measurements/measurements.jsonl').read_text().splitlines():
        row = json.loads(line); tn = row['geometry']['tile_n']; process = row['process_id']
        if (row['guard_code'] or row['development'] or row['input_state'] != 'A0 prompt-prefilled KV'
                or sha(row['raw']) != row['raw_sha256'] or process in isolated[tn]):
            raise ValueError('invalid or duplicate isolated sample')
        isolated[tn][process] = st.median(row['stage_ns']) / 1000
    legal = {v['tile_n']: v for v in read(directory / 'domain.jsonl')['variants'] if v['legal']}
    if set(isolated) != set(legal) or set(timing['arms']) != {'TN' + str(tn) for tn in legal}:
        raise ValueError('incomplete legal TN coverage')
    rows = []
    for tn, variant in legal.items():
        label = 'TN' + str(tn); artifact = read(directory / label / 'artifacts.json')
        if read(artifact['coordinates']) != variant['coordinates']:
            raise ValueError('I3 variant changed from the fixed-coordinate sweep')
        c2 = read(artifact['c2'])
        if (not c2['passed'] or len(c2['processes']) < 5 or
                c2['artifacts']['candidate']['sha256'] != sha(artifact['binary'])):
            raise ValueError('missing I3 full-model correctness')
        if len(isolated[tn]) != 3:
            raise ValueError('isolated TN needs three fresh processes')
        phases, instances, overhead = d0(directory / label / 'd0', cell['model'], family)
        rows.append(dict(model=cell['model'], batch=cell['batch'], family=family, tile_n=tn,
            isolated_us=st.median(isolated[tn].values()), isolated_processes=3,
            in_program_stage_us=phases['stage'], gate_up_us=phases.get('gate_up', ''),
            down_us=phases.get('down', ''), qkv_us=phases.get('qkv', ''),
            barrier_tail_us=phases['barrier_tail'], d0_processes=3,
            d0_instances='|'.join(map(str, instances)), instrumentation_overhead_pct=overhead,
            tpot_ms=timing['arms'][label]['tpot_ms'], range_pct=timing['arms'][label]['range_pct'],
            timing_processes=3, gpu=timing['rows'][0]['gpu'],
            coordinates_sha256=sha(artifact['coordinates']), binary_sha256=sha(artifact['binary']),
            evidence='verified', note='critical-CTA split at m2: gate-up includes local routing; down includes waits/combine; barrier tail separate; medians of per-instance differences; trial concurrency',
            raw=str(directory.resolve()), timing_sha256=sha(directory / 'timing.json')))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    write(a.out, [row for root in a.root for row in merge(root)])
