#!/usr/bin/env python3
"""Summarize guarded development pairs, correctness, and stage-only diagnostics."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics as st

from sm90d1_common import read


def output(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--states', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tags', nargs='+', default=['A7', 'A8'])
    p.add_argument('--prefix', default='A')
    a = p.parse_args()
    states = [read(path) for path in a.states]
    checks, diagnostics = {}, {}
    for state in states:
        for row in state.values():
            if row['status'] != 'done':
                continue
            folder = Path(row['out'])
            selected_path = folder / 'job/selected-config.json'
            if not selected_path.exists():
                continue
            selected = read(selected_path)
            if read(folder / 'guard_result.json')['code']:
                raise ValueError('a completed job has a rejected guard')
            config = read(selected['config'])
            command = selected['command']
            binary = command[command.index('--decode') + 1]
            if selected['stage'] == 'correctness':
                measurements = folder / 'job/measurements'
                c1 = measurements / 'c1.json'
                if c1.exists():
                    checks[selected['decode_sha256']] = (read(c1), read(measurements / 'c2.json'))
            elif selected['stage'] in ('trace', 'trace-control'):
                tag = Path(binary).parents[1].name
                if tag == 'reference2':
                    tag = 'reference'
                for past in (folder / 'job/trace').glob('past*'):
                    diagnostics[config['model'], config['batch'], tag] = past
    pairs = defaultdict(list)
    with a.pairs.open() as stream:
        for row in csv.DictReader(stream):
            selected = read(Path(row['candidate_run']) / 'job/selected-config.json')
            config = read(selected['config'])
            command = selected['command']
            binary = Path(command[command.index('--decode') + 1])
            tag = binary.parents[1].name
            if tag in a.tags:
                pairs[config['model'], config['batch'], tag,
                      row['baseline_tag'], row['baseline_sha256']].append(row)
    results = []
    for (model, batch, tag, baseline_tag, baseline_hash), rows in sorted(pairs.items()):
        hashes = {x['candidate_sha256'] for x in rows}
        if len(hashes) != 1:
            raise ValueError('one comparison must use one artifact')
        digest = hashes.pop()
        c1, c2 = checks[digest]
        before = [float(x['baseline_ms']) for x in rows]
        after = [float(x['candidate_ms']) for x in rows]
        change = (st.median(after) / st.median(before) - 1) * 100
        rb = (max(before) - min(before)) / st.median(before) * 100
        ra = (max(after) - min(after)) / st.median(after) * 100
        threshold = max(0.5, rb, ra)
        results.append(dict(model=model, batch=batch, tag=tag, baseline_tag=baseline_tag,
            baseline_sha256=baseline_hash, fresh_processes=len(rows),
            reference_ms=st.median(before), candidate_ms=st.median(after), change_pct=change,
            reference_range_pct=rb, candidate_range_pct=ra, threshold_pct=threshold,
            decision='indistinguishable' if abs(change) <= threshold else 'faster' if change < 0 else 'slower',
            c1_positions_per_sequence=c1['positions'] // batch, c1_total_positions=c1['positions'],
            c1_pass=c1['pass'], c1_max_gap=c1['max_gap'],
            c2_pass=c2['pass'], reference_tokens_equal=all(x['token_equality'] == 'True' for x in rows),
            binary_sha256=digest, ownership_verified=all(x['ownership_verified'] == 'True' for x in rows)))
    output(a.out / (a.prefix+'_development.csv'), results)
    ledger, overhead = [], []
    for (model, batch, tag), folder in sorted(diagnostics.items()):
        if tag not in ['reference-stage', *[x+'-stage' for x in a.tags]]:
            continue
        control = diagnostics.get((model, batch, tag.removesuffix('-stage')))
        if control is None:
            continue
        trace = read(folder / 'trace.json')
        plain = read(control / 'trace.json')
        overhead.append(dict(model=model, batch=batch, tag=tag, past=trace['past'],
            trace_ms=trace['mean_step_ms'], control_ms=plain['mean_step_ms'],
            overhead_ms=trace['mean_step_ms'] - plain['mean_step_ms'], diagnostic_only=True))
        path = folder / 'ledger.json'
        if not path.exists():
            continue
        data = read(path)
        if len(data['steps']) != 8 or any(abs(x['closure_error_fraction']) > 0.03 for x in data['steps']):
            raise ValueError('incomplete stage ledger or closure failure')
        groups = defaultdict(list)
        for row in data['stages']:
            groups[row['kind']].append(row)
        for kind, rows in sorted(groups.items()):
            by_step = defaultdict(list)
            for row in rows:
                by_step[row['iteration']].append(row)
            ledger.append(dict(model=model, batch=batch, tag=tag, past=trace['past'], kind=kind,
                instances_per_step=st.median([len(x) for x in by_step.values()]),
                envelope_ns=st.median([x['envelope_ns'] for x in rows]),
                floor_ns=st.median([x['floor_ns'] for x in rows]),
                excess_ns=st.median([x['envelope_ns'] - x['floor_ns'] for x in rows]),
                class_total_ns=st.median([sum(x['envelope_ns'] for x in v) for v in by_step.values()]),
                closure_max_fraction=max(abs(x['closure_error_fraction']) for x in data['steps']),
                diagnostic_only=True))
    output(a.out / (a.prefix+'_stage_ledger.csv'), ledger)
    output(a.out / (a.prefix+'_trace_overhead.csv'), overhead)
    print(f'{len(results)} development rows; {len(ledger)} stage rows; {len(overhead)} overhead rows')


if __name__ == '__main__':
    main()
