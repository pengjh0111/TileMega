#!/usr/bin/env python3
"""Separate typical CTA spans, maximum local spans, and launch-skew envelopes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


SHAPES = {'gate_C0': 64, 'gate_C3': 32, 'down_C0': 56, 'down_C3': 56}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_trial(rows):
    if not rows or len({r['cta'] for r in rows}) != len(rows):
        raise ValueError('unique CTA spans required')
    for row in rows:
        if not all(isinstance(row[k], int) for k in ('issue', 'first', 'release', 'final')):
            raise ValueError('integer endpoints required')
        if not 0 < row['issue'] <= row['first'] <= row['release'] <= row['final']:
            raise ValueError('reversed CTA endpoints')
    selected = min(rows, key=lambda r: (-(r['final'] - r['issue']), r['cta']))
    latest = min(rows, key=lambda r: (-r['final'], r['cta']))
    span = selected['final'] - selected['issue']
    first = selected['first'] - selected['issue']
    main = selected['release'] - selected['first']
    tail = selected['final'] - selected['release']
    assert first + main + tail == span
    return dict(ctas=len(rows), span_critical_cta=selected['cta'],
                latest_final_cta=latest['cta'], max_local_span_ns=span,
                typical_local_span_ns=statistics.median(r['final'] - r['issue'] for r in rows),
                selected_first_ns=first, selected_main_ns=main, selected_tail_ns=tail,
                absolute_envelope_with_launch_skew_ns=latest['final'] - min(r['issue'] for r in rows),
                issue_skew_ns=max(r['issue'] for r in rows) - min(r['issue'] for r in rows))


def read_trials(raw):
    groups = {}
    with Path(raw).open() as stream:
        for r in csv.DictReader(stream, delimiter='\t'):
            shape, trial, cta = r['shape'], int(r['trial']), int(r['cta'])
            if shape not in SHAPES or not 0 <= trial < 8 or not 0 <= cta < 128:
                raise ValueError('unexpected profile shape/trial/CTA')
            key = shape, trial, cta
            record = groups.setdefault(key, {})
            endpoint = r['role'], int(r['sequence']), r['event']
            if endpoint in record:
                raise ValueError('duplicate endpoint')
            record[endpoint] = int(r['global_ns'])
    expected = {(s, t, c) for s in SHAPES for t in range(8) for c in range(128)}
    if groups.keys() != expected:
        raise ValueError('missing profile shape/trial/CTA')
    trials = {}
    for (shape, trial, cta), values in groups.items():
        def envelope(sequence, event):
            return max(values[str(w), sequence, event] for w in range(4))
        last = SHAPES[shape] - 1
        row = dict(cta=cta, issue=values['producer', -1, 'issue_call_anchor'],
                   first=envelope(0, 'full_ready'), release=envelope(last, 'release'),
                   final=envelope(last, 'finalize_complete'))
        trials.setdefault((shape, trial), []).append(row)
    return trials


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('preserve the previous analysis')
    results = json.loads(args.results.read_text())
    if (len(results['processes']) != 5 or results['performance_eligible'] or
            results['coefficient_fit'] or results['scope']['tasks_per_cta'] != 1):
        raise ValueError('five accepted single-task diagnostic profiles required')
    home = args.results.resolve().parent.parent
    prepared = home / 'queues-prepared.json'
    if sha(prepared) != results['queue_manifest_sha256']:
        raise ValueError('profile queue manifest changed')
    queues = json.loads(prepared.read_text())['queues']
    state = json.loads(args.state.read_text())
    rows, bindings, pids = [], {}, set()
    for process, queue in zip(results['processes'], queues, strict=True):
        fresh = process['fresh']
        job = state[queue['name']]
        if job['status'] != 'done' or process['guard']['code'] or process['metadata']['numerical_failures']:
            raise ValueError('profile process rejected')
        pids.add(process['process']['actual_pid'])
        raw = Path(job['out']) / 'unit/probe/raw.tsv'
        if sha(raw) != process['raw_sha256']:
            raise ValueError('profile raw endpoint file changed')
        bindings[str(raw)] = sha(raw)
        for (shape, trial), ct in sorted(read_trials(raw).items()):
            rows.append(dict(shape=shape, fresh=fresh, trial=trial, **summarize_trial(ct)))
    if len(pids) != 5:
        raise ValueError('five distinct profile processes required')
    metrics = ('max_local_span_ns', 'typical_local_span_ns',
               'absolute_envelope_with_launch_skew_ns', 'issue_skew_ns')
    processes = []
    for shape in SHAPES:
        for fresh in range(5):
            selected = [r for r in rows if r['shape'] == shape and r['fresh'] == fresh]
            if len(selected) != 8:
                raise ValueError('eight trials per fresh process required')
            processes.append(dict(shape=shape, fresh=fresh,
                **{m: statistics.median(r[m] for r in selected) for m in metrics}))
    summary = {s: {m: statistics.median(r[m] for r in processes if r['shape'] == s)
                   for m in metrics} for s in SHAPES}
    deltas = {family: {m: summary[family + '_C3'][m] - summary[family + '_C0'][m]
                       for m in metrics} for family in ('gate', 'down')}
    args.out.mkdir(parents=True)
    write_csv(args.out / 'trial_spans.csv', rows)
    write_csv(args.out / 'process_spans.csv', processes)
    record = dict(summary=summary, C3_minus_C0_ns=deltas, raw_sha256=bindings,
        results_sha256=sha(args.results), script_sha256=sha(__file__),
        quantiles='median of eight trial statistics per process, then median of five processes',
        paired_component_scope='selected max-local-span CTA within each trial; marginal medians are not summed',
        evidence='verified raw-endpoint reduction; max local span excludes issue skew; absolute envelope includes kernel dispatch skew',
        production_stage_critical_claim=False, performance_eligible=False,
        coefficients_changed=False, solver_gate_pass=False)
    (args.out / 'summary.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(summary=summary, C3_minus_C0_ns=deltas)))


if __name__ == '__main__':
    main()
