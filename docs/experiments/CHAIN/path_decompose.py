#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Split each realized critical path's task work by the edge that carried it.

`path_report.tsv` reports the path's task work as one number, which cannot say
whether a candidate is slow because its path threads a heavier chain of real
dependencies or because it waits behind tasks merely queued ahead of it.  The
distinction is the whole of why chaining loses at mha4 s128 and it is invisible
in the aggregate: there chain and rotate carry almost the same task work on DAG
edges (378535 vs 360293 ns) and differ by 66318 ns on queue edges alone.

`block_ns` is deliberately not summed here.  It is `start - free_at[w]`
(ExecutionSimulator.cpp:208), the idle gap on one *worker* before a task, so it
neither telescopes along a path nor stays inside the span -- at gqa2 s128 rotate
it totals 394397 ns inside a 215127 ns span -- and its zero on every `q` and `s`
step is a tautology of the definition, not a measurement.
"""
import csv
from pathlib import Path
import sys

# `edge_to_next` links a step to the next row, and the dump is written
# sink-first, so the edge named on a row is the one that reaches it from its
# predecessor: `h` cross-worker task edge, `q` queue edge, `s` same-worker task
# edge, `-` the source.
KINDS = ('h', 'q', 's', '-')


def decompose(path):
    name = path.name[:-len('_path.tsv')]
    model, rest = name.split('_', 1)
    seq, candidate = rest.split('_', 1)
    row = {'model': model, 'seq': seq.lstrip('s'), 'candidate': candidate,
           'steps': 0}
    counts = {k: 0 for k in KINDS}
    task = {k: 0.0 for k in KINDS}
    with path.open() as stream:
        for entry in csv.DictReader(stream, delimiter='\t'):
            kind = entry['edge_to_next']
            if kind not in counts:
                continue
            row['steps'] += 1
            counts[kind] += 1
            task[kind] += float(entry['task_ns'])
    total = sum(task.values())
    for k in KINDS:
        row['n_' + (k if k != '-' else 'src')] = counts[k]
        row['task_' + (k if k != '-' else 'src') + '_ns'] = '%.1f' % task[k]
    row['task_total_ns'] = '%.1f' % total
    # The share the path spends waiting on work it does not depend on.
    row['queue_share'] = '%.4f' % (task['q'] / total if total else 0.0)
    return row


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent / 'raw/path'
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else root.parent / 'path_decompose.tsv'
    rows = [decompose(p) for p in sorted(root.glob('*_path.tsv'))]
    order = {'gqa2': 0, 'mha4': 1, 'real': 2}
    rows.sort(key=lambda r: (order.get(r['model'], 9), int(r['seq']), r['candidate']))
    fields = ['model', 'seq', 'candidate', 'steps', 'n_h', 'n_q', 'n_s', 'n_src',
              'task_h_ns', 'task_q_ns', 'task_s_ns', 'task_src_ns',
              'task_total_ns', 'queue_share']
    with out.open('w') as stream:
        writer = csv.DictWriter(stream, fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print('wrote %s (%d rows)' % (out, len(rows)))


if __name__ == '__main__':
    main()
