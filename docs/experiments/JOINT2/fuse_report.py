#!/usr/bin/env python3
"""Rebuild the legal-family fusion envelope from evaluator rows and traces.

Run after fuse_bound.cpp has evaluated each frozen selected CG. The common
0.232 fixed fraction is the prompt's extrapolation, not a measured per-pair
fraction. The additional entire-pair cap makes that assumption inspectable.
"""
import argparse
import csv
import json
from pathlib import Path
import analyze

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "fuse_upper")
    args = parser.parse_args()
    rows = []
    for name, path in json.loads((HERE / 'cells.json').read_text()).items():
        root = REPO / path
        raw = analyze.table(args.out / (name + '_selected.tsv'))
        supported = [r for r in raw if r['status'] == 'SUPPORTED']
        summed = lambda k: sum(float(r[k]) for r in supported)
        fixed = .232 * summed('separate_task_ns')
        traffic = sum(max(0., float(r['separate_task_ns']) -
                          float(r['traffic_floor_ns'])) for r in supported)
        upper = sum(min(float(r['separate_task_ns']),
                        .232 * float(r['separate_task_ns']) +
                        max(0., float(r['separate_task_ns']) -
                            float(r['traffic_floor_ns']))) for r in supported)
        trace = analyze.trace_cell(root, analyze.chosen(root))
        envelope = summed('separate_task_ns')
        rows.append(dict(cell=name, supported=len(supported),
                         eliminated_nodes=summed('eliminated_nodes'),
                         removed_global_bytes=summed('removed_global_bytes'),
                         fixed_upper_ns=fixed, traffic_upper_ns=traffic,
                         optimistic_upper_ns=upper, cp_ns=trace['cp_ns'],
                         queue_lb_ns=trace['queue_lb_ns'], floor_ns=trace['floor_ns'],
                         upper_share=upper / trace['floor_ns'],
                         entire_pair_envelope_cap_ns=envelope,
                         entire_pair_envelope_cap_share=envelope / trace['floor_ns']))
        print(json.dumps(rows[-1]), flush=True)
    with (args.out / 'bounds.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t',
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    maximum = max(r['upper_share'] for r in rows)
    line = (f'FUSE6 enter_r7={int(maximum >= .1)} '
            f'maximum_bound_share={maximum:.6f} cells={len(rows)}')
    (args.out / 'decision.txt').write_text(line + '\n')
    print(line, flush=True)


if __name__ == '__main__':
    main()
