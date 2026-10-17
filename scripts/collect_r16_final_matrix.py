#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Collect exactly the selected R16 30-cell policy from independent receipts."""
import argparse
import json
from pathlib import Path

from collect_r16_candidates import collect, csv_rows
from compare_r16_results import candidate_rows
from summarize_r16 import write_csv


MODELS = ('qwen3_0_6b', 'qwen3_1_7b', 'llama3_2_1b', 'llama3_2_3b',
          'llama3_1_8b', 'qwen3_30b_a3b')
BATCHES = (1, 2, 4, 8, 16)
EXPECTED = {(model, batch) for model in MODELS for batch in BATCHES}


def selected_policy(report):
    selected = {}
    for row in report['candidates']:
        if not row.get('selected_by_report'):
            continue
        key = (row['model'], row['batch'])
        if key not in EXPECTED:
            raise ValueError('selected workload is outside the final 30-cell matrix')
        if key in selected:
            raise ValueError('multiple selected candidates for the same final cell')
        if row['status'] != 'completed' or row['fresh_processes'] != 3 or not row['protocol_complete']:
            raise ValueError('selected workload lacks three valid final processes')
        selected[key] = row
    rows = [selected[key] for key in sorted(selected)]
    missing = [dict(model=model, batch=batch) for model, batch in sorted(EXPECTED-set(selected))]
    return rows, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--plan', type=Path, action='append', required=True)
    parser.add_argument('--scheduler-root', type=Path, action='append')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = collect(args.root, args.plan, args.scheduler_root)
    selected, missing = selected_policy(result)
    compact = candidate_rows(dict(result, candidates=selected))
    for row in compact:
        # A policy selects different bodies and executors at each coordinate.
        # Per-cell candidate/artifact identities remain present in the CSV.
        row['comparison_arm'] = 'R16-final-selected-policy-v1'
    summary = dict(evidence='verified_independent_receipt_analysis',
        scope='selected 64/1024 full models; random checkpoint; three fresh processes per cell',
        complete=not missing, completed_cells=len(selected), expected_cells=30,
        missing=missing, selected=selected, default_promoted=False,
        acceptance='strict' if selected and all(row['acceptance']=='strict' for row in selected)
            else 'provisional_container_pid_namespace',
        plans=result['plans'], rejections=result['rejections'],
        remaining='end anchors, affected-grid nonregression and baseline comparisons are separate checks')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'selected-final.json').write_text(json.dumps(summary, indent=2)+'\n')
    write_csv(args.out/'R16_selected_native.csv', csv_rows(selected))
    write_csv(args.out/'R16_selected_llm.csv', compact)
    print(json.dumps(dict(completed_cells=len(selected), expected_cells=30,
                         complete=not missing, rejected_records=len(result['rejections']))))
    return int(args.require_complete and bool(missing))


if __name__ == '__main__':
    raise SystemExit(main())
