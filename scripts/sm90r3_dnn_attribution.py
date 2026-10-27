#!/usr/bin/env python3
"""N-0 attribution (prompt §10): every stage of the five frozen DNN forwards.

Reads each model's diagnostic L2 trace and writes one row per stage: kind,
task count, critical-chain time, the mainloop/epilogue/wait split where the
task profile has it, and M/N/K for GEMM stages. DT-6 then classifies each
stage into the N-1 design item that owns it.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics

N = Path('/root/TileMega/runs/sm90r3/n0')
MODELS = ('resnet18', 'bert', 'mbv1', 'mbv2', 'nafnet')
# DT-6 table (prompt §10 N-0). Thresholds use the stage's own task bytes.
GRID = 132


def table(path, delimiter='\t'):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def integer(row, key, default=0):
    value = row.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def stage_rows(folder, gemms):
    stages = {integer(r, 'stage'): r for r in table(folder / 'runtime_stages.tsv')}
    spans = defaultdict(list)
    for r in table(folder / 'stage_trace.tsv'):
        if integer(r, 't_release'):
            spans[integer(r, 'stage')].append(r)
    profile = defaultdict(list)
    path = folder / 'task_profile.tsv'
    if path.exists():
        for r in table(path):
            profile[integer(r, 'stage')].append(r)
    gemm_index, rows = 0, []
    previous = min((integer(r, 't_begin') for rs in spans.values() for r in rs), default=0)
    for stage in sorted(spans):
        info = stages.get(stage, {})
        group = spans[stage]
        release = max(integer(r, 't_release') for r in group)
        tasks = sum(1 for r in group if integer(r, 'tasks') > 0)
        total = sum(integer(r, 'tasks') for r in group)
        row = dict(stage=stage, kind=info.get('kind_name', ''), name=info.get('name', ''),
                   tasks=total or tasks, workers=tasks,
                   critical_us=round((release - previous) / 1e3, 3),
                   weight_bytes=integer(info, 'weight_bytes'))
        previous = release
        if row['kind'] in ('kGemm', 'kDwPwFused') and gemm_index < len(gemms):
            g = gemms[gemm_index]
            gemm_index += 1
            row.update(m=g.get('tile_m'), n=g.get('tile_n'), k=g.get('tile_k'),
                       split_k=g.get('split_k'), impl=g.get('impl'))
        for key, field in (('mainloop_us', 'mainloop_ns'), ('epilogue_us', 'epilogue_ns'),
                           ('wait_us', 'wait_ns')):
            values = [integer(r, field) for r in profile.get(stage, []) if integer(r, field)]
            if values:
                row[key] = round(statistics.median(values) / 1e3, 3)
        rows.append(row)
    return rows


def classify(row):
    """DT-6: the N-1 design item that owns this stage's time."""
    kind, tasks = row['kind'], row['tasks'] or 0
    if kind == 'kLayerNorm':
        return 'deferred_ln'
    if kind == 'kDepthwiseConv':
        return 'dw_body'
    if kind in ('kPool', 'kGlobalPoolReduce', 'kLayoutConvert'):
        return 'small_body_or_la'
    if kind in ('kGemm', 'kDwPwFused'):
        if tasks < GRID / 2:
            return 'partition_rule'
        bytes_per_task = row['weight_bytes'] / max(tasks, 1)
        bound_us = 10 * (bytes_per_task / 20e3 + 2.0)
        return 'per_task_overhead' if row['critical_us'] / max(tasks / GRID, 1) > bound_us \
            else 'g_body'
    if kind == 'kEncoderAttention':
        return 'enc_att_body'
    if kind == 'kEmbeddingSum':
        return 'emb_ln_body'
    return 'unexpected'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=N / 'SM90R3_dnn_attribution.csv')
    a = p.parse_args()
    plans = json.loads((N / 'plans.json').read_text())
    fields = ['model', 'stage', 'kind', 'name', 'tasks', 'workers', 'critical_us', 'weight_bytes',
              'm', 'n', 'k', 'split_k', 'impl', 'mainloop_us', 'epilogue_us', 'wait_us', 'dt6']
    out, summary = [], {}
    for model in MODELS:
        folder = N / model / 'trace'
        if not (folder / 'runtime_stages.tsv').exists():
            print(model, 'no trace')
            continue
        gemms = json.loads(Path(plans[model]['library'] + '.plan.json').read_text()).get('gemms', [])
        rows = stage_rows(folder, gemms)
        shares = defaultdict(float)
        for row in rows:
            row['model'] = model
            row['dt6'] = classify(row)
            shares[row['dt6']] += row['critical_us']
            out.append({k: row.get(k, '') for k in fields})
        total = sum(shares.values()) or 1
        summary[model] = dict(stages=len(rows), total_us=round(total, 1),
                              shares={k: round(100 * v / total, 1) for k, v in
                                      sorted(shares.items(), key=lambda x: -x[1])})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(out)
    (N / 'dt6.json').write_text(json.dumps(summary, indent=1) + '\n')
    for model, value in summary.items():
        print(f"{model:9s} stages {value['stages']:3d} total {value['total_us']:9.1f} us  {value['shares']}")
    unexpected = {m: v['shares'].get('unexpected', 0) for m, v in summary.items()}
    print('unexpected share', unexpected)


if __name__ == '__main__':
    main()
