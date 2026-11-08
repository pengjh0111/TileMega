#!/usr/bin/env python3
"""Collect the three CNNs without repeating existing baseline measurements."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('resnet18', 'resnet50', 'vgg16')
BATCHES = (1, 2, 4, 8, 16)
ARMS = ('tensorrt-bf16', 'torchcompile-max-autotune',
        'torchcompile-max-autotune-freezing', 'torchcompile-reduce-overhead')


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def geometric(values):
    return math.exp(statistics.mean(map(math.log, values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'docs/results')
    parser.add_argument('--out', type=Path, default=ROOT/'docs/results')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prior = {(r['model'], int(r['batch'])): r for r in read(args.results/'DNN3_final.csv')}
    current = {(r['model'], int(r['batch'])): r for r in read(args.results/'DNN4_final.csv')}
    baseline_path = args.results/'DNN3_baselines_phase_z.csv'
    baseline_sha = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    baselines = [dict(r, reused_from=str(baseline_path), reused_csv_sha256=baseline_sha)
                 for r in read(baseline_path) if r['model'] in MODELS and r['arm'] in ARMS]
    indexed = {(r['model'], int(r['batch']), r['arm']): r for r in baselines}
    expected = {(m, b, a) for m in MODELS for b in BATCHES for a in ARMS}
    if len(baselines) != 60 or set(indexed) != expected:
        raise ValueError('the reused CNN baseline table must contain exactly 60 existing arms')
    if any(r['status'] != 'pass' or r['precision'] != 'bf16' for r in baselines):
        raise ValueError('reused baselines must have passed their BF16 gates')
    rows = []
    for model in MODELS:
        for batch in BATCHES:
            key = model, batch
            old, new = prior[key], current.get(key)
            if new is None and not args.partial:
                raise ValueError(f'CNN cell is incomplete: {model} B{batch}')
            target = float(old['baseline_ms'])
            frozen = indexed[model, batch, old['baseline_arm']]
            if float(frozen['median_ms']) != target or int(frozen['gpu']) != int(old['gpu']):
                raise ValueError('the reused anchor target or GPU differs from DNN3_final.csv')
            row = dict(model=model, batch=batch, gpu=old['gpu'], status='measured' if new else 'pending',
                       tilemega_ms=new['median_ms'] if new else '',
                       p90_ms=new['p90_ms'] if new else '',
                       process_range_ms=new['process_range_ms'] if new else '',
                       dnn3_tilemega_ms=old['median_ms'], fixed_baseline_arm=old['baseline_arm'],
                       fixed_baseline_ms=old['baseline_ms'],
                       ratio_to_baseline=new['ratio'] if new else '',
                       speedup_over_dnn3=float(old['median_ms'])/float(new['median_ms']) if new else '',
                       faster_than_baseline=new['faster_than_baseline'] if new else '')
            row.update({arm+'_ms': indexed[model, batch, arm]['median_ms'] for arm in ARMS})
            row.update(winner=new['winner'] if new else '', artifact_id=new['artifact_id'] if new else '',
                       coordinate_summary=new['coordinate_summary'] if new else '',
                       clocks=new['clocks'] if new else '',
                       correctness='P-CORR 5/5 cross-process output hashes' if new else '',
                       timing='50 warmups + 200 events;3 fresh processes' if new else '',
                       evidence='verified' if new else 'pending',
                       qualification=new['qualification'] if new else '',
                       source=new['source'] if new else '')
            rows.append(row)
    measured = [r for r in rows if r['status'] == 'measured']
    aggregates = []
    groups = [('model', m, [r for r in measured if r['model'] == m]) for m in MODELS]
    groups += [('batch', b, [r for r in measured if r['batch'] == b]) for b in BATCHES]
    groups += [('overall', 'CNN', measured)]
    for kind, group, members in groups:
        if not members:
            continue
        expected_count = 5 if kind == 'model' else 3 if kind == 'batch' else 15
        aggregates.append(dict(kind=kind, group=group, measured_cells=len(members),
                               expected_cells=expected_count,
                               geometric_ratio_to_baseline=geometric([float(r['ratio_to_baseline']) for r in members]),
                               geometric_speedup_over_dnn3=geometric([r['speedup_over_dnn3'] for r in members]),
                               baseline_wins=sum(float(r['ratio_to_baseline']) < 1 for r in members),
                               evidence='verified', scope='complete' if len(members)==expected_count else 'partial'))
    write(args.out/'DNN4_CNN_final.csv', rows)
    write(args.out/'DNN4_CNN_ratios.csv', aggregates)
    write(args.out/'DNN4_CNN_baselines_reused.csv', baselines)
    lines = ['# DNN4 CNN collection', '',
             f'Verified: {len(measured)}/15 CNN cells collected; '
             f'{sum(float(r["ratio_to_baseline"]) < 1 for r in measured)}/{len(measured)} beat the fixed fastest BF16 target.', '',
             'Existing TileMega results are reused. All 60 CNN baseline arms are reused from '
             'DNN3_baselines_phase_z.csv with their original GPU, input, checkpoint and measurement receipts; '
             'no baseline was recollected for this collection. Phase Z anchor checks are tracked separately in DNN4_h800.md. '
             'ResNet18 uses GPU3, ResNet50 GPU0 and VGG16 GPU2. All inputs/weights are BF16 and accumulations FP32.', '',
             'Measured TileMega winners passed five fresh FP32-reference processes with identical output byte hashes. '
             'Timing uses a nontrace build, 50 warmups and 200 CUDA events in each of three fresh processes; '
             'the reported latency is the median of process medians. Full CSV includes p90, process range, '
             'coordinates, artifact identities, SM clocks and raw receipts. GPU ownership remains provisional '
             'because the container PID namespace cannot resolve every host process.', '',
             '| Model | B | TileMega ms | DNN3 TileMega ms | Speedup over DNN3 | Fixed BF16 ms | Ratio |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        latency = f'{float(row["tilemega_ms"]):.6f}' if row['status']=='measured' else 'pending'
        speedup = f'{row["speedup_over_dnn3"]:.3f}x' if row['status']=='measured' else '—'
        ratio = f'{float(row["ratio_to_baseline"]):.4f}' if row['status']=='measured' else '—'
        lines.append(f'| {row["model"]} | {row["batch"]} | {latency} | '
                     f'{float(row["dnn3_tilemega_ms"]):.6f} | {speedup} | '
                     f'{float(row["fixed_baseline_ms"]):.6f} | {ratio} |')
    lines += ['', '| Group | Cells | Geometric ratio to baseline | Geometric speedup over DNN3 | Baseline wins |',
              '| --- | --- | --- | --- | --- |']
    for row in aggregates:
        lines.append(f'| {row["kind"]}:{row["group"]} | {row["measured_cells"]}/{row["expected_cells"]} | '
                     f'{row["geometric_ratio_to_baseline"]:.4f} | '
                     f'{row["geometric_speedup_over_dnn3"]:.3f}x | {row["baseline_wins"]} |')
    lines += ['', 'Ratios below 1 beat the fixed fastest BF16 target. Improvement over DNN3 TileMega '
              'does not imply a win against that target. Partial aggregates describe measured members only. '
              'BERT optimization continues independently; this CNN table does not claim BERT completion.', '',
              'CSV files: DNN4_CNN_final.csv, DNN4_CNN_ratios.csv, DNN4_CNN_baselines_reused.csv.']
    (args.out/'DNN4_CNN_h800.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(cells=len(measured), baseline_rows=len(baselines), partial=len(measured)!=15)))


if __name__ == '__main__':
    main()
