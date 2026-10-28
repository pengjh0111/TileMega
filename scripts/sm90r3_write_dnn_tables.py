#!/usr/bin/env python3
"""T10: the native DNN forward's measured latencies and baseline ratios.

Reads each model's P-DNN closeout receipt (or its selection receipt when the
closeout has not run yet), the artifact's plan manifest and the frozen
baselines, and writes docs/results/SM90R3_dnn_final.csv. Baselines are the
frozen historical ones; nothing here re-measures them.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / 'docs/results/R16_final_dnn_baselines.csv'
R1 = ROOT / 'docs/results/SM90R1_final_dnn.csv'
# §2's DNN acceptance targets (prompt): latency ceiling per model, in ms.
TARGETS = dict(resnet18=0.10, bert=0.25, mbv1=0.08, mbv2=0.12, nafnet=0.8)


def fastest_baseline(model):
    best = None
    for row in csv.DictReader(BASELINES.open()):
        if row['model'] != model or row['precision'] != 'bf16':
            continue
        value = float(row['median_ms'])
        if best is None or value < best[0]:
            best = (value, row['system'], row['compile_mode'] or '-')
    return best


def legacy(model):
    for row in csv.DictReader(R1.open()):
        if row['model'] == model:
            return float(row['median_ms']), float(row['p90_ms'])
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells', nargs='+', required=True,
                   help='model=<plan.so>[,<final result.json>] triples')
    p.add_argument('--out', type=Path, default=ROOT / 'docs/results/SM90R3_dnn_final.csv')
    a = p.parse_args()
    rows = []
    for cell in a.cells:
        model, _, paths = cell.partition('=')
        library, _, receipt = paths.partition(',')
        manifest = json.loads(Path(library + '.plan.json').read_text())
        coordinates = manifest['native_forward']
        bodies = {}
        for stage in coordinates['stages']:
            bodies[stage['body']] = bodies.get(stage['body'], 0) + 1
        measured = json.loads(Path(receipt).read_text()) if receipt else {}
        median = measured.get('forward_ms')
        rounds = measured.get('process_medians') or []
        baseline = fastest_baseline(model)
        r1 = legacy(model)
        row = dict(model=model, batch=manifest['runtime_batch'], precision='bf16',
                   stages=len(coordinates['stages']), residency=coordinates['residency'],
                   structure=';'.join(f'{k}:{v}' for k, v in sorted(bodies.items())),
                   split_k=';'.join(sorted({str(s['split_k']) for s in coordinates['stages']})),
                   tile_n=';'.join(sorted({str(s['tile_n']) for s in coordinates['stages']},
                                          key=int)),
                   origin=manifest['native_origin'],
                   artifact=library, protocol='P-DNN' if receipt else 'not measured',
                   fresh_processes=len(rounds), median_ms=median,
                   process_medians_ms=';'.join(f'{v:.6f}' for v in rounds),
                   range_pct=measured.get('range_pct'),
                   legacy_r1_median_ms=r1[0] if r1 else None,
                   ratio_vs_legacy_r1=round(r1[0] / median, 4) if r1 and median else None,
                   baseline_median_ms=baseline[0] if baseline else None,
                   baseline_system=f'{baseline[1]} {baseline[2]}' if baseline else None,
                   ratio_vs_baseline=round(baseline[0] / median, 4) if baseline and median else None,
                   target_ms=TARGETS.get(model),
                   meets_target=(median <= TARGETS[model]) if median and model in TARGETS else None,
                   baseline_scope='frozen historical baseline; not re-measured')
        rows.append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(table=str(a.out), rows=len(rows)), indent=1))
    for row in rows:
        print(f"{row['model']:9} {row['median_ms']} ms  vs legacy {row['ratio_vs_legacy_r1']}x"
              f"  vs baseline {row['ratio_vs_baseline']}x  target {row['target_ms']} ms"
              f"  meets {row['meets_target']}")


if __name__ == '__main__':
    main()
