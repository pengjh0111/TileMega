#!/usr/bin/env python3
"""Assemble Phase Z tables only from complete, anchored twenty-cell results."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'scripts'))
from tilemega.build.identity import sha, verify
from dnn1_baselines import measurement_clocks
from dnn3_anchor import decision
from dnn3_select import ORDER, select

MODELS = ('resnet18', 'resnet50', 'vgg16', 'bert')
BATCHES = (1, 2, 4, 8, 16)
ARMS = {'tensorrt-bf16', 'torchcompile-max-autotune',
        'torchcompile-max-autotune-freezing', 'torchcompile-reduce-overhead'}


def geometric(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('geometric mean requires finite positive ratios')
    return math.exp(statistics.mean(math.log(v) for v in values))


def coverage(cells):
    keys = [(c['model'], c['batch']) for c in cells]
    if len(keys) != 20 or set(keys) != {(m, b) for m in MODELS for b in BATCHES}:
        raise ValueError('require exactly twenty unique Phase Z cells')
    if any(c.get('status') != 'passed' or c.get('protocol') !=
           'P-SEL; winner P-CORR; P-DNN' for c in cells):
        raise ValueError('all cells must pass Phase Z selection/correctness/timing')


def baseline(cell, anchor, frozen, resampled):
    if (anchor.get('status') != 'passed' or anchor.get('protocol') != 'P-ANCHOR' or
            anchor.get('model') != cell['model'] or anchor.get('gpu') != cell['gpu'] or
            sorted(r['batch'] for r in anchor['anchors']) != [1, 16]):
        raise ValueError('cell lacks passed B1/B16 anchors on its own card')
    if anchor['action'] == 'retain_frozen':
        if any(r['action'] != 'retain_frozen' or
               decision(r['frozen_ms'], r['median_ms'])['action'] != 'retain_frozen'
               for r in anchor['anchors']):
            raise ValueError('anchor retention conflicts with an endpoint decision')
        rows = frozen
    elif anchor['action'] == 'resample_trt_tc_all_batches':
        rows = [r for r in resampled if r['model'] == cell['model']
                and int(r['gpu']) == cell['gpu'] and r['precision'] == 'bf16']
        if {(int(r['batch']), r['arm']) for r in rows} != {(b, a) for b in BATCHES for a in ARMS}:
            raise ValueError('DT-ANCHOR requires all five batches and four TRT/TC arms on this card')
        passed = [r for r in rows if r['status'] == 'pass']
        if any(int(r['correctness_processes']) < 5 or int(r['processes']) != 3 or
               r['strict_v1_status'] != 'pass' for r in passed):
            raise ValueError('resampled baselines lack required process/correctness gates')
    else:
        raise ValueError('unknown DT-ANCHOR action')
    eligible = [r for r in rows if r['model'] == cell['model'] and int(r['batch']) == cell['batch']
                and r['precision'] == 'bf16' and r['status'] == 'pass']
    if not eligible or any(not math.isfinite(float(r['median_ms'])) or
                           float(r['median_ms']) <= 0 for r in eligible):
        raise ValueError('missing or invalid eligible baseline')
    return min(eligible, key=lambda r: float(r['median_ms']))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells', type=Path, nargs='+', required=True)
    p.add_argument('--anchors', type=Path, nargs='+', required=True)
    p.add_argument('--native-gate', type=Path, required=True)
    p.add_argument('--resampled-csv', type=Path, nargs='*', default=[])
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    cells = [json.loads(path.read_text()) for path in args.cells]
    coverage(cells)
    gate = json.loads(args.native_gate.read_text())
    if gate['status'] != 'passed' or len(gate['rows']) != 80 or gate['ranks'] != list(range(4)):
        raise ValueError('final tables require the passed eighty-candidate native gate')
    cpu_path = Path(gate['cpu_gate'])
    if sha(cpu_path) != gate['cpu_gate_sha256']:
        raise ValueError('CPU solve receipt changed')
    cpu = json.loads(cpu_path.read_text())
    anchors = {}
    for path in args.anchors:
        a = json.loads(path.read_text()); key = (a['model'], a['gpu'])
        if key in anchors:
            raise ValueError('duplicate model/card anchor')
        anchors[key] = a
    frozen_paths = [ROOT / 'docs/results/DNN1_baselines.csv', ROOT / 'docs/results/DNN3_baselines.csv']
    def read_csv(paths):
        rows = []
        for path in paths:
            with path.open() as stream:
                rows.extend(csv.DictReader(stream))
        return rows
    frozen = read_csv(frozen_paths); resampled = read_csv(args.resampled_csv)
    inputs = {str(path.resolve()): sha(path) for path in [Path(__file__), args.native_gate,
              cpu_path, *args.cells, *args.anchors, *frozen_paths, *args.resampled_csv]}
    final = []; solver = []
    for cell, source in sorted(zip(cells, args.cells), key=lambda pair:
                               (MODELS.index(pair[0]['model']), pair[0]['batch'])):
        if cell['decision'] != json.loads(json.dumps(select(cell['selection']))):
            raise ValueError('recorded P-SEL decision differs from mirrored samples')
        if [r['rank'] for r in cell['selection']] != list(ORDER) or len(cell['final']) != 3:
            raise ValueError('incomplete P-SEL/P-DNN process coverage')
        positions = cell['selection'] + cell['final']
        if len({r['pid'] for r in positions}) != 11:
            raise ValueError('P-SEL/P-DNN positions did not use eleven fresh processes')
        match = [r for r in gate['rows'] if r['model'] == cell['model'] and r['batch'] == cell['batch']]
        if sorted(r['rank'] for r in match) != list(range(4)):
            raise ValueError('native gate lacks four unique cell candidates')
        artifacts = {r['rank']: r['artifact_id'] for r in match}
        clocks = []
        for row in positions:
            path = Path(row['source']); measured = json.loads(path.read_text())
            timing = measured['measurement']
            if (sha(path) != row['sha256'] or measured['median_ms'] != row['median_ms'] or
                    measured['p90_ms'] != row['p90_ms'] or measured['pid'] != row['pid'] or
                    measured['model'] != cell['model'] or measured['batch'] != cell['batch'] or
                    measured['gpu_index'] != cell['gpu'] or timing['warmup'] != 50 or
                    timing['iterations'] != 200 or measured['artifact_id'] != artifacts[row['rank']]):
                raise ValueError('raw process measurement differs from the cell result')
            inputs[str(path.resolve())] = sha(path)
            if row in cell['final']:
                clocks.append(measurement_clocks(measured, cell['gpu']))
        if cell['median_ms'] != statistics.median(r['median_ms'] for r in cell['final']):
            raise ValueError('final median differs from three fresh process medians')
        winner = cell['decision']['winner']
        chosen = next(r for r in match if r['rank'] == winner)
        if verify(Path(chosen['out']) / 'model.so')['artifact_id'] != cell['winner_artifact_id']:
            raise ValueError('winner artifact differs from the native gate')
        key = (cell['model'], cell['gpu'])
        if key not in anchors:
            raise ValueError('missing model/card anchors')
        ref = baseline(cell, anchors[key], frozen, resampled)
        ratio = cell['median_ms'] / float(ref['median_ms'])
        final.append(dict(model=cell['model'], batch=cell['batch'], gpu=cell['gpu'],
            median_ms=cell['median_ms'], p90_ms=cell['p90_ms'], process_range_ms=cell['process_range_ms'],
            baseline_arm=ref['arm'], baseline_ms=float(ref['median_ms']), ratio=ratio,
            faster_than_baseline=ratio < 1, winner_rank=winner, artifact_id=cell['winner_artifact_id'],
            clocks=json.dumps(clocks, separators=(',', ':')), evidence='verified',
            qualification=cell['qualification'], source=str(source.resolve())))
        cpu_row = next(r for r in cpu['rows'] if r['model'] == cell['model'] and r['batch'] == cell['batch'])
        search = json.loads(Path(cpu_row['receipt']).read_text())
        for row in sorted(match, key=lambda r: r['rank']):
            rank = row['rank']
            solver.append(dict(model=cell['model'], batch=cell['batch'], gpu=cell['gpu'], rank=rank,
                predicted_ns=search['ranked'][rank]['prediction']['forward_ns'],
                selection_ms=cell['decision']['medians_ms'][str(rank)], selected=rank == winner,
                artifact_id=row['artifact_id'], source=str(source.resolve()), evidence='verified'))
    ratios = [dict(scope='cell', model=r['model'], batch=r['batch'], ratio=r['ratio']) for r in final]
    for model in MODELS:
        ratios.append(dict(scope='model_geomean', model=model, batch='',
                           ratio=geometric([r['ratio'] for r in final if r['model'] == model])))
    for batch in BATCHES:
        ratios.append(dict(scope='batch_geomean', model='', batch=batch,
                           ratio=geometric([r['ratio'] for r in final if r['batch'] == batch])))
    ratios.append(dict(scope='overall_geomean', model='', batch='',
                       ratio=geometric([r['ratio'] for r in final])))
    args.out.mkdir(parents=True, exist_ok=False)
    for name, rows in [('final', final), ('solver', solver), ('ratios', ratios)]:
        write_csv(args.out / f'DNN3_{name}.csv', rows)
    (args.out / 'result.json').write_text(json.dumps(dict(evidence='verified', status='passed',
        cells=20, candidates=80, wins=sum(r['faster_than_baseline'] for r in final),
        inputs=inputs, ledger='separate required trace artifacts'), indent=2) + '\n')


if __name__ == '__main__':
    main()
