#!/usr/bin/env python3
"""Export DNN4 measured cells, fixed-target ratios, in-situ choices and ledger."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.build.identity import sha, verify

MODELS = ('resnet18', 'resnet50', 'vgg16', 'bert')
BATCHES = (1, 2, 4, 8, 16)
DEVICES = dict(resnet18=3, resnet50=0, vgg16=2, bert=3)


def read(path):
    return json.loads(path.read_text())


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def geometric(values):
    return math.exp(statistics.mean(map(math.log, values)))


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def append_existing(destination, name, additions):
    source = ROOT / 'docs/results' / name
    with source.open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    for addition in additions:
        row = {key: str(addition.get(key, '')) for key in fields}
        if row not in rows:
            rows.append(row)
    write(destination / name, rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells', type=Path, required=True)
    p.add_argument('--anchors', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--partial', action='store_true', help='export completed cells with explicit partial scope')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    targets = {(r['model'], int(r['batch'])): r for r in table(ROOT / 'docs/results/DNN3_final.csv')}
    cells = []
    for model in MODELS:
        for batch in BATCHES:
            path = args.cells / f'{model}_b{batch}/result.json'
            if not path.exists():
                if args.partial:
                    continue
                raise ValueError('missing completed cell: ' + str(path))
            result = read(path)
            if result['status'] != 'passed' or result['gpu'] != DEVICES[model]:
                raise ValueError('cell is incomplete or on the wrong card')
            cells.append((model, batch, path.parent, result))
    anchor_results = {}
    if not args.partial:
        if args.anchors is None:
            p.error('complete export requires --anchors')
        for model in MODELS:
            path = args.anchors / model / 'result.json'
            state = read(path)
            anchor = read(Path(state['out']) / 'anchor/result.json')
            if state['status'] != 'done' or anchor['status'] != 'passed' or anchor['gpu'] != DEVICES[model]:
                raise ValueError('P-ANCHOR did not complete on the prescribed card')
            if anchor['action'] != 'retain_frozen':
                raise ValueError('DT-ANCHOR resampling must be exported before complete assembly')
            anchor_results[model] = dict(source=str(path), sha256=sha(path), anchors=anchor['anchors'])
    finals, ledger, tune, codes, correctness, model_rows = [], [], [], [], [], []
    for model, batch, folder, result in cells:
        final = read(folder / 'final/result.json')
        measurements = final['measurements']
        if final['protocol'] != 'P-DNN' or len(measurements) != 3 or {m['arm'] for m in measurements} != {'B'}:
            raise ValueError('winner requires three fresh non-trace P-DNN processes')
        values = [m['median_ms'] for m in measurements]
        if final['candidate_ms'] != statistics.median(values):
            raise ValueError('final statistic differs from process-median median')
        winner = result['winner']
        library = Path(winner['library'])
        identity = verify(library)
        if identity['trace'] or identity['plan']['native_forward']['abi'] != 'f3':
            raise ValueError('final timing requires a non-trace F3 winner')
        gate = read(Path(winner['gate']).parent.parent / 'result.json')
        if gate['status'] != 'passed' or gate['passes'] != '5/5':
            raise ValueError('winner lacks five-process FP32 correctness')
        coordinate = identity['plan']['native_forward']
        stages = coordinate['stages']
        target = float(targets[(model, batch)]['baseline_ms'])
        ratio = final['candidate_ms'] / target
        summary = Counter(f"{s['tile_m']}x{s['tile_n']}:K{s['tk']}:S{s['split_k']}:L{s['slots']}:{s['store']}"
                          for s in stages if s['body'] in ('gemm', 'conv'))
        clocks = compact([{k: m[k] for k in ('start_sm_clock_mhz', 'end_sm_clock_mhz')} for m in measurements])
        finals.append(dict(model=model, batch=batch, gpu=result['gpu'], median_ms=final['candidate_ms'],
            p90_ms=final['p90_ms'], process_range_ms=final['process_range_ms'],
            baseline_arm=targets[(model, batch)]['baseline_arm'], baseline_ms=target,
            ratio=ratio, faster_than_baseline=ratio < 1, winner=winner['name'],
            coordinate_summary=compact(summary), artifact_id=identity['artifact_id'], clocks=clocks,
            evidence='verified', scope='partial Phase Z' if args.partial else 'complete anchored Phase Z',
            qualification=result['qualification'], source=str(folder / 'result.json')))
        for row in table(Path(result['ledger'])):
            stage = int(row['stage'])
            ledger.append(dict(model=model, batch=batch, gpu=result['gpu'], **row,
                coordinate=compact(stages[stage]), evidence='verified P-TRACE1;attribution only',
                source=result['ledger']))
        prepared = read(folder / 'tune_inputs.json')
        selected = read(folder / 'tune_selection.json')
        paired = read(folder / 'tuned_pair_decision.json')
        rank0_ms = paired.get('reference_ms')
        passes, model_source = '3/3', folder / 'tuned_pair/result.json'
        if rank0_ms is None:
            selection = read(folder / 'selection/result.json')
            rank0_library = str(next(c['library'] for c in read(folder / 'selection_candidates.json')
                                    if read(Path(c['library'] + '.plan.json'))['native_forward'] == prepared['rank0']))
            rank0_ms = next(c['median_ms'] for c in selection['ranking'] if c['library'] == rank0_library)
            passes, model_source = '2/2', folder / 'selection/result.json'
        predicted, observed = prepared['predicted_ns'], rank0_ms*1e6
        model_rows.append(dict(model=model, batch=batch, stage='TOTAL_Z', body='program',
            predicted_ns=predicted, observed_ns=observed, relative_error=abs(predicted-observed)/observed,
            actual_over_predicted=observed/predicted, passes=passes, protocol='P-PAIR/P-SEL rank0;DT-MODEL six choices',
            evidence='verified non-trace time;prediction remains unvalidated', source=str(model_source)))
        for group, selection in zip(prepared['groups'], selected['groups']):
            for rank, window in enumerate(selection['windows_ns']):
                option = group['candidates'][min(rank, len(group['candidates'])-1)]
                for stage in group['stages']:
                    tune.append(dict(model=model, batch=batch, gpu=result['gpu'], group=group['group'],
                        stage=stage, candidate=rank, group_window_ns=window, selected=rank == selection['candidate'],
                        coordinate=compact(option['coordinate']), processes='3/3', samples_per_process=50,
                        warmup=20, evidence='verified P-TRACE3;median of per-launch group sums',
                        source=str(folder / 'tune_selection.json')))
        sass = library.parent / 'model.sass'
        instructions = len(re.findall(r'/\*[0-9a-f]+\*/\s+[^/\n]', sass.read_text()))
        codes.append(dict(kind='f4_phase_z_winner', model=model, batch=batch,
                          sass_instructions=instructions, sass_bytes=instructions*16,
                          sass_sha256=sha(sass), library_sha256=sha(library), source=str(sass),
                          evidence='verified CPU disassembly;primary F3 kernel;16B instructions'))
        correctness.append(dict(phase='Z', suite=winner['name'], protocol='P-CORR', model=model,
            batch=batch, passes='5/5', status='passed', output_hash='', scope='final winner;FP32 oracle;five fresh deterministic processes',
            evidence='verified', source=str(Path(winner['gate']).parent.parent / 'result.json')))
    ratios = []
    if finals:
        for kind, groups in [('model', MODELS), ('batch', BATCHES), ('overall', ['all'])]:
            for group in groups:
                chosen = [r for r in finals if kind == 'overall' or r[kind] == group]
                if chosen:
                    ratios.append(dict(kind=kind, group=group, cells=len(chosen),
                        geometric_ratio=geometric([r['ratio'] for r in chosen]),
                        geometric_speedup=geometric([1/r['ratio'] for r in chosen]),
                        wins=sum(r['faster_than_baseline'] for r in chosen), evidence='verified',
                        scope='partial Phase Z' if args.partial else 'complete anchored Phase Z'))
    for name, rows in [('final', finals), ('ratios', ratios), ('tune', tune), ('ledger', ledger)]:
        write(args.out / f'DNN4_{name}.csv', rows)
    for name, additions in [('DNN4_correctness.csv', correctness), ('DNN4_icache.csv', codes),
                            ('DNN4_model.csv', model_rows)]:
        append_existing(args.out, name, additions)
    save_path = args.out / 'assembly.json'
    save_path.write_text(json.dumps(dict(status='partial' if args.partial else 'passed', cells=len(finals),
        fixed_targets_sha256=sha(ROOT / 'docs/results/DNN3_final.csv'), anchors=anchor_results,
        final_code=codes, correctness=correctness), indent=2) + '\n')
    print(json.dumps(dict(cells=len(finals), wins=sum(r['faster_than_baseline'] for r in finals), partial=args.partial)))


if __name__ == '__main__':
    main()
