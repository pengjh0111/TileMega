#!/usr/bin/env python3
"""Export CPU selector audits and three-process boundary-window model errors."""
import argparse
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solve', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    solved = json.loads((args.solve / 'result.json').read_text())
    solver_rows = []
    model_rows = []
    gates = []
    for cell in solved['rows']:
        if cell['status'] != 'passed':
            continue
        receipt = json.loads(Path(cell['receipt']).read_text())
        model, batch = cell['model'], cell['batch']
        key = f'{model}_b{batch}'
        for rank, candidate in enumerate(receipt['ranked']):
            histogram = collections.Counter(
                tuple(s[k] for k in ('tile_m', 'tile_n', 'tk', 'split_k', 'slots', 'store', 'mc'))
                for s in candidate['coordinates']['stages'] if s['body'] in ('gemm', 'conv'))
            coords = Path(cell['candidates'][rank]['coordinates'])
            solver_rows.append(dict(model=model, batch=batch, rank=rank,
                origin=candidate['origin'], predicted_ms=candidate['prediction']['forward_ns'] / 1e6,
                coordinate_summary=json.dumps(histogram.most_common(), separators=(',', ':')),
                coordinates_sha256=hashlib.sha256(coords.read_bytes()).hexdigest(),
                codegen_passed=True, native_compile_scope='separate artifact receipts',
                coordinate_sensitivity=json.dumps(cell['sensitivity']['fields'], separators=(',', ':')),
                candidate_selfcheck=cell['selfcheck']['passed'],
                evidence='verified CPU solve/codegen only;prediction is an estimate', source=cell['receipt']))
        trace_paths = []
        for process in range(3):
            matches = list((args.models / f'{key}_marks3_p{process}').glob('raw/*/attempt*/trace/ledger.csv'))
            if len(matches) == 1:
                trace_paths.append(matches[0])
        if len(trace_paths) != 3:
            continue
        traces = []
        for path in trace_paths:
            with path.open(newline='') as stream:
                traces.append(list(csv.DictReader(stream)))
        predicted = receipt['ranked'][0]['prediction']
        matrix_errors = []
        cell_rows = []
        for stage, pred in enumerate(predicted['stages']):
            observed = statistics.median(float(t[stage]['stage_ns']) for t in traces)
            value = pred['stage_ns']
            error = abs(value - observed) / observed
            if pred['body'] in ('gemm', 'conv'):
                matrix_errors.append(error)
            cell_rows.append(dict(model=model, batch=batch, stage=stage, body=pred['body'],
                predicted_ns=value, observed_ns=observed, relative_error=error,
                actual_over_predicted=observed / value, passes='3/3', protocol='P-MODEL/P-TRACE3',
                evidence='verified boundary windows;prediction is an estimate',
                source=';'.join(str(p) for p in trace_paths)))
        observed = statistics.median(sum(float(s['stage_ns']) for s in t) for t in traces)
        error = abs(predicted['forward_ns'] - observed) / observed
        median = statistics.median(matrix_errors)
        p90 = percentile(matrix_errors, .9)
        passed = median <= .2 and p90 <= .4 and error <= .15
        gates.append(dict(model=model, batch=batch, matrix_median_error=median,
            matrix_p90_error=p90, total_error=error, passed=passed,
            worst_stages=sorted(cell_rows, key=lambda r: r['relative_error'], reverse=True)[:5]))
        model_rows += cell_rows
        model_rows.append(dict(model=model, batch=batch, stage='total', body='program',
            predicted_ns=predicted['forward_ns'], observed_ns=observed, relative_error=error,
            actual_over_predicted=observed / predicted['forward_ns'], passes='3/3',
            protocol='P-MODEL/P-TRACE3', evidence='verified boundary windows;prediction is an estimate',
            source=';'.join(str(p) for p in trace_paths)))
    write(ROOT / 'docs/results/DNN4_solver.csv', solver_rows)
    write(ROOT / 'docs/results/DNN4_model.csv', model_rows)
    result = dict(status='partial' if len(gates) < 4 else 'passed' if all(g['passed'] for g in gates) else 'failed',
        evidence='verified', coordinate_sensitivity=solved.get('coordinate_sensitivity'),
        candidate_selfcheck_failed_cells=solved.get('candidate_selfcheck_failed_cells'), gates=gates)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
