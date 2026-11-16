#!/usr/bin/env python3
"""Gate one BERT mechanism against the published winner on an idle timing GPU."""
import argparse
import hashlib
import json
from pathlib import Path

from dnn4_phase_z import Cell, ROOT, read, save


def main():
    driver_source = Path(__file__).read_bytes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', type=int, choices=(2, 3), default=2)
    parser.add_argument('--description', required=True)
    parser.add_argument('--decision', choices=('P-PAIR', 'DT-CODE'), default='P-PAIR')
    parser.add_argument('--gate', type=Path, help='reuse a completed strict correctness cohort')
    args = parser.parse_args()
    args.model, args.slot, args.cohorts = 'bert', 0, True
    args.solve = ROOT/'runs/dnn4/phase_z/solve_v1'
    cell = Cell(args)
    cell.device = args.device
    cell.result['gpu'] = args.device
    cell.cpu = {2: '104-111', 3: '112-119'}[args.device]
    (cell.out/'driver.py').write_bytes(driver_source)
    winner = read(ROOT/f'runs/dnn4/phase_z/cells_v1/bert_b{args.batch}/result.json')['winner']
    library = args.library.resolve()
    gate = args.gate.resolve() if args.gate else cell.gate(library)
    cohort = read(gate.parent.parent/'result.json')
    if cohort.get('status') != 'passed' or cohort.get('passes') != '5/5' or any(
            not output.get('output_sha256')
            for case in cohort['cases'] for output in case['outputs']):
        raise ValueError('requires five fresh byte-hashed FP32 reference processes')
    pair = cell.measure('pair', library, gate, 'P-PAIR', [
        '--reference-library', winner['library'], '--reference-gate', winner['gate'],
        '--reference-bridge', winner['bridge']])
    result = dict(evidence='verified', model='bert', batch=args.batch, gpu=args.device,
        protocol=args.description, library=str(library), gate=str(gate),
        reference=winner['library'], reference_gate=winner['gate'], pair=pair,
        decision=args.decision,
        accepted=pair['ratio'] < 1-max(pair['threshold'], .03 if args.decision == 'DT-CODE' else 0),
        qualification=('G22 comparison' if args.device == 3 else 'GPU2 diagnostic') +
                      ' against the published final winner',
        driver_sha256=hashlib.sha256(driver_source).hexdigest())
    save(cell.out/'experiment.json', result)
    print(json.dumps({k: pair[k] for k in ('reference_ms', 'candidate_ms', 'ratio', 'threshold')}))


if __name__ == '__main__':
    main()
