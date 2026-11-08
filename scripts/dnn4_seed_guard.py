#!/usr/bin/env python3
"""Guard a completed Phase Z selection against an accepted coordinate."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from dnn4_phase_z import Cell, ROOT, read, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell', type=Path, required=True)
    p.add_argument('--accepted-library', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--slot', type=int, choices=(0, 1), default=1)
    p.add_argument('--reason', default='known accepted seed excluded from model top-six pool;latest user authorizes end-to-end decisions')
    p.add_argument('--winner-name', default='accepted_seed_guard')
    args = p.parse_args()
    original = read(args.cell / 'result.json')
    if original['status'] != 'passed':
        raise ValueError('accepted-seed guard requires a completed Phase Z cell')
    base = ROOT / 'runs/dnn4/phase_z'
    options = argparse.Namespace(model=original['model'], batch=original['batch'], out=args.out,
        solve=base/'solve_v1', native=base/'rank0_native_v1', binary=base/'frozen/tilemega',
        target=ROOT/'configs/targets/sm_90_h800.json', slot=args.slot,
        compile_cache=base/'trace_precompiled_v1', cohorts=True)
    cell = Cell(options)
    cell.prepared = read(args.cell / 'tune_inputs.json')
    accepted_plan = read(Path(str(args.accepted_library) + '.plan.json'))
    if accepted_plan['runtime_batch'] != options.batch or accepted_plan['model'] != cell.bridge.stem:
        raise ValueError('accepted seed belongs to a different cell')
    coordinates = accepted_plan['native_forward']
    reference = Path(original['winner']['library'])
    if coordinates == read(Path(str(reference)+'.plan.json'))['native_forward']:
        save(cell.out/'result.json', dict(evidence='verified', model=options.model,
            batch=options.batch, accepted=False, deduplicated=True,
            reason='accepted coordinate already matches the final winner;reuse existing measurement'))
        return
    library = cell.build(coordinates)
    gate = cell.gate(library)
    prior_final = read(Path(original.get('final_measurement', args.cell / 'final/result.json')))
    reference_gate = Path(read(Path(prior_final['measurements'][0]['measurement']))['numerical_gate_path'])
    paired = cell.measure('pair', library, gate, 'P-PAIR',
        ['--reference-library', reference, '--reference-gate', reference_gate,
         '--reference-bridge', cell.bridge])
    accepted = paired['ratio'] < 1-paired['threshold']
    result = dict(evidence='verified', protocol='post-selection P-PAIR against an accepted coordinate',
        model=options.model, batch=options.batch, accepted=accepted, pair=paired,
        reason=args.reason)
    if accepted:
        measured = cell.measure('final', library, gate, 'P-DNN')
        trace = cell.build(coordinates, trace=1)
        ledger = cell.trace('ledger', trace, samples=1, reference=library)
        updated = copy.deepcopy(original)
        updated.update(winner=dict(library=str(library), gate=str(gate), bridge=str(cell.bridge),
                                   name=args.winner_name), median_ms=measured['candidate_ms'],
                       p90_ms=measured['p90_ms'], process_range_ms=measured['process_range_ms'],
                       ledger=str(ledger/'ledger.csv'), final_measurement=str(cell.out/'final/result.json'),
                       accepted_seed_guard=str(cell.out/'result.json'))
        archive = args.cell / 'result.before_seed_guard.json'
        if not archive.exists():
            shutil.copyfile(args.cell / 'result.json', archive)
        result['winner'] = updated['winner']
        save(cell.out / 'result.json', result)
        save(args.cell / 'result.json', updated)
    else:
        save(cell.out / 'result.json', result)
    print(json.dumps(dict(accepted=accepted, ratio=paired['ratio'], threshold=paired['threshold'])))


if __name__ == '__main__':
    main()
