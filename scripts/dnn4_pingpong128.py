#!/usr/bin/env python3
"""Check the prescribed 128-row WG ping-pong tile on BERT B16."""
import argparse
import copy
import json
from pathlib import Path

from dnn4_phase_z import Cell, ROOT, read, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--unit', type=Path, required=True)
    args = parser.parse_args()
    base = ROOT / 'runs/dnn4/phase_z'
    options = argparse.Namespace(model='bert', batch=16, out=args.out,
        solve=base/'solve_v1', native=base/'rank0_native_v1',
        binary=base/'frozen/tilemega', target=ROOT/'configs/targets/sm_90_h800.json',
        slot=0, compile_cache=base/'trace_precompiled_v1')
    cell = Cell(options)
    # This is a Phase X sentinel comparison, separate from the G22 final card.
    cell.device, cell.cpu = 2, '104-111'
    rank0 = read(cell.source/'winner.cu.search/candidate-0.tmexec.json')
    cell.prepared = dict(rank0=rank0)
    coordinates = copy.deepcopy(rank0)
    selected = [1+5*layer+offset for layer in range(12) for offset in (0, 3)]
    for index in selected:
        stage = coordinates['stages'][index]
        if stage['body'] != 'gemm' or stage['split_k'] != 1:
            raise ValueError('expected unsplit QKV/FFN1 stage')
        stage.update(tile_m=256, tile_n=128, slots=6, split_k=1,
                     tk=64, store='reg', mc=0, sched='pp')
    save(cell.out/'coordinates.json', coordinates)
    library = cell.build(coordinates)
    unit = read(args.unit/'scheduler/state.json')['dnn4-focused-f3-gates']
    if unit['status'] != 'done':
        raise ValueError('128-row ping-pong P-UNIT/P-SAN has not completed')
    unit_result = read(Path(unit['out'])/'cohort/result.json')
    if not unit_result.get('passed') or unit_result['unit_passes'] != '5/5':
        raise ValueError('128-row ping-pong P-UNIT/P-SAN failed')
    gate = cell.gate(library)
    reference = cell.build(rank0)
    reference_gate = cell.gate(reference)
    pair = cell.measure('pair', library, gate, 'P-PAIR',
        ['--reference-library', reference, '--reference-gate', reference_gate,
         '--reference-bridge', cell.bridge])
    result = dict(evidence='verified', protocol='DT-PP prescribed WG128 recheck',
        model='bert', batch=16, gpu=2, stages=selected,
        unit=str(Path(unit['out'])/'cohort/result.json'), library=str(library),
        reference=str(reference), gate=str(gate), pair=pair,
        accepted=pair['ratio'] <= .97,
        scope='Phase X sentinel;not a Phase Z winner;no automatic default change')
    save(cell.out/'result.json', result)
    print(json.dumps(dict(accepted=result['accepted'], ratio=pair['ratio'])))


if __name__ == '__main__':
    main()
