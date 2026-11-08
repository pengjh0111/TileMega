#!/usr/bin/env python3
"""Recheck the prescribed TMA stem against a completed cell on its own card."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

from dnn4_phase_z import Cell, ROOT, read, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--slot', type=int, choices=(0, 1), default=0)
    args = p.parse_args()
    original = read(args.cell / 'result.json')
    if original['status'] != 'passed' or original['model'] not in ('resnet18', 'resnet50', 'vgg16'):
        raise ValueError('TMA stem guard requires a completed CNN cell')
    base = ROOT / 'runs/dnn4/phase_z'
    options = argparse.Namespace(model=original['model'], batch=original['batch'], out=args.out,
        solve=base/'solve_v1', native=base/'rank0_native_v1', binary=base/'frozen/tilemega',
        target=ROOT/'configs/targets/sm_90_h800.json', slot=args.slot,
        compile_cache=base/'trace_precompiled_v1')
    cell = Cell(options)
    reference = Path(original['winner']['library'])
    coordinates = read(Path(str(reference) + '.plan.json'))['native_forward']
    folder = cell.out/'source'
    folder.mkdir(parents=True, exist_ok=True)
    coords, prefix = folder/'coordinates.json', folder/'input.so'
    save(coords, coordinates)
    cell.run('codegen-stem', ['env', 'TILEMEGA_F3_TMA_STEM=1', 'taskset', '-c', '128-135',
        options.binary, 'compile', cell.bridge, str(prefix)+'.cu', '--emit', 'serving',
        '--serving', 'forward', '--frontend', 'dnn', '--batch', options.batch,
        '--abi', 'f3', '--tmexec-config', coords, '--runtime-target', options.target],
        Path(str(prefix)+'.cu'))
    if 'Sm90F3Mode::kTmaStem' not in Path(str(prefix)+'.cu').read_text():
        raise ValueError('diagnostic codegen did not select the TMA stem')
    for old, new in [('.cu.plan.json', '.plan.json'), ('.cu.tmexec.mlir', '.tmexec.mlir')]:
        shutil.copyfile(str(prefix)+old, str(prefix)+new)
    native = cell.out/'native'
    cell.run('native-stem', [sys.executable, ROOT/'scripts/dnn4_compile.py', '--input', prefix,
        '--out', native, '--cpu-set', cell.compile_cpu, '--build-lock',
        f'/tmp/tilemega_dnn4_native_z{options.slot}.lock'], native/'result.json')
    library = native/'model.so'
    gate = cell.gate(library)
    prior_final = read(Path(original.get('final_measurement', args.cell/'final/result.json')))
    reference_gate = Path(read(Path(prior_final['measurements'][0]['measurement']))['numerical_gate_path'])
    paired = cell.measure('pair', library, gate, 'P-PAIR',
        ['--reference-library', reference, '--reference-gate', reference_gate,
         '--reference-bridge', cell.bridge])
    accepted = paired['ratio'] < 1-paired['threshold']
    result = dict(evidence='verified', protocol='DT-STEM same-cell end-to-end P-PAIR',
        model=options.model, batch=options.batch, accepted=accepted, pair=paired,
        reason='RN50B16 final trace stem101.696us misses60us gate;VGG rejection does not establish RN50 performance')
    if accepted:
        measured = cell.measure('final', library, gate, 'P-DNN')
        trace = cell.out/'trace1'
        cell.run('native-trace1', [sys.executable, ROOT/'scripts/dnn4_compile.py', '--input', library,
            '--out', trace, '--trace-level', '1', '--cpu-set', cell.compile_cpu, '--build-lock',
            f'/tmp/tilemega_dnn4_native_z{options.slot}.lock'], trace/'result.json')
        ledger = cell.trace('ledger', trace/'model.so', samples=1, reference=library)
        updated = copy.deepcopy(original)
        updated.update(winner=dict(library=str(library), gate=str(gate), bridge=str(cell.bridge),
                                   name='DT-STEM guard'), median_ms=measured['candidate_ms'],
                       p90_ms=measured['p90_ms'], process_range_ms=measured['process_range_ms'],
                       ledger=str(ledger/'ledger.csv'), final_measurement=str(cell.out/'final/result.json'),
                       stem_guard=str(cell.out/'result.json'))
        archive = args.cell/'result.before_stem_guard.json'
        if not archive.exists():
            shutil.copyfile(args.cell/'result.json', archive)
        result['winner'] = updated['winner']
        save(cell.out/'result.json', result)
        save(args.cell/'result.json', updated)
    else:
        save(cell.out/'result.json', result)
    print(json.dumps(dict(accepted=accepted, ratio=paired['ratio'], threshold=paired['threshold'])))


if __name__ == '__main__':
    main()
