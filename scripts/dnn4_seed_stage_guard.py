#!/usr/bin/env python3
"""Check one existing seed per P-TUNE group, then guard the assembled model."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import statistics
import sys

from dnn4_phase_z import Cell, ROOT, read, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell',type=Path,required=True)
    p.add_argument('--seed-library',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--slot',type=int,choices=(0,1),default=0)
    args=p.parse_args()
    original=read(args.cell/'result.json')
    if original['status']!='passed':
        raise ValueError('stage seed guard requires a completed cell')
    base=ROOT/'runs/dnn4/phase_z'
    options=argparse.Namespace(model=original['model'],batch=original['batch'],out=args.out,
        solve=base/'solve_v1',native=base/'rank0_native_v1',binary=base/'frozen/tilemega',
        target=ROOT/'configs/targets/sm_90_h800.json',slot=args.slot,
        compile_cache=base/'trace_precompiled_v1',cohorts=True)
    cell=Cell(options)
    prepared=read(args.cell/'tune_inputs.json')
    selected=read(args.cell/'tune_selection.json')
    cell.prepared=prepared
    seed_plan=read(Path(str(args.seed_library)+'.plan.json'))
    if seed_plan['model']!=cell.bridge.stem or seed_plan['runtime_batch']!=options.batch:
        raise ValueError('seed belongs to a different model/batch')
    reference=Path(original['winner']['library'])
    coordinates=read(Path(str(reference)+'.plan.json'))['native_forward']
    seed=seed_plan['native_forward']
    if len(seed['stages'])!=len(coordinates['stages']):
        raise ValueError('seed and winner stage counts differ')
    probe=copy.deepcopy(coordinates)
    for group in prepared['groups']:
        if len({json.dumps(seed['stages'][i],sort_keys=True) for i in group['stages']})!=1:
            raise ValueError('seed breaks the existing P-TUNE group equivalence')
        for i in group['stages']:
            probe['stages'][i]=copy.deepcopy(seed['stages'][i])
    library=cell.build(probe)
    cell.gate(library)
    trace=cell.build(probe,trace=3)
    dest=cell.out/'seed_windows'
    cell.run('seed_windows',[sys.executable,ROOT/'scripts/dnn4_trace_cohort.py',
        '--library',trace,*cell.common(),'--device',cell.device,'--cpu-set',cell.cpu,
        '--out',dest],dest/'result.json')
    processes=[read(Path(path)/'samples.json') for path in read(dest/'result.json')['traces']]
    if len(processes)!=3:
        raise ValueError('seed requires three fresh trace processes')
    assembled=copy.deepcopy(coordinates)
    choices=[]
    for group,old in zip(prepared['groups'],selected['groups']):
        if group['stages']!=old['stages']:
            raise ValueError('P-TUNE group order changed')
        window=statistics.median(statistics.median(sum(sample[i]['stage_ns'] for i in group['stages'])
            for sample in process) for process in processes)
        prior=min(old['windows_ns'])
        take=window<prior
        if take:
            for i in group['stages']:
                assembled['stages'][i]=copy.deepcopy(seed['stages'][i])
        choices.append(dict(group=group['group'],stages=group['stages'],prior_window_ns=prior,
                            seed_window_ns=window,selected=take,
                            coordinate=seed['stages'][group['stages'][0]]))
    result=dict(evidence='verified',protocol='bounded existing-seed per-stage rescue',
        model=options.model,batch=options.batch,gpu=cell.device,choices=choices,
        seed_library=str(args.seed_library.resolve()),reference=str(reference),accepted=False,
        limitation='one additional existing coordinate per group;historical windows guide assembly;fresh whole-model P-PAIR decides admission')
    save(cell.out/'choices.json',result)
    if assembled==coordinates:
        result['reason']='seed did not change the selected coordinates'
        save(cell.out/'result.json',result)
        return
    library=cell.build(assembled)
    gate=cell.gate(library)
    final=read(Path(original.get('final_measurement',args.cell/'final/result.json')))
    reference_gate=Path(read(Path(final['measurements'][0]['measurement']))['numerical_gate_path'])
    pair=cell.measure('pair',library,gate,'P-PAIR',
        ['--reference-library',reference,'--reference-gate',reference_gate,
         '--reference-bridge',cell.bridge])
    result.update(pair=pair,accepted=pair['ratio']<1-pair['threshold'])
    if result['accepted']:
        measured=cell.measure('final',library,gate,'P-DNN')
        trace=cell.build(assembled,trace=1)
        ledger=cell.trace('ledger',trace,samples=1,reference=library)
        updated=copy.deepcopy(original)
        updated.update(winner=dict(name='seed_stage_guard',library=str(library),gate=str(gate),
                                   bridge=str(cell.bridge)),median_ms=measured['candidate_ms'],
            p90_ms=measured['p90_ms'],process_range_ms=measured['process_range_ms'],
            ledger=str(ledger/'ledger.csv'),final_measurement=str(cell.out/'final/result.json'),
            seed_stage_guard=str(cell.out/'result.json'))
        archive=args.cell/'result.before_seed_stage_guard.json'
        if not archive.exists():
            shutil.copyfile(args.cell/'result.json',archive)
        result['winner']=updated['winner']
        save(cell.out/'result.json',result)
        save(args.cell/'result.json',updated)
    else:
        save(cell.out/'result.json',result)
    print(json.dumps(dict(accepted=result['accepted'],ratio=pair['ratio'],threshold=pair['threshold'])))


if __name__=='__main__':
    main()
