#!/usr/bin/env python3
"""Run the eight R6 ablation arms in three rotating rounds on one card."""
import argparse
import json
from pathlib import Path

from sm90d1_common import read,sha,write
from sm90r4_dispatch import run_unit

ARMS=('A0','AB1a','AB1b','AB2','AB4a','AB4b','AB4c','F')


def validate(arms,model,batch):
    if set(arms)!=set(ARMS):raise ValueError('P-ABL requires all eight arms including F')
    result={};identities={}
    for label in ARMS:
        arm=arms[label];binary=Path(arm['binary']).resolve();digest=sha(binary)
        identity=read(str(binary)+'.identity.json')
        identities[label]=identity
        if identity['binary_sha256']!=digest:raise ValueError('changed binary: '+label)
        c2=read(arm['c2'])
        if (not c2['passed'] or len(c2['processes'])<5 or c2['steps']!=1024 or
                c2['model']!=model or c2['batch']!=batch or
                c2['artifacts']['candidate']['sha256']!=digest):
            raise ValueError('missing winner C-2 evidence: '+label)
        if label in ('AB4b','AB4c'):
            c1=read(arm['c1']);spec=c1['spec']
            if (not c1['pass'] or not c1['c1_pass'] or spec['correctness']['c1']!=256 or
                    spec['model']!=model or spec['batch']!=batch or
                    {a['decode_sha256'] for a in spec['arms'].values()}!={digest}):
                raise ValueError('missing 256-position C-1 evidence: '+label)
        result[label]=dict(decode=str(binary),prefill='prompt')
    if read(arms['AB4c']['c2'])['artifacts']['reference']['sha256']!=sha(arms['AB4b']['binary']):
        raise ValueError('AB4c must match the selected AB4b megakernel')
    reference,candidate=identities['AB4b'],identities['AB4c']
    if (candidate['plan'].get('execution')!='kernel_sequence' or
            any(reference['plan'][key]!=candidate['plan'][key] for key in
                ('native_dense','export_sha256','grid','runtime_batch')) or
            reference['shared_memory_bytes']!=candidate['shared_memory_bytes']):
        raise ValueError('AB4c changed the selected AB4b coordinates or launch geometry')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--batch',type=int,choices=(1,16),required=True)
    p.add_argument('--arms',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();arms=read(a.arms)
    validated=validate(arms,a.model,a.batch)
    a.out.mkdir(parents=True,exist_ok=False)
    paths=[a.arms,Path(__file__)]
    for arm in arms.values():
        paths.extend(Path(arm[key]) for key in ('binary','c2','c1') if key in arm)
        paths.append(Path(arm['binary']+'.identity.json'))
    inputs={str(path.resolve()):sha(path) for path in paths};write(a.out/'inputs.json',inputs)
    sequence=[label for round in range(3) for label in ARMS[round:]+ARMS[:round]]
    # The existing guarded selector runner supports an explicit process order;
    # P-ABL adds comparisons against A0 without selecting a replacement winner.
    spec=dict(name=f'sm90r6-abl-{a.model}-B{a.batch}',kind='sel',protocol='P-ABL',
        model=a.model,batch=a.batch,gpu='auto',concurrency='trial',arms=validated,order=sequence)
    write(a.out/'spec.json',spec)
    result=run_unit(a.out/'spec.json',a.out/'unit','trial')
    if any(sha(path)!=digest for path,digest in inputs.items()):raise ValueError('P-ABL inputs changed during collection')
    rows=result['rows']
    if ([r['arm'] for r in rows]!=sequence or len({r['gpu'] for r in rows})!=1 or
            any(r['gpu']==6 for r in rows) or any(r['processes']!=3 for r in result['arms'].values())):
        raise ValueError('P-ABL process order, same-card rule or process count failed')
    checkpoints=set()
    for row in rows:
        paths=list(Path(row['tokens']).parent.rglob('measurements.json'))
        if len(paths)!=1:raise ValueError('ambiguous P-ABL process measurements')
        measurement=read(paths[0]);checkpoints.add(measurement['checkpoint']['checkpoint_sha256'])
    if len(checkpoints)!=1:raise ValueError('P-ABL checkpoints differ across processes')
    reference=result['arms']['A0'];comparisons={}
    for label,value in result['arms'].items():
        change=(value['tpot_ms']/reference['tpot_ms']-1)*100
        threshold=max(.5,2*max(value['range_pct'],reference['range_pct']))
        comparisons[label]=dict(change_pct=change,threshold_pct=threshold,
            decision='indistinguishable' if abs(change)<=threshold else 'faster' if change<0 else 'slower')
    write(a.out/'result.json',dict(protocol='P-ABL',model=a.model,batch=a.batch,
        arms=result['arms'],comparisons=comparisons,gpu=rows[0]['gpu'],order=sequence,
        checkpoint=checkpoints.pop(),evidence='verified',timing=True,
        ownership='provisional_container_pid_namespace',source=str(a.arms.resolve()),
        source_sha256=sha(a.arms),raw=str((a.out/'unit/result.json').resolve())))
    print(json.dumps(dict(model=a.model,batch=a.batch,arms=8,processes=len(rows))),flush=True)


if __name__=='__main__':main()
