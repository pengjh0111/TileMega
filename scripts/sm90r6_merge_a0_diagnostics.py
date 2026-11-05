#!/usr/bin/env python3
"""Reproduce DT-A0 decisions and export their complete retained decomposition."""
import argparse
import json
from pathlib import Path
from sm90d1_common import read,sha
from sm90r6_diagnose_a0 import summarize,triggered
from sm90r6_merge_analysis import write

COMPONENTS=('original_ns','no_sync_ns','no_fixed_ns','infinite_ns','no_external_ns',
            'synchronization_ns','fixed_ns','contention_ns','chain_ns','pg_upper_bound_ns')


def merge(root,only_ready=False):
    rows=[]
    for cell in read(root/'ab5/domains-v1/cells.json'):
        name=cell['id'];directory=root/'ablation/dt-a0-v1'/name;path=directory/'result.json'
        if only_ready and not path.exists():continue
        data=read(path);measurement=Path(data['measurement']);measured=read(measurement)
        manifest=measurement.parent.parent/'arms.json'
        if (sha(measurement)!=data['measurement_sha256'] or
                sha(manifest)!=measured['source_sha256'] or triggered(measured)!=data['triggered'] or
                data['measurement_protocol']!=measured['protocol']):
            raise ValueError('DT-A0 measurement or decision changed')
        row=dict(model=cell['model'],batch=cell['batch'],triggered=data['triggered'],
            measurement_protocol=measured['protocol'],change_F_vs_A0_pct=measured['comparisons']['F']['change_pct'],
            threshold_pct=measured['comparisons']['F']['threshold_pct'],F_present_in_E='',E_size='',
            F_pool_rank='',F_replay_rank='',F_structure='',F_structure_in_finalists='',delta_replay_ns='')
        row.update({'delta_A0_minus_F_'+key:'' for key in COMPONENTS})
        row.update(action=data['action'],measurement=str(measurement),measurement_sha256=sha(measurement),
            diagnostic=str(path),diagnostic_sha256=sha(path),predictions='',predictions_sha256='',inputs_sha256='')
        if data['triggered']:
            inputs=directory/'inputs.json'
            if (sha(inputs)!=data['inputs_sha256'] or sha(data['predictions'])!=data['predictions_sha256'] or
                    any(sha(p)!=digest for p,digest in read(inputs).items())):
                raise ValueError('DT-A0 diagnostic evidence changed')
            source=read(root/'a0/finalists-v1'/name/'source.json')
            if sha(source['receipt'])!=source['receipt_sha256']:raise ValueError('A0 receipt changed')
            arms=read(manifest);coords={}
            for arm in ('A0','F'):
                binary=arms[arm]['binary'];identity=read(binary+'.identity.json')
                if sha(binary)!=identity['binary_sha256']:raise ValueError('DT-A0 binary changed')
                coords[arm]=identity['plan']['native_dense']
            predictions=[json.loads(line) for line in Path(data['predictions']).read_text().splitlines() if line.strip()]
            result=summarize(read(source['receipt']),predictions,coords['A0'],coords['F'])
            if any(data[key]!=value for key,value in result.items()):raise ValueError('DT-A0 decomposition does not reproduce')
            for key in ('F_present_in_E','E_size','F_pool_rank','F_replay_rank','F_structure','F_structure_in_finalists','delta_replay_ns'):
                row[key]=result[key]
            row.update({'delta_A0_minus_F_'+key:result['delta_A0_minus_F'][key] for key in COMPONENTS})
            row.update(predictions=data['predictions'],predictions_sha256=data['predictions_sha256'],inputs_sha256=data['inputs_sha256'])
        rows.append(row)
    if not rows:raise ValueError('no completed DT-A0 diagnostics')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--only-ready',action='store_true')
    a=p.parse_args();rows=merge(a.root,a.only_ready);write(a.out,rows)
    print(json.dumps(dict(cells=len(rows),triggered=sum(row['triggered'] for row in rows))))
