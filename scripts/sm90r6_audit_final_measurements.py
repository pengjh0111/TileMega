#!/usr/bin/env python3
"""Recheck final P-ABL/P-PAIR timing processes and their immutable raw evidence."""
import argparse
import itertools
import math
from pathlib import Path
import statistics as st
from sm90d1_common import read,sha,write
from sm90r3_llm import processes,span
from sm90r6_ablate import ARMS


def audit_directory(directory,order,model,batch):
    inputs=read(directory/'inputs.json')
    if any(sha(path)!=digest for path,digest in inputs.items()):raise ValueError('measurement input changed')
    raw_path=directory/'unit/result.json';raw=read(raw_path);rows=raw['rows']
    if [r['arm'] for r in rows]!=order:raise ValueError('measurement order changed')
    gpus={r['gpu'] for r in rows}
    if len(gpus)!=1 or not gpus<={0,1,2,3,4,5,7}:raise ValueError('measurement card changed')
    checkpoints=set();pgids=set();pids=set();hashes={str(raw_path):sha(raw_path)}
    prompt='olmo' if model.startswith('olmo') else 'llama3' if model.startswith('llama') else 'qwen'
    prompt_path=Path(__file__).resolve().parents[1]/f'configs/prompts/{prompt}_isl64.json'
    for row in rows:
        tokens=Path(row['tokens']);job=tokens.parent.parent;attempt=job.parent
        guard=attempt/'guard_result.json';record=read(job/'sm90r3-process.json')
        pgid=(attempt/'pgid').read_text().strip()
        if read(guard)['code'] or record['exit_code'] or pgid in pgids:raise ValueError('unaccepted or reused timing process')
        pgids.add(pgid);paths=list(tokens.parent.rglob('measurements.json'))
        if len(paths)!=1:raise ValueError('ambiguous measurement')
        path=paths[0];data=read(path)
        if (data['pid'] in pids or data['batch']!=batch or data['max_tokens']!=1024 or
                data['gpu_indices']!=[row['gpu']] or data['prompt_ids_sha256']!=sha(prompt_path) or
                data['configuration']['decode_sha256']!=record['decode_sha256']):raise ValueError('timing process identity changed')
        pids.add(data['pid']);checkpoints.add(data['checkpoint']['checkpoint_sha256'])
        values={}
        for n in (1,1024):
            runs=[r for r in data['generation_runs'] if r['N']==n]
            if [(r['run'],r['warmup']) for r in runs]!=[(0,True),(1,False),(2,False),(3,False)]:raise ValueError('warmup or sample count changed')
            values[n]=st.median(r['wall_seconds'] for r in runs if not r['warmup'])
            reference=read(tokens) if n==1024 else None
            for run in runs:
                sample=path.parent/run['tokens_file'];value=read(sample)
                if len(value)!=batch or any(len(t)!=n for t in value):raise ValueError('incomplete token sample')
                if reference is not None and value!=reference:raise ValueError('within-process token mismatch')
                hashes[str(sample)]=sha(sample)
        expected={'ttft_ms':values[1]*1000,'e2e_ms':values[1024]*1000,
                  'tpot_ms':(values[1024]-values[1])*1000/1023}
        if any(not math.isclose(row[k],v,rel_tol=1e-12,abs_tol=1e-9) for k,v in expected.items()):raise ValueError('timing summary does not reproduce')
        for p in (tokens,path,guard,job/'sm90r3-process.json',attempt/'pgid'):hashes[str(p)]=sha(p)
    if len(checkpoints)!=1:raise ValueError('checkpoint changed between processes')
    for arm,summary in raw['arms'].items():
        values=[r['tpot_ms'] for r in rows if r['arm']==arm]
        if summary['tpot_ms']!=st.median(values) or summary['range_pct']!=span(values) or summary['processes']!=len(values):raise ValueError('arm summary changed')
    return dict(directory=str(directory),processes=len(rows),gpu=next(iter(gpus)),checkpoint=next(iter(checkpoints)),hashes=hashes)


def audit(root):
    units=[]
    for cell in read(root/'ab5/domains-v1/cells.json'):
        name=cell['id'];partial=name in ('olmoe_1b_7b-B1','qwen3_30b_a3b-B16')
        directory=root/'ablation'/('pabl-partial-v1' if partial else 'pabl-v1')/name/'measurement'
        measured=read(directory/'result.json');arms=tuple(arm for arm in ARMS if arm!=measured.get('excluded_arm'))
        order=[arm for r in range(3) for arm in arms[r:]+arms[:r]]
        units.append(dict(cell=name,protocol=measured['protocol'],**audit_directory(directory,order,cell['model'],cell['batch'])))
        pair=root/'ab5/pair-v1'/name;result=read(pair/'result.json')
        if (pair/'measurement/unit/result.json').exists():
            reps=list(dict.fromkeys(result['aliases'].values()));pairs=list(itertools.combinations(reps,2))
            order=[arm for pair_ in pairs for arm in list(pair_)*processes(cell['model'])]
            units.append(dict(cell=name,protocol='P-PAIR',**audit_directory(pair/'measurement',order,cell['model'],cell['batch'])))
    return dict(verified=True,units=len(units),accepted_processes=sum(u['processes'] for u in units),records=units)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=audit(a.root);write(a.out,result)
    print({k:v for k,v in result.items() if k!='records'})
