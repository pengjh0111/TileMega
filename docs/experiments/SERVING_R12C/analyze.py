#!/usr/bin/env python3
"""CPU-only evidence tables. Missing observations remain missing, never zero."""
import argparse,csv,json,re,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent

def table(name,rows):
    out=HERE/'results'/name;out.parent.mkdir(parents=True,exist_ok=True)
    if not rows:out.write_text('status\nnot_collected\n');return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with out.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)

def t0():
    base=HERE/'r12b_archive';rows=[]
    for path in base.glob('TileMega/runs/r12b-*/plans.json'):
        data=json.loads(path.read_text());model=path.parent.name.replace('r12b-','')
        for batch,choice in data.get('decode_pg_choice',{}).items():
            for pg,values in choice['samples_ms'].items():rows.append(dict(model=model,batch=batch,pg=pg,samples_ms=json.dumps(values),median_ms=statistics.median(values),selected=choice['selected']))
    table('T0_pg_choice.tsv',rows);rows=[]
    arms=json.loads((HERE/'arms.json').read_text())
    for cell,items in arms.items():
        for arm in items:
            if arm['kind']=='vllm':continue
            so=Path(arm['decode']);manifest=arm.get('binaries',{}).get('decode',{}).get('manifest',{})
            archived=base/so.parent.relative_to('/root')
            search=archived/(so.name+'.search.tsv')
            text=search.read_text() if search.exists() else ''
            evaluated=next((line for line in text.splitlines() if line.startswith('EVALUATE\t')),'')
            seed=evaluated.split('\t')[1] if evaluated else ''
            top=archived/(so.name+'.top3.tsv')
            top_rows=list(csv.DictReader(top.open(),delimiter='\t')) if top.exists() else []
            cu=Path(str(so)+'.cu');src=cu.read_text() if cu.exists() else ''
            stages=re.search(r'constexpr StageDesc kStages\[\] = \{(.*?)\n\};',src,re.S)
            kinds=re.findall(r'\{TaskKind::(\w+),\s*(\d+)u',stages.group(1)) if stages else []
            runtime=len(kinds)+sum(g.get('split_k',1)>1 for g in manifest.get('gemms',[]))
            rows.append(dict(cell=cell,arm=arm['label'],pg=manifest.get('pg','off'),pages=json.dumps(manifest.get('pages')),gemms=json.dumps(manifest.get('gemms',[])),seed=seed,seed_in_top3=any(r.get('key')==seed for r in top_rows) if seed else 'unknown',budget=' | '.join(l for l in text.splitlines() if l.startswith('SEARCH_BUDGET')),spec_stages=len(kinds) or 'unknown',runtime_stages=runtime if kinds else 'unknown',elided_stages=src.count('true /* handoff_elided */') if kinds else 'unknown'))
    table('T0_plans.tsv',rows)
    rows=[]
    for path in base.rglob('step_times.tsv'):
        bins=step_buckets(path)
        for low,value in bins.items():rows.append(dict(source=str(path.relative_to(HERE)),past_lo=low,mean_ms=value))
    table('T0_past.tsv',rows)

def step_buckets(path):
    rows=list(csv.DictReader(path.open(),delimiter='\t'));bins={}
    for row in rows:
        step=int(row['step'])
        if step==0:continue
        past=63+step;lo=64+128*((past-64)//128)
        if 64<=lo<1088:bins.setdefault(lo,[]).append(float(row['gpu_ms']))
    return {lo:statistics.mean(v) for lo,v in bins.items()}

def collected():
    rows=[]
    for step in (HERE/'raw').glob('*'):
        for path in step.rglob('round*.json'):
            try:data=json.loads(path.read_text())
            except (ValueError,OSError):continue
            if data.get('invalidated'):continue
            for arm,r in data.get('arms',{}).items():
                if r.get('exit_code') or 'e2e_seconds' not in r:continue
                row=dict(step=step.name,cell=data['cell'],round=data['round'],arm=arm,ttft_s=r['ttft_seconds'],e2e_s=r['e2e_seconds'],tpot_s=(r['e2e_seconds']-r['ttft_seconds'])/1023,tpot_p50_s=r.get('tpot_p50_seconds',''),tpot_p90_s=r.get('tpot_p90_seconds',''),tok_s=r.get('output_tokens_per_second',''))
                vl=data['arms'].get('vllm',{});row['tm_vllm']=vl.get('e2e_seconds',0)/r['e2e_seconds'] if vl.get('e2e_seconds') else ''
                rows.append(row)
    table('observations.tsv',rows)
    for prefix,name in [('A1','T1'),('A4','T2'),('A2','T3'),('A3','T4'),('B2','T5'),('B1','T6'),('B6','T7'),('B5','T8'),('B7','T9')]:
        table(name+'.tsv',[r for r in rows if r['step'].startswith(prefix)])
    print('analysis: '+str(len(rows))+' collected arm observations')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--t0',action='store_true');a=p.parse_args()
    if a.t0:t0()
    else:collected()
