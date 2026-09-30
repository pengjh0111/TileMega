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
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)

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
            seed=evaluated.split('\t')[2] if evaluated else ''
            top=archived/(so.name+'.top3.tsv')
            top_rows=list(csv.DictReader(top.open(),delimiter='\t')) if top.exists() else []
            cu=Path(str(so)+'.cu');src=cu.read_text() if cu.exists() else ''
            stages=re.search(r'constexpr StageDesc kStages\[\] = \{(.*?)\n\};',src,re.S)
            kinds=re.findall(r'\{TaskKind::(\w+),\s*(\d+)u',stages.group(1)) if stages else []
            runtime=len(kinds)+sum(g.get('split_k',1)>1 for g in manifest.get('gemms',[]))
            rows.append(dict(cell=cell,arm=arm['label'],pg=manifest.get('pg','off'),pages=json.dumps(manifest.get('pages')),gemms=json.dumps(manifest.get('gemms',[])),seed=seed,seed_in_top3=any(r.get('key')==seed for r in top_rows) if seed else 'unknown',solve_seconds=json.loads((archived/'record.json').read_text()).get('solve_seconds','unknown') if (archived/'record.json').exists() else 'unknown',top3=json.dumps(top_rows),budget=' | '.join(l for l in text.splitlines() if l.startswith('SEARCH_BUDGET')),spec_stages=len(kinds) or 'unknown',runtime_stages=runtime if kinds else 'unknown',elided_stages=sum(bool(re.search(r', true\},?$',line.strip())) for line in stages.group(1).splitlines())+sum(g.get('split_k',1)>1 for g in manifest.get('gemms',[])) if kinds and manifest.get('handoff')=='last_arriver' else 0 if kinds else 'unknown'))
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
    rows=[];tokens={};past_rows=[];canaries=[]
    for step in (HERE/'raw').glob('*'):
        for path in step.rglob('round*.json'):
            try:data=json.loads(path.read_text())
            except (ValueError,OSError):continue
            if data.get('invalidated'):continue
            for arm,r in data.get('arms',{}).items():
                if r.get('exit_code') or 'e2e_seconds' not in r:continue
                row=dict(step=step.name,cell=data['cell'],round=data['round'],arm=arm,ttft_s=r['ttft_seconds'],e2e_s=r['e2e_seconds'],tpot_s=(r['e2e_seconds']-r['ttft_seconds'])/1023,tpot_p50_s=r.get('tpot_p50_seconds',''),tpot_p90_s=r.get('tpot_p90_seconds',''),tok_s=r.get('output_tokens_per_second',''))
                vl=data['arms'].get('vllm',{});row['tm_vllm']=vl.get('e2e_seconds',0)/r['e2e_seconds'] if vl.get('e2e_seconds') else ''
                out=Path(r['out']);times=out/'step_times.tsv'
                if times.exists():
                    raw_times=list(csv.DictReader(times.open(),delimiter='\t'))
                    points=[(63+int(x['step']),float(x['gpu_ms'])) for x in raw_times if int(x['step'])>0]
                    if len(points)>1:
                        xbar=statistics.mean(x for x,y in points);ybar=statistics.mean(y for x,y in points)
                        row['slope_us_per_100_tokens']=100000*sum((x-xbar)*(y-ybar) for x,y in points)/sum((x-xbar)**2 for x,y in points)
                    for low,value in step_buckets(times).items():past_rows.append(dict(step=step.name,cell=data['cell'],arm=arm,round=data['round'],past_lo=low,mean_ms=value))
                # Only compare equal-numerics controls; DN and vLLM are excluded.
                runs=r.get('runs',r.get('generation_runs',[]))
                for run in runs:
                    if run.get('N')==1024 and not run.get('warmup'):
                        tok=run.get('tokens')
                        if tok is None and run.get('tokens_file'):
                            tp=out/run['tokens_file'];tok=json.loads(tp.read_text()) if tp.exists() else None
                        if tok is not None:tokens[(step.name[:2],data['cell'],arm,data['round'])]=tok
                        break
                model,batch=data['cell'].split('_B')
                for p in (HERE/'r12b_archive/TileMega/runs'/f'r12b-{model}/plans.json',):
                    if p.exists():
                        old=json.loads(p.read_text());so=Path(old[batch]['decode']);floor=HERE/'r12b_archive'/so.parent.relative_to('/root')/'floor.json'
                        if floor.exists():
                            row['floor_source']=str(floor.relative_to(HERE))
                            total=json.loads(floor.read_text())['sum_floor_seconds']
                            pf=Path(old[batch]['prefill']);prefill_floor=HERE/'r12b_archive'/pf.parent.relative_to('/root')/'floor.json'
                            if prefill_floor.exists():total+=json.loads(prefill_floor.read_text())['sum_floor_seconds']
                            row['e2e_over_sum_floor']=r['e2e_seconds']/total
                rows.append(row)
    table('observations.tsv',rows);table('past_observations.tsv',past_rows)
    grouped={}
    for row in rows:grouped.setdefault((row['step'].split('_')[0],row['cell'],row['arm']),[]).append(row)
    aggregates=[]
    for (matrix,cell,arm),values in grouped.items():
        entry=dict(matrix=matrix,cell=cell,arm=arm,rounds=len(values))
        for key in ('ttft_s','tpot_s','tpot_p50_s','tpot_p90_s','e2e_s','tok_s','tm_vllm','slope_us_per_100_tokens','e2e_over_sum_floor'):
            numeric=[r[key] for r in values if isinstance(r.get(key),(int,float))]
            if numeric:entry[key]=statistics.median(numeric);entry[key+'_range']=max(numeric)-min(numeric)
        aggregates.append(entry)
    table('aggregates.tsv',aggregates)
    for entry in aggregates:
        if entry['arm']=='vllm' or entry['arm'] in ('loop','P-base','R12bN_L2'):
            for row in grouped[(entry['matrix'],entry['cell'],entry['arm'])]:
                deviation=abs(row['e2e_s']/entry['e2e_s']-1)
                if deviation>.02:canaries.append(dict(matrix=entry['matrix'],cell=entry['cell'],round=row['round'],deviation=deviation,eligible_reruns=1))
    table('canaries.tsv',canaries)
    for prefixes,name in [(['A1'],'T1'),(['A4','B8'],'T2'),(['A3'],'T4'),(['B1','B3'],'T6'),(['B5'],'T8')]:
        table(name+'.tsv',[r for r in aggregates if r['matrix'] in prefixes])
    comparison_specs=[('A1','R12bN_L1','R12bN_L2','executor'),('B8','R10C','R10C_new','measurement tool'),('B8','R10C','R10G-0','code evolution'),('B8','R10G-0','R10G-1','V3'),('B8','R10G-1','R10G-2','L2 prefetch'),('B8','R10G-2','R10G-3','DN'),('B8','R10G-3','R10G-3_L2','executor at R10 geometry'),('B8','R10G-3','N-R12b','geometry/search'),('B8','N-R12b','R12bN_L1','compiler headers'),('A4','R12bN_L2','R12bN_L2_nowd','watchdog runtime lower bound')]
    comparisons=[]
    lookup={(r['matrix'],r['cell'],r['arm']):r for r in aggregates}
    for matrix,before,after,cause in comparison_specs:
        for cell in {r['cell'] for r in aggregates}:
            a=lookup.get((matrix,cell,before));b=lookup.get((matrix,cell,after))
            if not a or not b:continue
            delta=b['tpot_s']-a['tpot_s'];spread=max(a['tpot_s_range'],b['tpot_s_range'])
            comparisons.append(dict(cell=cell,cause=cause,before=before,after=after,tpot_delta_s=delta,ttft_delta_s=b['ttft_s']-a['ttft_s'],relative_change=b['tpot_s']/a['tpot_s']-1,resolvable=abs(delta)>spread,max_range_s=spread))
    table('T2.tsv',comparisons)
    control_changes=[]
    for r in aggregates:
        if r['matrix'] not in ('A3','B1','B3'):continue
        baseline='loop' if r['matrix']=='A3' else 'P-base'
        base=lookup.get((r['matrix'],r['cell'],baseline))
        if base:
            control_changes.append(dict(**r,baseline=baseline,relative_tpot_change=r['tpot_s']/base['tpot_s']-1,resolvable=abs(r['tpot_s']-base['tpot_s'])>max(r['tpot_s_range'],base['tpot_s_range'])))
    table('T4.tsv',[r for r in control_changes if r['matrix']=='A3']);table('T6.tsv',[r for r in control_changes if r['matrix'] in ('B1','B3')])
    candidate_rows=[]
    for path in (HERE/'raw').glob('A2_*/**/measurements.json'):
        data=json.loads(path.read_text())
        for past,value in data.get('modes',{}).get('L2',{}).get('by_past',{}).items():candidate_rows.append(dict(source=str(path.relative_to(HERE)),past=past,**value))
    # Match candidate protocol to the corresponding E2E past bucket (same SO).
    for r in candidate_rows:
        parts=Path(r['source']).parts;step=parts[1];cell=step[3:].rsplit('_r',1)[0];label=parts[2]
        low=64+128*((int(r['past'])-64)//128)
        values=[x['mean_ms'] for x in past_rows if x['step'].startswith('A1_') and x['cell']==cell and x['arm']==label and x['past_lo']==low]
        if values:r['candidate_over_e2e_bucket']=r['mean_ms']/statistics.median(values)
    table('T3.tsv',candidate_rows)
    from fidelity import fidelity
    candidates=[]
    for model in ('llama','qwen3'):
        plans=HERE.parents[2]/f'runs/r12c-{model}/plans.json'
        if plans.exists():
            for batch,pair in json.loads(plans.read_text()).items():
                if not batch.isdigit():continue
                for phase,so in pair.items():
                    for row in fidelity(so):candidates.append(dict(model=model,batch=batch,phase=phase,**row))
    table('T5.tsv',[dict(r,gemms=json.dumps(r['gemms']),pages=json.dumps(r['pages'])) for r in candidates])
    trace=[]
    for p in (HERE/'raw/B6').glob('**/summary.json'):
        data=json.loads(p.read_text());trace.append(dict(source=str(p.relative_to(HERE)),**data))
    table('T7.tsv',trace)
    checks=[]
    for p in (HERE/'raw').glob('B*/**/*.json'):
        if p.name in ('smoke.json','mode_check.json','summary.json','result.json','hf_check.json'):
            try:data=json.loads(p.read_text())
            except ValueError:continue
            checks.append(dict(source=str(p.relative_to(HERE)),result=json.dumps(data)))
    for matrix,baseline,allowed in [('A3','loop',None),('B1','P-base',None)]:
        for key,tok in tokens.items():
            m,cell,arm,rnd=key
            if m!=matrix or arm==baseline:continue
            reference=tokens.get((matrix,cell,baseline,rnd))
            if reference is not None:
                different=sum(a!=b for row_a,row_b in zip(reference,tok) for a,b in zip(row_a,row_b))
                checks.append(dict(source=matrix,cell=cell,arm=arm,round=rnd,token_mismatches=different))
    for key,tok in tokens.items():
        matrix,cell,arm,rnd=key
        if matrix=='B5' and arm=='R12cP' and rnd:
            reference=tokens.get((matrix,cell,arm,0))
            if reference is not None:checks.append(dict(source='B5 timed repeat',cell=cell,round=rnd,token_mismatches=sum(a!=b for x,y in zip(reference,tok) for a,b in zip(x,y))))
    table('T9.tsv',checks)
    print('analysis: '+str(len(rows))+' collected arm observations')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--t0',action='store_true');a=p.parse_args()
    if a.t0:t0()
    else:collected()
