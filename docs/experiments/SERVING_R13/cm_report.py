#!/usr/bin/env python3
"""Report-only task-price errors and bounded linear corrections; no defaults mutate."""
import argparse,json,statistics,re
from collections import defaultdict
from pathlib import Path
from ledger import read,write
from fidelity import tau

def fit(points):
    if not points:return None
    x=statistics.mean(p[0] for p in points);y=statistics.mean(p[1] for p in points)
    denominator=sum((a-x)**2 for a,b in points)
    if denominator==0:return dict(identifiable=False,fixed_ns_bias=y,byte_ns_bias=None,n=len(points))
    slope=sum((a-x)*(b-y) for a,b in points)/denominator
    return dict(identifiable=True,fixed_ns_bias=y-slope*x,byte_ns_bias=slope,n=len(points))

def compare(prices,trace,manifest=None):
    prediction=[{k:(v if k in ('space','category','handoff_priced') else float(v)) for k,v in r.items()} for r in read(prices)]
    actual={}
    for r in read(trace/'slots.tsv'):
        if int(r['run_begin']):actual[(int(r['stage']),int(r['logical_task']))]=int(r['run_end'])-int(r['run_begin'])
    counts=defaultdict(int)
    for r in prediction:counts[int(r['stage'])]+=1
    stages={int(r['stage']):r for r in read(trace/'runtime_stages.tsv')}
    geometry={};gemm=-1
    if manifest:
        gemms=json.loads(Path(manifest).read_text())['gemms']
        for stage,r in sorted(stages.items()):
            if int(r['kind'])==0:gemm+=1
            if int(r['kind']) in (0,6):
                if gemm<0 or gemm>=len(gemms):raise ValueError('runtime GEMM order differs from manifest')
                g=gemms[gemm]
                geometry[stage]='x'.join(str(g[k]) for k in ('tile_m','tile_n','tile_k','stages','split_k'))
        if gemm+1!=len(gemms):raise ValueError('runtime GEMM count differs from manifest')
    # Refuse joins across numbering systems. The flow uses the projected
    # runtime graph; every projected stage/task count must agree with its dump.
    if set(counts)!=set(stages) or any(counts[s]!=int(stages[s]['active_tasks']) for s in stages):
        return [],dict(status='incompatible projected stage/task counts',prices=str(prices),trace=str(trace))
    output=[]
    for r in prediction:
        key=(int(r['stage']),int(r['task']))
        if key not in actual:continue  # Elided reducers are reported separately.
        measured=actual[key];predicted=r['predicted_ns']
        output.append(dict(stage=key[0],task=key[1],category=r['category'],space=r['space'],geometry=geometry.get(key[0],'non-GEMM'),
            measured_run_ns=measured,predicted_ns=predicted,relative_error=measured/predicted-1 if predicted else None,
            fixed_ns=r['fixed_ns'],compute_ns=r['compute_ns'],dram_bytes=r['dram_bytes'],
            residual_ns=measured-predicted,pricing_dram_gbps=r['dram_gbps'],handoff_priced=r['handoff_priced']))
    return output,dict(status='projected stage/task counts agree',prices=str(prices),trace=str(trace))

def collect(root,out):
    """Match each trace to its fixed build before fitting any correction."""
    path=root/'builds_b.json';records=json.loads(path.read_text()) if path.exists() else []
    table={(r['cell'],r['label']):r for r in records};rows=[];joins=[];stage_rows=[]
    for trace in sorted({p.parent for p in (root/'raw').rglob('slots.tsv')}):
        match=re.search(r'(llama|qwen3)_B(1|16)(?!\d)',str(trace))
        cell=match.group(0) if match else None
        variant=next((r for r in records if r['label'].endswith(('-v2','-pages')) and
                      r['cell']==cell and r['label'] in str(trace)),None)
        if not variant:continue
        base=table.get((variant['cell'],variant['label'].rsplit('-',1)[0]))
        if not base:continue
        exports=sorted(Path(base['out']).glob('plan.so*.task_prices.tsv'))
        matched=None
        for prices in exports:
            candidates,record=compare(prices,trace,base['so']+'.plan.json')
            record.update(cell=base['cell'],label=base['label']);joins.append(record)
            if candidates:
                for r in candidates:r.update(cell=base['cell'],label=base['label'],trace=str(trace))
                rows+=candidates;matched=prices;break
        if matched:
            predicted=Path(str(matched).removesuffix('.task_prices.tsv')+'.flow_spaces.tsv')
            if predicted.exists():
                for r in read(predicted):
                    stage_rows.append(dict(cell=base['cell'],label=base['label'],stage=int(r['stage']),
                        predicted_active_span_ns=float(r['last_end'])-float(r['first_start']),
                        source=str(predicted),scope='active-task span; compare with T4 inter-barrier span'))
    corrections=defaultdict(list);geometry=defaultdict(list)
    for r in rows:
        corrections[r['category']].append((r['dram_bytes'],r['residual_ns']))
        geometry[(r['category'],r['geometry'])].append((r['dram_bytes'],r['residual_ns']))
    fitted={kind:fit(points) for kind,points in corrections.items()}
    write(out/'T12.tsv',rows);write(out/'T12_joins.tsv',joins)
    write(out/'T12_fits.tsv',[dict(category=k,**v) for k,v in fitted.items()])
    write(out/'T12_geometry.tsv',[dict(category=k[0],geometry=k[1],**fit(v)) for k,v in geometry.items()])
    write(out/'T12_stage_predictions.tsv',stage_rows)
    plans={Path(r['so']) for r in records if not r.get('exit_code') and r['phase']=='decode' and not r['label'].endswith(('-trace','-v2','-pages'))}
    # Include prefill and decode shortlists. Source paths retain their
    # original per-candidate flow exports; unavailable exports stay explicit.
    repo=root.parents[2]
    for folder in list((repo/'runs').glob('r12c-*'))+list((repo/'runs').glob('r13-*')):
        file=folder/'plans.json'
        if not file.exists():continue
        for pair in json.loads(file.read_text()).values():
            if isinstance(pair,dict):
                for phase in ('prefill','decode'):
                    if phase in pair:plans.add(Path(pair[phase]))
                for c in pair.get('decode_pg_choice',{}).get('candidates',[]):plans.add(Path(c['library']))
    ranking=[]
    for plan in sorted(plans):ranking+=rerank(plan,fitted)
    write(out/'T12_reranked.tsv',ranking)
    (out/'T12_scope.json').write_text(json.dumps(dict(default_model_changed=False,
        component_scope='task run interval; excludes dependency waits and event publication',
        geometry_source='runtime GEMM order and the exact traced binary manifest',
        rerank_scope='report-only critical-chain residual inference, not a new solver fit'),indent=2)+'\n')

def rerank(plan,corrections):
    top=Path(str(plan)+'.top3.tsv');observations=Path(str(plan)+'.top3_measured.tsv')
    if not top.exists() or not observations.exists():return []
    scores={int(r['rank']):r for r in read(top)}
    measured={int(r['rank']):float(r['mean_ms'])*1e6 for r in read(observations) if r.get('mean_ms') and r.get('status')!='rejected'}
    adjusted={};rows=[]
    for rank,r in scores.items():
        # A report-only critical-chain correction. It is explicitly an
        # inference; it does not rerun the scheduler or refit the target.
        source=Path(r.get('source',''));prefix=str(source).removesuffix('.cu')
        chain=Path(prefix+'.flow_chain.tsv');parts=Path(prefix+'.flow_parts.tsv');delta=0;used=0
        byte_by_stage=defaultdict(list)
        if parts.exists():
            for part in read(parts):byte_by_stage[int(part['stage'])].append(float(part['dram_bytes_per_task']))
        if chain.exists():
            for link in read(chain):
                c=corrections.get(link['category'])
                if c:
                    bytes_=byte_by_stage.get(int(link['stage']),[])
                    byte_value=statistics.mean(bytes_) if bytes_ else 0
                    delta+=c['fixed_ns_bias']+(c['byte_ns_bias'] or 0)*byte_value;used+=1
        adjusted[rank]=float(r['predicted_ns'])+delta
        rows.append(dict(plan=str(plan),rank=rank,old_predicted_ns=float(r['predicted_ns']),
            corrected_ns=adjusted[rank],measured_ns=measured.get(rank),corrected_links=used,
            method='inferred fixed and byte residual on exported critical chain; equal-weight price-piece byte mean when task mapping is unavailable'))
    correlation=tau(adjusted,measured)
    for row in rows:row['corrected_tau']=correlation
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--pair',nargs=2,action='append',default=[],metavar=('PRICES','TRACE'))
    p.add_argument('--plan',type=Path,action='append',default=[]);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=[];joins=[]
    for price,trace in a.pair:
        output,join=compare(Path(price),Path(trace));rows+=output;joins.append(join)
    categories=defaultdict(list)
    for r in rows:categories[r['category']].append((r['dram_bytes'],r['residual_ns']))
    corrections={kind:fit(points) for kind,points in categories.items()}
    write(a.out/'task_errors.tsv',rows);write(a.out/'joins.tsv',joins)
    write(a.out/'fits.tsv',[dict(category=k,**v) for k,v in corrections.items()])
    ranked=[]
    for plan in a.plan:ranked+=rerank(plan,corrections)
    write(a.out/'reranked.tsv',ranked)
    (a.out/'model_error.json').write_text(json.dumps(dict(corrections=corrections,
        default_model_changed=False,scope='run interval excludes task waits and publication; bandwidth/handoff pricing retained',
        unidentified_byte_terms=[k for k,v in corrections.items() if not v['identifiable']]),indent=2)+'\n')
    print('CM-1 matched tasks '+str(len(rows)))
if __name__=='__main__':main()
