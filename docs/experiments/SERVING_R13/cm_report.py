#!/usr/bin/env python3
"""Report-only task-price errors and bounded linear corrections; no defaults mutate."""
import argparse,json,statistics
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

def compare(prices,trace):
    prediction=[{k:(v if k in ('space','category','handoff_priced') else float(v)) for k,v in r.items()} for r in read(prices)]
    actual={}
    for r in read(trace/'slots.tsv'):
        if int(r['run_begin']):actual[(int(r['stage']),int(r['logical_task']))]=int(r['run_end'])-int(r['run_begin'])
    counts=defaultdict(int)
    for r in prediction:counts[int(r['stage'])]+=1
    stages={int(r['stage']):r for r in read(trace/'runtime_stages.tsv')}
    # Refuse joins across numbering systems. The flow uses the projected
    # runtime graph; every projected stage/task count must agree with its dump.
    if set(counts)!=set(stages) or any(counts[s]!=int(stages[s]['active_tasks']) for s in stages):
        return [],dict(status='incompatible projected stage/task counts',prices=str(prices),trace=str(trace))
    output=[]
    for r in prediction:
        key=(int(r['stage']),int(r['task']))
        if key not in actual:continue  # Elided reducers are reported separately.
        measured=actual[key];predicted=r['predicted_ns']
        output.append(dict(stage=key[0],task=key[1],category=r['category'],space=r['space'],
            measured_run_ns=measured,predicted_ns=predicted,relative_error=measured/predicted-1 if predicted else None,
            fixed_ns=r['fixed_ns'],compute_ns=r['compute_ns'],dram_bytes=r['dram_bytes'],
            residual_ns=measured-predicted,pricing_dram_gbps=r['dram_gbps'],handoff_priced=r['handoff_priced']))
    return output,dict(status='projected stage/task counts agree',prices=str(prices),trace=str(trace))

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
