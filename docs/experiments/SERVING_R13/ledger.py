#!/usr/bin/env python3
"""Stage and boundary accounting from runtime tables, never inferred elision."""
import argparse,csv,json,statistics
from pathlib import Path
def read(p):return list(csv.DictReader(Path(p).open(),delimiter='\t'))
def write(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['status'],delimiter='\t',lineterminator='\n');w.writeheader()
        w.writerows(rows or [dict(status='not_collected')])
def stages(folder,batch,ceilings):
    folder=Path(folder);meta={int(r['stage']):r for r in read(folder/'runtime_stages.tsv')}
    launches={}
    for r in read(folder/'stage_trace.tsv'):
        r={k:int(v) for k,v in r.items()};launches.setdefault(r['iteration'],{}).setdefault(r['stage'],[]).append(r)
    output=[]
    for iteration,by_stage in launches.items():
        previous=None
        for s,workers in sorted(by_stage.items()):
            descriptor=meta[s];past=workers[0]['past']
            finish=max(w['t_release'] for w in workers)
            # Stage zero starts at its own earliest begin; subsequent stages
            # use the preceding non-elided stage's latest barrier release.
            begin=previous if previous is not None else min(w['t_begin'] for w in workers)
            elapsed=finish-begin;previous=finish
            byte_count=int(descriptor['weight_bytes'])
            # kFusedAttention is decoded using the generated TaskKind enum,
            # rather than a hardware/model-specific stage list.
            if descriptor['kind_name']=='kFusedAttention':
                byte_count=2*batch*int(descriptor['extent'])*(past+1)*int(descriptor['width'])*2
            endings=[w['t_tasks_end'] for w in workers if w['tasks']]
            row=dict(iteration=iteration,past=past,stage=s,kind=descriptor['kind_name'],name=descriptor['name'],
                     duration_ns=elapsed,bytes=byte_count,gbps=byte_count/elapsed if elapsed else None,
                     tail_ns=max(endings)-statistics.median(endings) if endings else 0,
                     tasks=sum(w['tasks'] for w in workers))
            for label,gbps in ceilings.items():
                row['floor_ns_'+label]=byte_count/gbps;row['excess_ns_'+label]=elapsed-byte_count/gbps
            output.append(row)
    return output
def steps(folder):
    grouped={}
    for r in read(Path(folder)/'step_trace.tsv'):
        r={k:int(v) for k,v in r.items()};grouped.setdefault(r['iteration'],[]).append(r)
    output=[];previous=None
    for iteration,workers in sorted(grouped.items()):
        first=[w['first_task'] for w in workers if w['first_task']]
        last=[w['last_task'] for w in workers if w['last_task']]
        if not first or not last:continue
        begin=min(first);end=max(last)
        span=max(w['kernel_end'] for w in workers)-min(w['kernel_begin'] for w in workers)
        row=dict(iteration=iteration,past=workers[0]['past'],span_ns=span,
                 boundary_gap_ns=begin-previous if previous is not None else None)
        for key in ('token_lag','kv_lag','last_barrier_wait'):
            row[key+'_mean_ns']=statistics.mean(w[key] for w in workers)
            row[key+'_mean_fraction']=row[key+'_mean_ns']/span if span else None
        output.append(row);previous=end
    return output
def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--batch',type=int,required=True);p.add_argument('--ceiling',type=float,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if (a.trace/'stage_trace.tsv').exists():
        raw=stages(a.trace,a.batch,{'884_5':884.5,'981_6':981.6,'measured':a.ceiling})
        write(a.out/'stages.tsv',raw)
        groups={}
        for r in raw:groups.setdefault((r['past'],r['stage']),[]).append(r)
        medians=[]
        for _,records in groups.items():
            r=dict(records[0]);r.pop('iteration')
            for k,v in r.items():
                if isinstance(v,(int,float)) and k not in ('past','stage','bytes','tasks'):
                    r[k]=statistics.median(x[k] for x in records)
            medians.append(r)
        write(a.out/'stage_medians.tsv',medians)
        write(a.out/'top_excess.tsv',sorted(medians,key=lambda r:r['excess_ns_measured'],reverse=True)[:10])
    if (a.trace/'step_trace.tsv').exists():write(a.out/'steps.tsv',steps(a.trace))
    print('ledger written')
if __name__=='__main__':main()
