#!/usr/bin/env python3
"""Report tables from raw processes and corrected task-DAG traces."""
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path
from verify import trace,chosen,paired,timing,table,source
from search import write,REPO,HERE
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['gqa2','mha4','real']);ap.add_argument('--out',type=Path,default=HERE);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=[];ranks=[]
 for m in a.models:
  for s in (4,128):
   c=HERE/f'raw/{m}_s{s}';arm=chosen(c);top=table(c/'top3.tsv')[int(arm[-1])-1];ratio,lo,hi=paired(c/'measure',arm,'control')
   for tag,src in [('control',source(m)),(arm,c/'plans'/top['key']/(top['placement']+'.cu'))]:
    dump=c/'trace'/tag/'dump';v=trace.analyze(dump,src,1);_,slots,_,_=trace.load(dump);path={tuple(map(int,x.split(':'))) for x in v['cp_corrected_path'].split(',')}
    times=[r['run_end']-r['run_begin'] for r in slots if (r['stage'],r['logical_task']) in path]
    logs=[timing(c/'measure'/tag/f'r{i}.log') for i in range(25)];l2=statistics.median(r['l2_ms'] for r in logs);floor=max(v['cp_corrected_ns'],v['queue_lb_ns'])
    row=dict(model=m,seq=s,arm=tag,config=top['key'] if tag!='control' else '128x128x16s3k1_k1_natural',placement=top['placement'] if tag!='control' else 'rotate',l2_ms=l2,l1_ms=statistics.median(r['l1_ms'] for r in logs),l2_l1=statistics.median(r['l2_ms']/r['l1_ms'] for r in logs),ratio=ratio if tag!='control' else 1.,ci_low=lo if tag!='control' else 1.,ci_high=hi if tag!='control' else 1.,cp_ns=v['cp_corrected_ns'],nodes=v['cp_corrected_nodes'],task_mean_ns=statistics.mean(times),task_p50_ns=statistics.median(times),task_p90_ns=trace.percentile(times,.9),task_max_ns=max(times),queue_lb_ns=v['queue_lb_ns'],queue_cp=v['queue_lb_ns']/v['cp_corrected_ns'],busy_fraction=1-v['idle_fraction_of_worker_time'],floor_ns=floor,l2_floor=l2*1e6/floor,trace_l2_ms=v['measured_l2_ms'],dump=str(dump.relative_to(REPO)))
    control=[timing(c/'measure'/'control'/f'r{i}.log') for i in range(25)]
    row['control_l1_ms']=statistics.median(r['l1_ms'] for r in control)
    row['l2_control_l1']=statistics.median(x['l2_ms']/y['l1_ms'] for x,y in zip(logs,control))
    rows.append(row);print(json.dumps(row),flush=True)
   med={t:statistics.median(timing(c/'measure'/t/f'r{i}.log')['l2_ms'] for i in range(25)) for t in ('top1','top2','top3')}
   for i,t in enumerate(('top1','top2','top3')):
    r=table(c/'top3.tsv')[i];ranks.append(dict(model=m,seq=s,arm=t,key=r['key'],placement=r['placement'],predicted_ns=r['makespan_ns'],measured_ms=med[t],measured_rank=sorted(med,key=med.get).index(t)+1))
 write(a.out/'comparisons.tsv',rows);write(a.out/'measured_ranking.tsv',ranks)
 phases=[];operands=[]
 for m in ('gqa2','mha4','real'):
  for s in (4,128):
   for p in (0,5):
    d=REPO/f'docs/experiments/PHASE/raw/trace/{m}_s{s}_p{p}/dump';v,nodes=trace.analyze_phases(d,source(m))
    for group in ('cp','all'):
     row=dict(model=m,seq=s,placement=p,group=group,total_ns=v[f'phase_{group}_run_ns'])
     for phase in ('setup','load_wait','mainloop','epilogue'):
      row[phase+'_ns']=v[f'phase_{group}_{phase}_ns'];row[phase+'_share']=v[f'phase_{group}_{phase}_share_ns']
     phases.append(row)
    groups=defaultdict(list)
    for n in nodes:
     if n['kind']==0:groups[tuple(n[k] for k in ('tile_m','tile_n','tile_k','split_k','operand_bytes'))].append(n)
    for shape,ns in groups.items():operands.append(dict(model=m,seq=s,placement=p,**dict(zip(('tile_m','tile_n','tile_k','split_k','operand_bytes'),shape)),nodes=len(ns),load_wait_mean_ns=statistics.mean(n['load_wait_ns'] for n in ns),load_wait_p50_ns=statistics.median(n['load_wait_ns'] for n in ns),load_wait_p90_ns=trace.percentile([n['load_wait_ns'] for n in ns],.9)))
 write(a.out/'phases.tsv',phases);write(a.out/'operand_load.tsv',operands)
if __name__=='__main__':main()
