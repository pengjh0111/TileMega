#!/usr/bin/env python3
"""Diagnose selected geometries after search without changing the frozen fork."""
import argparse,statistics
from collections import defaultdict
from pathlib import Path
from verify import trace,chosen,table
from search import HERE,write

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--complete',action='store_true');a=ap.parse_args();rows=[];operands=[]
 for m in ('gqa2','mha4','real'):
  for s in (4,128):
   c=HERE/f'raw/{m}_s{s}'
   if not (c/'choice.json').exists():
    assert not a.complete,c
    continue
   arm=chosen(c);top=table(c/'top3.tsv')[int(arm[-1])-1];d=c/'phase'/arm/'dump'
   if not d.exists():
    assert not a.complete,d
    continue
   values,nodes=trace.analyze_phases(d,c/'plans'/top['key']/(top['placement']+'.cu'))
   for group in ('cp','all'):
    row=dict(model=m,seq=s,config=top['key'],placement=top['placement'],group=group,total_ns=values[f'phase_{group}_run_ns'])
    for phase in ('setup','load_wait','mainloop','epilogue'):
     row[phase+'_ns']=values[f'phase_{group}_{phase}_ns'];row[phase+'_share']=values[f'phase_{group}_{phase}_share_ns']
    rows.append(row)
   groups=defaultdict(list)
   for n in nodes:
    if n['kind']==0:groups[tuple(n[k] for k in ('tile_m','tile_n','tile_k','split_k','operand_bytes'))].append(n)
   for shape,ns in groups.items():operands.append(dict(model=m,seq=s,config=top['key'],**dict(zip(('tile_m','tile_n','tile_k','split_k','operand_bytes'),shape)),nodes=len(ns),load_wait_mean_ns=statistics.mean(n['load_wait_ns'] for n in ns),load_wait_p50_ns=statistics.median(n['load_wait_ns'] for n in ns),load_wait_p90_ns=trace.percentile([n['load_wait_ns'] for n in ns],.9),load_wait_mean_cycles=statistics.mean(n['load_wait_cycles'] for n in ns)))
 probes=[]
 for d in sorted((HERE/'raw/real_s4/operand_probe').glob('*')):
  values,nodes=trace.analyze_phases(d/'dump',HERE/'raw/real_s4/plans'/d.name/'eft.cu')
  gems=[n for n in nodes if n['kind']==0]
  probes.append(dict(config=d.name,nodes=len(gems),split=gems[0]['split_k'],tile_m=gems[0]['tile_m'],tile_n=gems[0]['tile_n'],tile_k=gems[0]['tile_k'],operand_bytes=gems[0]['operand_bytes'],load_mean_ns=statistics.mean(n['load_wait_ns'] for n in gems),load_p50_ns=statistics.median(n['load_wait_ns'] for n in gems),load_mean_cycles=statistics.mean(n['load_wait_cycles'] for n in gems),cp_load_share=values['phase_cp_load_wait_share_ns'],cp_total_ns=values['phase_cp_run_ns']))
 write(HERE/'selected_phases.tsv',rows);write(HERE/'selected_operand_load.tsv',operands)
 if probes:write(HERE/'operand_split_probe.tsv',probes)
 for r in rows:
  if r['group']=='cp':print(r,flush=True)
if __name__=='__main__':main()
