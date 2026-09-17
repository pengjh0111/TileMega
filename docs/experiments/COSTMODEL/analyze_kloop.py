#!/usr/bin/env python3
"""Recompute R6 loop partitions from raw phase stamps; preserve R5 partitions.

The exposed interval includes the existing cp.async wait and CTA rendezvous.
It is operand availability latency, not an estimate of pure DRAM service.
clock64 partitions are local to each task; no clocks from different SMs are
subtracted. Nanoseconds are derived with that task's measured ns/cycle ratio.
FORK6 uses GEMM mainloops on the corrected path, the scope of this probe.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('r4trace',REPO/'docs/experiments/TRACE_V2/analyze.py')
trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)
def table(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def analyze(dump,source):
    old,nodes=trace.analyze_phases(dump,source)
    raw={int(r['slot']):r for r in table(dump/'phases.tsv')}
    for r in nodes:
        p=raw[r['slot']]
        lo,hi,wait,iters=(int(p[k]) for k in ('loop_begin_cycles','loop_end_cycles','operand_wait_cycles','k_iterations'))
        valid=r['kind']==0
        if valid:
            if not iters or not int(p['first_operand_ready_cycles'])<=lo<=hi<=int(p['mainloop_end_cycles']):
                raise ValueError(f'loop outside mainloop at {dump}: {r["slot"]}')
            loop=hi-lo;fixed=r['mainloop_cycles']-loop;body=loop-wait
            if min(fixed,body,wait)<0:raise ValueError('negative loop partition')
            if body+wait+fixed!=r['mainloop_cycles']:raise ValueError('partition closure')
        else:fixed=body=wait=0
        scale=r['run_ns']/r['run_cycles']
        r.update(kloop_measured=int(valid),k_iterations=iters,kloop_body_cycles=body,
                 kloop_wait_cycles=wait,kloop_fixed_cycles=fixed,
                 kloop_body_ns=body*scale,kloop_wait_ns=wait*scale,kloop_fixed_ns=fixed*scale,
                 kloop_iteration_ns=(body+wait)*scale/iters if iters else 0.)
    result=dict(old)
    for label,rs in [('all',nodes),('cp',[r for r in nodes if r['on_cp']])]:
        measured=[r for r in rs if r['kloop_measured']]
        denom=sum(r['mainloop_cycles'] for r in measured)
        if not measured:raise ValueError('no measured GEMM mainloops')
        result[label+'_kloop_nodes']=len(measured)
        result[label+'_unmeasured_mainloop_cycles']=sum(r['mainloop_cycles'] for r in rs if not r['kloop_measured'])
        result[label+'_kloop_mainloop_cycles']=denom
        for part in ('body','wait','fixed'):
            for unit in ('cycles','ns'):
                result[f'{label}_kloop_{part}_{unit}']=sum(r[f'kloop_{part}_{unit}'] for r in measured)
            result[f'{label}_kloop_{part}_share']=sum(r[f'kloop_{part}_cycles'] for r in measured)/denom
        result[label+'_kloop_iteration_p50_ns']=statistics.median(r['kloop_iteration_ns'] for r in measured)
        result[label+'_kloop_fixed_p50_ns']=statistics.median(r['kloop_fixed_ns'] for r in measured)
    return result,nodes

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--raw',type=Path,default=HERE/'raw_kloop');a=ap.parse_args()
    results=[]
    for cell in sorted(a.raw.glob('*_s*')):
        model,seq=cell.name.rsplit('_s',1);ss=json.loads((cell/'specs.json').read_text())
        for arm,s in ss.items():
            dump=cell/'phase'/arm/'dump'
            if not dump.exists():continue
            result,nodes=analyze(dump,Path(s['source']))
            write(dump.parent/'node_kloop.tsv',nodes)
            results.append(dict(result,model=model,seq=int(seq),arm=arm,source=s['source'],dump=str(dump.relative_to(REPO) if dump.is_relative_to(REPO) else dump)))
    if not results:raise ValueError('no raw dumps')
    write(a.raw/'analysis.tsv',results)
    reference=[r for r in results if r['model'] in ('gqa2','mha4') and r['arm']=='selected']
    if len(reference)!=4:raise ValueError(f'FORK6 needs four reference cells, got {len(reference)}')
    wait=statistics.median(r['cp_kloop_wait_share'] for r in reference)
    fixed=statistics.median(r['cp_kloop_fixed_share'] for r in reference)
    line=f'FORK6 rule={1 if wait>=.25 else 2} mainloop_exposed_wait_share={wait:.3f} kloop_fixed_share={fixed:.3f} cells=4'
    print(line)
    (a.raw/'fork6.txt').write_text(line+'\n')
if __name__=='__main__':main()
