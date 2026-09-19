#!/usr/bin/env python3
"""B0: four-segment task-phase partition and the FORK7 line.

FORK6 quoted its wait share against the critical-path GEMM mainloops. That
denominator excluded every SIMT body and every non-mainloop part of a GEMM, so
it could not say what a pipelining change would be worth for the pipeline. Here
the denominator is the whole critical path -- every task's `run` interval, all
kinds -- and the numerator is the exposed wait measured inside it.

Two kinds of exposed wait are summed:
  * GEMM: the first-operand wait plus `operand_wait_cycles`, the existing
    cp.async wait and CTA rendezvous inside the instrumented K-loop (FORK6).
  * SIMT: `simt_wait_cycles`, thread zero's time inside the barriers the body
    already executed (B0).

Both are lower bounds, for different reasons, and the summary records them as
such: thread zero waits for nobody when it arrives last, and a SIMT body with
no barrier at all reports zero although its loads still stall. `body_share`
per kind is published alongside, because a kind's exposed wait cannot exceed
the share of the path it occupies.
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

KIND={0:'gemm',1:'rmsnorm',2:'rope',3:'kvappend',4:'elementwise',5:'attention',
      6:'gemm_combine',7:'gemm_add',8:'gemm_rmsnorm',9:'rope_kvappend',10:'add',
      11:'embedding',12:'qknorm'}

def table(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))

def write(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
        w.writeheader();w.writerows(rows)

def partition(dump,source):
    """Per-slot four-segment partition; closure against `run` is checked."""
    _,nodes=trace.analyze_phases(dump,source)
    raw={int(r['slot']):r for r in table(dump/'phases.tsv')}
    for r in nodes:
        p=raw[r['slot']]
        gemm=r['kind']==0 and int(p['k_iterations'])>0
        inner=int(p['operand_wait_cycles']) if gemm else int(p['simt_wait_cycles'])
        if inner>r['mainloop_cycles']:
            raise ValueError(f'inner wait exceeds mainloop at {dump}: {r["slot"]}')
        r.update(kind_name=KIND.get(r['kind'],str(r['kind'])),
                 simt_barriers=int(p['simt_barriers']),
                 seg_setup=r['setup_cycles'],
                 seg_wait=r['load_wait_cycles']+inner,
                 seg_compute=r['mainloop_cycles']-inner,
                 seg_epilogue=r['epilogue_cycles'])
        if r['seg_setup']+r['seg_wait']+r['seg_compute']+r['seg_epilogue']!=r['run_cycles']:
            raise ValueError(f'segment closure at {dump}: {r["slot"]}')
    return nodes

def shares(nodes):
    """Aggregate over a node set: overall shares plus one row per task kind."""
    total=sum(r['run_cycles'] for r in nodes)
    if not total: raise ValueError('empty denominator')
    out=dict(nodes=len(nodes),run_cycles=total)
    for seg in ('setup','wait','compute','epilogue'):
        out[seg+'_share']=sum(r['seg_'+seg] for r in nodes)/total
    for label,pick in (('gemm',lambda r:r['kind']==0),('simt',lambda r:r['kind']!=0)):
        sel=[r for r in nodes if pick(r)]
        out[label+'_wait_share']=sum(r['seg_wait'] for r in sel)/total
        out[label+'_body_share']=sum(r['run_cycles'] for r in sel)/total
        # The head of a body -- operand resolution plus the first-load wait --
        # is what a predecessor's epilogue could overlap. It is reported
        # separately from the exposed wait because it is not idle time: it is
        # work that runs too late, which is a different claim and a different
        # fix. B1 is sized against this, not against `*_wait_share`.
        out[label+'_head_share']=sum(r['seg_setup']+r['seg_wait'] for r in sel)/total
    out['exposed_wait_share']=out['gemm_wait_share']+out['simt_wait_share']
    return out

def per_kind(nodes,cell):
    total=sum(r['run_cycles'] for r in nodes)
    rows=[]
    for name in sorted({r['kind_name'] for r in nodes}):
        sel=[r for r in nodes if r['kind_name']==name]
        body=sum(r['run_cycles'] for r in sel)
        rows.append(dict(cell=cell,kind=name,nodes=len(sel),
            barriers=sum(r['simt_barriers'] for r in sel),
            body_share_of_path=body/total,
            **{seg+'_share':sum(r['seg_'+seg] for r in sel)/body
               for seg in ('setup','wait','compute','epilogue')}))
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw',type=Path,default=HERE/'raw')
    ap.add_argument('--threshold',type=float,default=0.15)
    a=ap.parse_args()
    results=[];segments=[];cells=[]
    for cell in sorted(a.raw.glob('*_s*')):
        model,seq=cell.name.rsplit('_s',1)
        ss=json.loads((cell/'specs.json').read_text())
        for arm,s in ss.items():
            dumps=sorted((cell/'phase').glob(f'{arm}_r*/dump'))
            if not dumps:continue
            rounds=[]
            for dump in dumps:
                nodes=partition(dump,Path(s['source']))
                write(dump.parent/'node_segments.tsv',nodes)
                cp=[r for r in nodes if r['on_cp']]
                row=dict(model=model,seq=int(seq),arm=arm,key=s['key'],
                         round=int(dump.parent.name.rsplit('_r',1)[1]),
                         dump=str(dump.relative_to(REPO)))
                for label,sel in (('all',nodes),('cp',cp)):
                    row.update({f'{label}_{k}':v for k,v in shares(sel).items()})
                results.append(row);rounds.append(row)
                if row['round']==0: segments+=per_kind(cp,f'{model}_s{seq}')
            # Per cell, the median over its fresh processes; then the median
            # over cells, which is FORK6's aggregation.
            cells.append(dict(model=model,seq=int(seq),arm=arm,key=s['key'],
                rounds=len(rounds),
                **{k:statistics.median(r[k] for r in rounds)
                   for k in rounds[0] if k.startswith(('cp_','all_'))}))
    if not results:raise ValueError('no raw dumps')
    write(a.raw/'analysis.tsv',results);write(a.raw/'segments.tsv',segments)
    write(a.raw/'cells.tsv',cells)
    ref=[r for r in cells if r['model'] in ('gqa2','mha4') and r['arm']=='selected']
    if len(ref)!=4:raise ValueError(f'FORK7 needs four reference cells, got {len(ref)}')
    med=lambda k:statistics.median(r[k] for r in ref)
    gemm,simt=med('cp_gemm_wait_share'),med('cp_simt_wait_share')
    # Quoted as the sum so the three numbers are consistent; with four cells the
    # medians need not come from the same pair, so the direct median is checked.
    whole=gemm+simt
    direct=med('cp_exposed_wait_share')
    if abs(whole-direct)>5e-4:
        raise ValueError(f'median decomposition disagrees: {whole} vs {direct}')
    line=(f'FORK7 rule={1 if whole>=a.threshold else 2} '
          f'whole_pipeline_exposed_wait_share={whole:.3f} '
          f'gemm_share={gemm:.3f} simt_share={simt:.3f} cells=4')
    print(line)
    for r in ref:
        print(f"  {r['model']}_s{r['seq']:<4d} {r['key']:22s} rounds={r['rounds']} "
              f"cp_wait={r['cp_exposed_wait_share']:.4f} "
              f"gemm={r['cp_gemm_wait_share']:.4f} simt={r['cp_simt_wait_share']:.4f} "
              f"simt_body={r['cp_simt_body_share']:.4f} "
              f"simt_head={r['cp_simt_head_share']:.4f}")
    print(f"  median simt_head_share={med('cp_simt_head_share'):.3f} "
          f"gemm_head_share={med('cp_gemm_head_share'):.3f}")
    (a.raw/'fork7.txt').write_text(line+'\n')

if __name__=='__main__':main()
