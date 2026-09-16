#!/usr/bin/env python3
"""Fit publication and residual visibility separately; retain every fit input."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import sys
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
spec=importlib.util.spec_from_file_location('r4_trace',REPO/'docs/experiments/TRACE_V2/analyze.py')
trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)

def table(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def timing(p):
    line=re.findall(r'^E2E_TIME (.*)$',p.read_text(),re.M)
    assert len(line)==1
    return {k:float(v) for k,v in re.findall(r'(\w+)=([\d.eE+-]+)',line[0])}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fence-root',type=Path,default=REPO/'docs/experiments/FENCE/raw/paired')
    ap.add_argument('--trace-root',type=Path)
    ap.add_argument('--hop',type=Path,default=REPO/'docs/experiments/SIMULATOR/hop_ns.tsv')
    ap.add_argument('--out',type=Path,default=HERE)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for m in ('gqa2','mha4'):
        for seq in (4,128):
            for p,name in [(0,'legacy_grid_stride'),(5,'rotate')]:
                dump=REPO/f'docs/experiments/SYNC_V3/targets_raw/dump/{m}_s{seq}_{name}'
                if args.trace_root:dump=args.trace_root/f'{m}_s{seq}_p{p}'/'dump'
                source=(Path(os.environ['R5_INPUT_ROOT'])/'src'/f'{m}.cu') if os.getenv('R5_INPUT_ROOT') else REPO/f'docs/experiments/PLAN_CONTRACT/legacy_identity/plan/{m}.cu'
                # A phase-only dump deliberately contains zero event timestamps.
                # It cannot identify publication/wait residuals.
                _,_,_,raw_events=trace.load(dump)
                if not any(e['publish_ns'] for e in raw_events):
                    raise ValueError(f'publication calibration requires TRACE_V2 event timestamps: {dump}')
                a=trace.analyze(dump,source,1)
                _,slots,_,events=trace.load(dump)
                active={e['stage'] for e in events if e['fanin']>0}
                by_key={(r['stage'],r['logical_task']):r for r in slots}
                path=[by_key[tuple(map(int,x.split(':')))] for x in a['cp_realized_path'].split(',')]
                perworker={w:sum(s['stage'] in active for s in slots if s['worker']==w) for w in {s['worker'] for s in slots}}
                pub=max(perworker.values())
                cp_pub=sum(s['stage'] in active for s in path)
                cp_wait=sum(s['wait_count']>0 for s in path)
                hops=sum(x['worker']!=y['worker'] for x,y in zip(path,path[1:]))
                f=args.fence_root/f'{m}_s{seq}_p{p}'
                notify=[];fence=[]
                for i in range(25):
                    t={arm:timing(f/arm/f'r{i}.log')['l2_ms'] for arm in ('full','nofence','nowait','neither')}
                    notify.append((t['nowait']-t['neither'])*1e6)
                    fence.append((t['full']-t['nofence'])*1e6)
                rows.append(dict(model=m,seq=seq,placement=p,dump=str(dump),fence_logs=str(f),
                    notify_delta_ns=statistics.median(notify),fence_delta_ns=statistics.median(fence),
                    max_worker_publishers=pub,realized_publishers=cp_pub,realized_waiters=cp_wait,realized_cross_hops=hops,
                    realized_ns=a['cp_reconstructed_ns'],task_ns=a['cp_split_task_ns'],
                    trace_publish_ns=a['cp_split_publish_ns'],publication_estimate_ns=statistics.median(notify)/pub))
    publication=statistics.median(r['publication_estimate_ns'] for r in rows)
    # Residual is retained signed before projection onto a nonnegative physical delay.
    for r in rows:
        r['residual_visibility_ns']=(r['realized_ns']-r['task_ns']-publication*r['realized_publishers'])/max(1,r['realized_cross_hops'])
    # The primitive hop remains an independently measured R2 transfer term.
    # A queue-critical path may have zero cross edges yet many global waits;
    # attributing all its residual to a per-hop constant is unidentifiable.
    hop={line.split('\t')[0]:float(line.split('\t')[1])
         for line in args.hop.read_text().splitlines()
         if line.startswith(('c0\t','c1\t','c2\t'))}
    visibility=hop['c0']
    for r in rows:
        r['consumer_wait_estimate_ns']=(r['realized_ns']-r['task_ns']-publication*r['realized_publishers']-visibility*r['realized_cross_hops'])/max(1,r['realized_waiters'])
    residual=statistics.median(r['consumer_wait_estimate_ns'] for r in rows)
    with (args.out/'publication_inputs.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
    result=dict(publication_ns=max(0,publication),visibility_ns=visibility,consumer_wait_ns=max(0,residual),unconstrained_publication_ns=publication,
        unconstrained_consumer_wait_ns=residual,method='pooled median: unsafe nowait-neither per busiest publishing queue, then corrected causal-path residual per waiting consumer, retaining the independently calibrated hop',
        limitation='Marginal probes include schedule/resource changes; these are model parameters, not identified instruction latencies. Calibration provenance is identified by publication_inputs.tsv; historical inputs do not become fresh performance measurements.',
        publication_requirement='Use actual runtime event flags; minimal cross-consumer DAG mask is a separate idealized option. Stage-wide fine flags can publish tasks with only local direct consumers.')
    (args.out/'publication.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
