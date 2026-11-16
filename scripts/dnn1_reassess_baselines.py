#!/usr/bin/env python3
"""Reassess existing guarded receipts under the explicit P-CORR-v2 amendment.

This does not run models or replace measurements. Old strict rows remain input
artifacts; every admitted process must independently pass the original gate.
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import dnn1_baselines as baseline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    rows=list(csv.DictReader(args.input.open()))
    fields=list(rows[0])+['correctness_policy','strict_v1_status','correctness_spread']
    changed=[];evidence=[]
    for row in rows:
        row.update(correctness_policy='not_reassessed',strict_v1_status=row['status'],correctness_spread='')
        if row['system']=='tvm':continue
        path=baseline.ROOT/row['source'];session=baseline.read(path)
        guard=baseline.read(path.parents[1]/'guard_result.json')
        accepted=guard['code']==0 and session['status']=='complete'
        cell=baseline.read(session['cell']);cell['correctness_policy']='five-independent-gates-v2'
        jobs=[j for j in session['logical_jobs'] if j['arm']['arm']==row['arm']]
        new=baseline.aggregate_arm(cell,session,row['arm'],jobs,accepted)
        metrics=[];pids=[];inputs=set();sources=[]
        for job in jobs:
            if job['status']!='pass':continue
            source=Path(job['final']['measurement']);m=baseline.read(source)
            metrics.append(m['correctness']);pids.append(m['pid']);inputs.add(m['configuration']['inputs_sha256'])
            sources.append(dict(path=str(source),sha256=baseline.sha(source)))
        if new['status']=='pass':
            assert len(set(pids))==5 and len(inputs)==1
            spread={}
            for index in range(len(metrics[0])):
                for key in ('cosine_vs_fp32','cosine_min_vs_fp32','max_abs_vs_fp32','mse_vs_fp32','psnr_min_vs_fp32_db'):
                    if key not in metrics[0][index]:continue
                    values=[x[index][key] for x in metrics]
                    spread[f'output{index}.{key}']=dict(min=min(values),max=max(values),range=max(values)-min(values))
            row['correctness_spread']=baseline.compact(spread)
        row['correctness_policy']=cell['correctness_policy']
        if row['status']!=new['status']:
            assert row['status']=='N/A' and new['status']=='pass'
            changed.append(dict(model=row['model'],batch=row['batch'],arm=row['arm'],old=row['status'],new=new['status']))
            for key in ('status','reason','median_ms','p90_ms','process_range_ms','process_range_pct'):
                row[key]=new[key]
        evidence.append(dict(model=row['model'],batch=row['batch'],arm=row['arm'],session_sha256=baseline.sha(path),
            guard_sha256=baseline.sha(path.parents[1]/'guard_result.json'),pids=pids,input_hashes=sorted(inputs),sources=sources))
    with (args.out/'DNN1_baselines_v2.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    summary=dict(evidence='verified reassessment of existing measurements, not a new timing run',
        authorization='user 2026-11-04 permits data-driven baseline repair; explicit P-CORR-v2 amendment',
        numerical_thresholds='unchanged',policy='five independently passing processes; retain all exact metric spreads; no exact-equality requirement',
        input_sha256=baseline.sha(args.input),output_sha256=baseline.sha(args.out/'DNN1_baselines_v2.csv'),
        counts=dict(collections.Counter(r['status'] for r in rows)),changed=changed,receipts=evidence)
    baseline.write(args.out/'reassessment.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='receipts'},indent=2))


if __name__=='__main__':main()
