#!/usr/bin/env python3
"""Fit diagnostic preissue costs from matched native stage-only controls.

The fit does not enable measured solve or satisfy E-3. Its held-out ranking,
other workloads and L2-only overlap remain separate requirements.
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.optimize import least_squares
from sm90d1_common import ROOT, read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--overhead', type=Path, required=True)
    p.add_argument('--baseline-tag', required=True)
    p.add_argument('--tags', nargs='+', required=True)
    p.add_argument('--probe', type=Path, default=ROOT / 'build/sm90r2_price_probe')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args();a.out = a.out.resolve()
    if a.out.exists() or not a.out.is_relative_to(ROOT / 'runs/sm90r2'):
        raise ValueError('choose a new diagnostic evidence directory')
    target = read(a.target);cal = target['sm90_dense_decode']
    if cal['fit_status'] != 'initial_fitted':
        raise ValueError('start with an immutable component/stage fit')
    config = read(a.config);model, batch = config['model'], config['batch']
    hf = read(ROOT / f'configs/serving/sm90d1/inputs/{model}/config.json')
    tags = [a.baseline_tag, *a.tags]
    if len(set(tags)) != len(tags):
        raise ValueError('duplicate calibration arm')
    coordinates, artifacts, workload_identity = [], [], None
    for tag in tags:
        binary = ROOT / f'runs/sm90r2/build/{tag}/{model}-B{batch}/decode.so'
        trace = ROOT / f'runs/sm90r2/build/{tag}-stage/{model}-B{batch}/decode.so'
        identity, instrumented = read(str(binary)+'.identity.json'), read(str(trace)+'.identity.json')
        for artifact, ident in ((binary, identity), (trace, instrumented)):
            if sha(artifact) != ident['binary_sha256'] or any(
                    ident['source']['source_files'].get(k) != v for k,v in cal['native_headers'].items()):
                raise ValueError('overlap observations differ from calibrated device sources')
        if identity['trace'] or not instrumented['trace'] or sha(str(binary)+'.cu') != sha(str(trace)+'.cu'):
            raise ValueError('stage-only arm must use exactly the uninstrumented program')
        plan = read(str(binary)+'.plan.json')
        signature=[plan[k] for k in ('export_sha256','runtime_batch','capacity')]
        signature.append([(g['family'],g['n'],g['k']) for g in plan['gemms']])
        if workload_identity is not None and signature != workload_identity:
            raise ValueError('overlap arms mix model workloads')
        workload_identity=signature;coordinates.append(plan['native_dense'])
        artifacts.append(dict(tag=tag,binary_sha256=sha(binary),trace_sha256=sha(trace)))
    if any(g['preissue_smem_bytes'] or g['prefetch_l2_bytes'] for g in
           [*coordinates[0]['gemms'].values(), coordinates[0]['attention']]):
        raise ValueError('baseline must have no incoming transfer')
    shapes = {g['family']:[g['n'],g['k']] for g in plan['gemms']}
    work = dict(shapes=[shapes[k] for k in ('qkv','o','gate_up','down','lm_head')],
        batch=batch,layers=hf['num_hidden_layers'],
        head_dim=hf.get('head_dim',hf['hidden_size']//hf['num_attention_heads']),
        queries=hf['num_attention_heads']//hf['num_key_value_heads'],qk_norm=hf['model_type']=='qwen3')
    ledger = {(r['tag'].removesuffix('-stage'),r['kind']):r for r in csv.DictReader(a.ledger.open())
              if r['model']==model and int(r['batch'])==batch}
    overhead = {r['tag'].removesuffix('-stage'):float(r['overhead_ms'])*1e6/(5*work['layers']+3)
                for r in csv.DictReader(a.overhead.open()) if r['model']==model and int(r['batch'])==batch}
    def observed(tag, kind):
        row = ledger[tag,kind]
        if abs(float(row['closure_max_fraction'])) > .03:
            raise ValueError('stage ledger does not close')
        return float(row['class_total_ns'])/float(row['instances_per_step'])-overhead[tag]
    a.out.mkdir(parents=True);probe = a.out/'price-probe';shutil.copy2(a.probe,probe)
    write(a.out/'workload.json',work);write(a.out/'coordinates.json',coordinates)
    def predict(value, name):
        path = a.out/(name+'.target.json');write(path,value)
        command = [str(probe),str(path),str(a.out/'workload.json'),str(a.out/'coordinates.json')]
        output = json.loads(subprocess.check_output(command,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='')))
        write(a.out/(name+'.predictions.json'),output)
        return [r['past575']['stages'] for r in output['predictions']]
    for fit in cal['body_fits'].values():
        for name in ('preissue_efficiency','tail_efficiency','l2_efficiency','issue_ns'):
            if name in fit['coefficients']:fit['coefficients'][name]=0
    baseline = predict(target,'no-overlap')
    keys = sorted({s['key'] for stages in baseline[1:] for s in stages
                   if s['preissue_service_ns'] or s['late_service_ns']})
    issue = {}
    for key in keys:
        unit = copy.deepcopy(target);unit['sm90_dense_decode']['body_fits'][key]['coefficients']['issue_ns']=1000
        issue[key] = predict(unit,'issue-'+key)
    rows = []
    for index,tag in enumerate(tags[1:],1):
        for stage_index,s in enumerate(baseline[index]):
            before = baseline[0][stage_index]
            delta = observed(tag,s['kind'])-observed(a.baseline_tag,s['kind'])
            rows.append(dict(tag=tag,kind=s['kind'],key=s['key'],observed_delta_ns=delta,
                base_delta_ns=s['stage_ns']-before['stage_ns'],limit=s['overlap_limit_ns'],
                early=s['preissue_service_ns'],late=s['late_service_ns'],
                issue=[issue[k][index][stage_index]['issue_ns'] for k in keys]))
    y = np.array([r['observed_delta_ns'] for r in rows])
    def deltas(x):
        values=[]
        for row in rows:
            cost=sum(row['issue'][i]*x[3*i+2] for i in range(len(keys)))
            if row['key'] in keys:
                i=keys.index(row['key']);cost-=min(row['limit'],row['early']*x[3*i]+row['late']*x[3*i+1])
            values.append(row['base_delta_ns']+cost)
        return np.array(values)
    prior = np.tile([.05,.05,.3],len(keys))
    optimum = least_squares(lambda x:np.r_[(deltas(x)-y)/1000,.05*(x-prior)],prior,
        bounds=(np.zeros(len(prior)),np.tile([1,1,5],len(keys))),loss='soft_l1')
    for i,key in enumerate(keys):
        fit=cal['body_fits'][key];fit['coefficients'].update(preissue_efficiency=float(optimum.x[3*i]),
            tail_efficiency=float(optimum.x[3*i+1]),issue_ns=float(optimum.x[3*i+2]*1000))
        fit['overlap_evidence']=dict(evidence='inferred',model=model,batch=batch,tags=a.tags,
            method='bounded stage-delta fit; uniform instrumentation correction; training only')
    cal['fit_status']='initial_overlap_fitted'
    cal['pending']=['independent class/step/ranking validation on seven cells',
                    'L2-only overlap calibration','E-4 measured selection']
    write(a.out/'target.json',target)
    fitted = predict(target,'fitted')
    for row,value in zip(rows,deltas(optimum.x)):
        index=tags.index(row['tag']);stage=next(s for s in fitted[index] if s['kind']==row['kind'])
        before=next(s for s in fitted[0] if s['kind']==row['kind'])
        if abs(stage['stage_ns']-before['stage_ns']-value)>.001:
            raise ValueError('Python fit and native pricing disagree')
        row['predicted_delta_ns']=float(value);row['residual_ns']=float(value-row['observed_delta_ns'])
        for key in ('limit','early','late','issue'):del row[key]
    with (a.out/'training.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(a.out/'receipt.json',dict(evidence='diagnostic calibration; not independent E-3 validation',
        artifacts=artifacts,source_target_sha256=sha(a.target),target_sha256=sha(a.out/'target.json'),
        ledger_sha256=sha(a.ledger),overhead_sha256=sha(a.overhead),probe_sha256=sha(probe),
        script_sha256=sha(__file__),training_rows=len(rows),fitted_keys=keys))
    print(json.dumps(dict(training_rows=len(rows),fitted_keys=keys,fit_status=cal['fit_status'])))


if __name__=='__main__':main()
