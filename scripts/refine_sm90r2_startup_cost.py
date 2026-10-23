#!/usr/bin/env python3
"""Recalibrate native fixed costs from immutable base and stage-control observations.

All supplied observations become training evidence. Traffic, MMA, epilogue and
issuer coefficients remain fixed; this does not promote a solver calibration.
"""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.optimize import least_squares
from sm90d1_common import ROOT, read, sha, write
from verify_sm90r2_correctness import verify_receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--base-fit', type=Path, required=True)
    p.add_argument('--class-audit', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args();a.out = a.out.resolve()
    if a.out.exists() or not a.out.is_relative_to(ROOT/'runs/sm90r2'):
        raise ValueError('retain previous evidence; choose a new runs/sm90r2 directory')
    target = read(a.target);cal = target['sm90_dense_decode']
    receipt = read(a.base_fit/'fit-receipt.json')
    if sha(a.base_fit/'target.json') != receipt['target_sha256']:
        raise ValueError('base stage fit changed')
    base_cal = read(a.base_fit/'target.json')['sm90_dense_decode']
    if base_cal['native_headers'] != cal['native_headers']:
        raise ValueError('base and stage-control observations use different native implementations')
    audit = read(a.class_audit/'receipt.json')
    if (audit['inputs']['target']['sha256'] != sha(a.target) or
            audit['inputs']['probe']['sha256'] != sha(a.probe)):
        raise ValueError('stage audit and calibration must share one frozen price model')
    if 'control' not in audit['inputs']:
        raise ValueError('use an explicit stage-control audit with its correctness dependency')
    for item in audit['inputs'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('stage-control audit input changed')
    control = read(audit['inputs']['control']['path'])
    proof = verify_receipt(Path(control['state']),control['job'],
                           Path(control['binary']),Path(control['config']))
    if proof['binary_sha256'] != audit['binary_sha256']:
        raise ValueError('stage-control correctness belongs to another artifact')
    a.out.mkdir();probe = a.out/'price-probe';shutil.copy2(a.probe,probe)
    rows,cases = [],[]
    def predict(work,coordinates,label,target_path=a.target):
        wp,cp = a.out/(label+'.workload.json'),a.out/(label+'.coordinates.json')
        write(wp,work);write(cp,coordinates)
        value = json.loads(subprocess.check_output([str(probe),str(target_path.resolve()),
            str(wp),str(cp)],text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='')))
        return value['predictions'][0]['past575']['stages']
    base_rows = list(csv.DictReader((a.base_fit/'training.csv').open()))
    for artifact in receipt['artifacts']:
        model,batch,tag = artifact['model'],artifact['batch'],artifact['tag']
        label = f'{tag}-{model}-B{batch}'
        work = read(a.base_fit/(label+'.workload.json'))
        coordinates = read(a.base_fit/(label+'.tmexec.json'))
        stages = predict(work,coordinates,label);cases.append((label,work,coordinates,stages))
        for row in base_rows:
            if (row['model'],int(row['batch']),row['tag']) != (model,batch,tag):continue
            s = next(s for s in stages if s['kind']==row['kind'])
            if abs(s['stage_ns']-float(row['prediction_ns'])) > .05:
                raise ValueError('zero-transfer base prices changed before startup refinement')
            rows.append(dict(source='base',label=label,kind=s['kind'],key=s['key'],
                observed_ns=float(row['corrected_stage_ns']),prediction_ns=s['stage_ns']))
    extra_rows = list(csv.DictReader(Path(audit['inputs']['ledger']['path']).open()))
    overhead = list(csv.DictReader(Path(audit['inputs']['overhead']['path']).open()))
    model,batch,tag = audit['model'],audit['batch'],audit['tag']
    overhead = [r for r in overhead if (r['model'],int(r['batch']),r['tag'])==(model,batch,tag+'-stage')]
    if len(overhead)!=1:raise ValueError('one matched trace overhead is required')
    work = read(a.class_audit/'workload.json');coordinates = read(a.class_audit/'coordinates.json')
    label = f'{tag}-{model}-B{batch}';stages = predict(work,coordinates,label)
    cases.append((label,work,coordinates,stages))
    correction = float(overhead[0]['overhead_ms'])*1e6/(5*work['layers']+3)
    for s in stages:
        observed = [r for r in extra_rows if (r['model'],int(r['batch']),r['tag'],r['kind'])==
                    (model,batch,tag+'-stage',s['kind'])]
        if len(observed)!=1 or abs(float(observed[0]['closure_max_fraction']))>.03:
            raise ValueError('stage-control ledger is incomplete or does not close')
        measured = float(observed[0]['class_total_ns'])/float(observed[0]['instances_per_step'])-correction
        if measured<=s['barrier_ns']:raise ValueError('overhead correction exceeds the stage signal')
        rows.append(dict(source='stage_control',label=label,kind=s['kind'],key=s['key'],
                         observed_ns=measured,prediction_ns=s['stage_ns']))
    fits=cal['body_fits'];keys=sorted({r['key'] for r in rows})
    start=np.array([fits[k]['coefficients']['startup_ns']/1000 for k in keys])
    def predicted(x):return np.array([r['prediction_ns']+1000*x[keys.index(r['key'])] for r in rows])
    observed=np.array([r['observed_ns'] for r in rows])
    optimum=least_squares(lambda x:np.r_[(predicted(x)-observed)/np.maximum(observed,1000),.03*x],
                         np.zeros(len(keys)),bounds=(-start,np.full(len(keys),5.)),loss='soft_l1')
    counts=Counter(r['key'] for r in rows)
    for key,delta in zip(keys,optimum.x):
        fit=fits[key];fit['coefficients']['startup_ns']+=float(delta*1000)
        fit['startup_refinement']=dict(evidence='inferred',samples=counts[key],delta_ns=float(delta*1000),
            method='regularized shared-body fixed-cost update; all supplied stages are training')
    cal['fit_status']='initial_overlap_fitted'
    cal.setdefault('stage_training_inputs',[]).append(dict(base_fit=str(a.base_fit),
        base_fit_receipt_sha256=sha(a.base_fit/'fit-receipt.json'),class_audit=str(a.class_audit),
        class_audit_receipt_sha256=sha(a.class_audit/'receipt.json'),
        observation_inputs=audit['inputs'],correctness=proof))
    cal['pending']=['fresh E-3 class, step and ranking validation on all seven development cells',
                    'E-4 measured selection; no training result is acceptance']
    output=a.out/'target.json';write(output,target)
    differences=[]
    for label,work,coordinates,before in cases:
        after=predict(work,coordinates,label+'-refined',output)
        for old,new in zip(before,after):
            expected=old['stage_ns']+1000*optimum.x[keys.index(old['key'])]
            differences.append(abs(new['stage_ns']-expected))
    if max(differences)>.05:raise ValueError('startup fit and native stage prices disagree')
    for row,value in zip(rows,predicted(optimum.x)):
        row.update(refined_ns=float(value),error_pct=float((value/row['observed_ns']-1)*100),
                   independent_validation=False)
    with(a.out/'training.csv').open('w')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write(a.out/'receipt.json',dict(source_target_sha256=sha(a.target),target_sha256=sha(output),
        probe_sha256=sha(probe),script_sha256=sha(__file__),class_audit_sha256=sha(a.class_audit/'receipt.json'),
        base_training_sha256=sha(a.base_fit/'training.csv'),
        training_stages=len(rows),native_price_max_difference_ns=max(differences),solver_gate_pass=False,
        evidence='training-only fixed-cost recalibration; class-control observations are no longer held out'))
    main_kinds={'qkv','attention','o','gate_up','down','lm_head'}
    print(json.dumps(dict(training_stages=len(rows),native_price_max_difference_ns=max(differences),
        max_training_error_pct=max(abs(r['error_pct']) for r in rows),
        max_main_class_training_error_pct=max(abs(r['error_pct']) for r in rows if r['kind'] in main_kinds),
        max_boundary_training_error_pct=max(abs(r['error_pct']) for r in rows if r['kind'] not in main_kinds),
        solver_gate_pass=False)))


if __name__=='__main__':main()
