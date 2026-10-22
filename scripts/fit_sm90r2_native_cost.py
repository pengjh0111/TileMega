#!/usr/bin/env python3
"""Fit initial native prices from measured components and retained stage-only controls.

This is calibration, not E-3 validation. It writes only under runs/sm90r2 and
keeps measured solve disabled until overlap and independent ranking are checked.
"""
import argparse
from collections import defaultdict
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
    p.add_argument('--components', type=Path, nargs='+', required=True)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--overhead', type=Path, required=True)
    p.add_argument('--probe', type=Path, default=ROOT / 'build/sm90r2_price_probe')
    p.add_argument('--target', type=Path, default=ROOT / 'configs/targets/sm_90_h800.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--build-tag', nargs='+', default=['CP2'])
    a = p.parse_args();a.out = a.out.resolve()
    if not a.out.is_relative_to(ROOT / 'runs/sm90r2') or a.out.exists():
        raise ValueError('retain initial-fit evidence in a new runs/sm90r2 directory')
    a.out.mkdir(parents=True)
    probe = a.out / 'price-probe'
    shutil.copy2(a.probe.resolve(), probe)
    values = [read(f) for f in a.components]
    if any(value.get('usable_for_calibration') is False for value in values):
        raise ValueError('a component input is explicitly excluded from calibration')
    observations = [row for value in values for row in value['observations']]
    if any((r['metric'].startswith(('attention_', 'v3_'))) and
           r.get('timestamp_arithmetic') != 'integer_delta_v1' for r in observations):
        raise ValueError('globaltimer calibration must subtract integer timestamps before conversion')
    if any(r['metric'] == 'native_ldmatrix_mma_k16' and not r.get('production_compatible')
           for r in observations):
        raise ValueError('native mainloop measurements require a matching resource audit')
    headers = {p['header_sha256'] for value in values
               if any(r['metric'] == 'native_ldmatrix_mma_k16' for r in value['observations'])
               for p in value['provenance']}
    if len(headers) != 1:
        raise ValueError('native mainloop components must bind one implementation header')
    columns = sorted(set().union(*(row.keys() for row in observations)))
    fits = {}
    def profile(key, coefficients):
        fits[key] = dict(evidence='inferred', sample_count=1, coefficients=coefficients,
                         method='unfitted feature extraction; not usable for measured solve')
    for family in ('gemm_mma_sync_tn8_32', 'gemm_mma_sync_tn40_64', 'gemm_mma_sync_tn72_plus'):
        for warps_k in (1,2,4):
            profile(f'{family}_wk{warps_k}', dict(startup_ns=0, memory_scale=1, mma_scale=1, activation_scale=0,
                             epilogue_scale=1, preissue_efficiency=0, tail_efficiency=0,
                             l2_efficiency=0, issue_ns=0,issue_service_scale=0))
    for d,q,norm in ((128,2,1),(64,4,0),(128,3,0),(128,4,0)):
        for mode in ('simt','mma_sync'):
            profile(f'attention_{mode}_d{d}_q{q}_norm{norm}', dict(body_scale=1,startup_ns=0,
                preissue_efficiency=0,tail_efficiency=0,l2_efficiency=0,issue_ns=0,issue_service_scale=0))
    for name in ('embedding','argmax'):profile('native_'+name,dict(startup_ns=0,per_element_ns=1,issue_ns=0,issue_service_scale=0))
    cal = dict(schema=1, flow_model='overlapped_operands_v1',issue_model='issuer_service_v2',
        overlap_model='operand_service_only_v2',
        attention_grid_model='explicit_grid_v1',
        barrier_model='active_arrival_v1',
        gemm_key_model='tn_warp_k_v1',
        fit_status='diagnostic_unfitted', observation_columns=columns,
        native_gemm_header_sha256=next(iter(headers)),
        observations=[[r.get(k) for k in columns] for r in observations],body_fits=fits,
        method='Initial stage fit; global trace overhead divided uniformly across stages is inferred',
        mainloop_method='Shared-load plus MMA per owned k16; exact warp partition, inferred TN/TK/register interpolation',
        pending=['preissue overlap calibration','held-out E-3 class and step errors',
                 '8-candidate ranking in each development cell','E-4 measured selection'],
        provenance=[dict(path=str(f),sha256=sha(f)) for f in a.components])
    target = read(a.target);target['sm90_dense_decode'] = cal
    initial = a.out/'feature-target.json';write(initial,target)
    trace_tags = {tag+'-stage' for tag in a.build_tag}
    ledger = [v for v in csv.DictReader(a.ledger.open()) if v['tag'] in trace_tags]
    overhead = {(v['model'],int(v['batch']),v['tag'].removesuffix('-stage')):float(v['overhead_ms'])*1e6
                for v in csv.DictReader(a.overhead.open()) if v['tag'] in trace_tags}
    if not ledger or not overhead:
        raise ValueError('matching stage-only and control observations are required')
    if {tag for _,_,tag in overhead} != set(a.build_tag):
        raise ValueError('one requested base geometry has no matched stage observation')
    points, workloads, artifacts = defaultdict(list), [], []
    native_headers = None
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    for model,batch,tag in sorted(overhead):
        binary=ROOT/f'runs/sm90r2/build/{tag}/{model}-B{batch}/decode.so'
        trace=ROOT/f'runs/sm90r2/build/{tag}-stage/{model}-B{batch}/decode.so'
        manifest=read(str(binary)+'.plan.json')
        identity=read(str(binary)+'.identity.json');instrumented=read(str(trace)+'.identity.json')
        if sha(binary)!=identity['binary_sha256'] or sha(trace)!=instrumented['binary_sha256'] or \
                identity['trace'] or not instrumented['trace'] or sha(str(binary)+'.cu')!=sha(str(trace)+'.cu'):
            raise ValueError('stage calibration needs unchanged matched control and trace programs')
        artifacts.append(dict(model=model,batch=batch,tag=tag,
            binary_sha256=sha(binary),trace_sha256=sha(trace)))
        if identity['source']['source_files']['include/tilemega/Backend/Sm90DenseGemm.h'] not in headers:
            raise ValueError('stage and component calibration use different native GEMM implementations')
        device_headers={k:v for k,v in identity['source']['source_files'].items()
                        if k.startswith(('include/tilemega/Backend/Sm90Dense',
                                         'include/tilemega/Codegen/tasks/Sm90Dense'))}
        if native_headers is not None and device_headers != native_headers:
            raise ValueError('stage observations mix native program implementations')
        native_headers=device_headers;cal['native_headers']=device_headers
        config=read(ROOT/f'configs/serving/sm90d1/inputs/{model}/config.json')
        coordinates=manifest['native_dense'];shapes={}
        if any(g['preissue_smem_bytes'] or g['prefetch_l2_bytes'] for g in
               [*coordinates['gemms'].values(),coordinates['attention']]):
            raise ValueError('base pricing must use controls without incoming transfers')
        for g in manifest['gemms']:shapes[g['family']]=[g['n'],g['k']]
        layers=config['num_hidden_layers']
        if len(manifest['gemms'])!=4*layers+1:raise ValueError('unexpected CP2 dense plan structure')
        work=dict(shapes=[shapes[n] for n in ('qkv','o','gate_up','down','lm_head')],batch=batch,
            layers=layers,head_dim=config.get('head_dim',config['hidden_size']//config['num_attention_heads']),
            queries=config['num_attention_heads']//config['num_key_value_heads'],qk_norm=config['model_type']=='qwen3')
        stem=a.out/f'{tag+"-" if len(a.build_tag)>1 else ""}{model}-B{batch}'
        write(Path(str(stem)+'.workload.json'),work)
        coord=Path(str(stem)+'.tmexec.json');write(coord,coordinates)
        command=[str(probe),str(initial),str(stem)+'.workload.json',str(coord)]
        raw=json.loads(subprocess.check_output(command,env=env,text=True))
        write(Path(str(stem)+'.features.json'),raw);workloads.append((model,batch,stem))
        measured={r['kind']:r for r in ledger if r['model']==model and int(r['batch'])==batch
                  and r['tag']==tag+'-stage'}
        for stage in raw['predictions'][0]['past575']['stages']:
            row=measured[stage['kind']]
            observed=float(row['class_total_ns'])/float(row['instances_per_step'])-overhead[model,batch,tag]/(5*layers+3)
            if observed<=stage['barrier_ns']:raise ValueError('overhead correction exceeds stage signal')
            points[stage['key']].append(dict(model=model,batch=batch,tag=tag,kind=stage['kind'],
                corrected_ns=observed,stage=stage))
    training=[]
    for key,fit in fits.items():
        rows=points[key]
        if not rows:
            count=sum(r['metric']=='native_ldmatrix_mma_k16' and r.get('warps_k')==int(key[-1])
                      for r in observations) if key.startswith('gemm_') else sum(
                      r['metric']=='attention_critical_before_barrier' for r in observations)
            fit.update(sample_count=count,
                method='Unvalidated cross-shape component prior; no CP2 stage observation')
            continue
        y=np.array([r['corrected_ns'] for r in rows]);bar=np.array([r['stage']['barrier_ns'] for r in rows])
        if key.startswith('gemm_'):
            start=np.array([r['stage']['startup_ns'] for r in rows]);floor=np.array([r['stage']['dram_ns'] for r in rows])
            memory=np.array([r['stage']['stream_ns'] for r in rows]);mma=np.array([r['stage']['compute_ns'] for r in rows])
            epi=np.array([r['stage']['epilogue_ns'] for r in rows]);act=np.array([r['stage']['activation_ns'] for r in rows])
            def predict(x):return bar+start+x[0]*1000+np.maximum.reduce([floor,memory*x[1],act*x[4],mma*x[2]])+epi*x[3]
            prior=np.array([1.,1.,1.,1.,1.])
            bounds=([0,.2,.1,.05,0],[20,4,8,4,2])
            names=['startup_ns','memory_scale','mma_scale','epilogue_scale','activation_scale']
        elif key.startswith('attention_'):
            body=np.array([r['stage']['flow_ns'] for r in rows])
            def predict(x):return bar+x[0]*1000+body*x[1]
            prior=np.array([1.,1.]);bounds=([0,.25],[20,4]);names=['startup_ns','body_scale']
        else:
            count=np.array([r['stage']['compute_ns'] for r in rows])
            def predict(x):return bar+x[0]*1000+count*x[1]
            prior=np.array([1.,1.]);bounds=([0,0],[20,10]);names=['startup_ns','per_element_ns']
        def residual(x):return np.r_[(predict(x)-y)/np.maximum(y,1000),.03*(x-prior)]
        optimum=least_squares(residual,prior,bounds=bounds,loss='soft_l1')
        for name,x in zip(names,optimum.x):fit['coefficients'][name]=float(x*1000 if name=='startup_ns' else x)
        fit.update(sample_count=len(rows),method='regularized nonnegative overlapped-operand stage fit; training only',
                   inputs=[dict(model=r['model'],batch=r['batch'],tag=r['tag'],kind=r['kind']) for r in rows],
                   training_median_error_pct=float(np.median(np.abs(predict(optimum.x)/y-1))*100))
        for r,pred in zip(rows,predict(optimum.x)):
            training.append(dict(model=r['model'],batch=r['batch'],tag=r['tag'],kind=r['kind'],body_key=key,
                corrected_stage_ns=r['corrected_ns'],prediction_ns=float(pred),
                error_pct=float((pred/r['corrected_ns']-1)*100),independent_validation=False))
    cal['fit_status']='initial_fitted';cal['body_fits']=fits
    write(a.out/'target.json',target)
    with (a.out/'training.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(training[0]));writer.writeheader();writer.writerows(training)
    write(a.out/'fit-receipt.json',dict(stage='initial calibration, not E-3 validation',
        artifacts=artifacts,
        ledger_sha256=sha(a.ledger),overhead_sha256=sha(a.overhead),probe_sha256=sha(probe),
        probe_snapshot=str(probe),
        fitter_sha256=sha(__file__),target_sha256=sha(a.out/'target.json'),source_target_sha256=sha(a.target)))
    for model,batch,stem in workloads:
        command=[str(probe),str(a.out/'target.json'),str(stem)+'.workload.json',str(stem)+'.tmexec.json']
        raw=json.loads(subprocess.check_output(command,env=env,text=True));write(Path(str(stem)+'.fitted.json'),raw)
    print(json.dumps(dict(training_stages=len(training),bodies=len(fits),fit_status=cal['fit_status'])))


if __name__=='__main__':main()
