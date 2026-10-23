#!/usr/bin/env python3
"""Fit native body and transfer prices together from frozen training evidence.

Absolute stage observations constrain attribution; paired requests constrain
geometry and transfer choices. The native predictor independently checks the
Python feature equations before and after fitting. No input is validation.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.optimize import least_squares

from sm90d1_common import ROOT, read, sha, write


class JointPrices:
    def __init__(self, rows, calibration):
        self.rows = rows
        self.stages, unique, indices, weights, row_ids = [], {}, [], [], []
        for row_id, row in enumerate(rows):
            for term in row['terms']:
                key = json.dumps(term['stage'], sort_keys=True)
                if key not in unique:
                    unique[key] = len(self.stages)
                    self.stages.append(term['stage'])
                indices.append(unique[key]); weights.append(term['weight']); row_ids.append(row_id)
        self.indices, self.weights, self.row_ids = np.array(indices), np.array(weights), np.array(row_ids)
        self.parameters, prior, lower, upper, lookup = [], [], [], [], {}
        fits = calibration['body_fits']

        def add(key, name, low, high, scale=1):
            lookup[key, name] = len(prior)
            self.parameters.append((key, name, scale))
            prior.append(fits[key]['coefficients'][name]/scale)
            lower.append(low); upper.append(high)

        body_keys = sorted({s['key'] for s in self.stages})
        for key in body_keys:
            if key.startswith('gemm_'):
                for name, low, high, scale in [('startup_ns',0,20,1000),('memory_scale',.2,4,1),
                        ('activation_scale',0,2,1),('mma_scale',.1,8,1)]:
                    add(key,name,low,high,scale)
            elif key.startswith('attention_'):
                add(key,'startup_ns',0,20,1000); add(key,'body_scale',.25,4)
        ops = dict(qkv='store',o='residual',gate_up='swiglu',down='residual',lm_head='argmax')
        for op in sorted(set(ops.values())):
            add('native_epilogue_'+op,'scale',.05,4)
        protocol_keys = sorted({s['protocol_key'] for s in self.stages
            if s['preissue_service_ns'] or s['late_service_ns'] or s['issue_count'] or s['receive_count']})
        for key in protocol_keys:
            for name, low, high, scale in [('preissue_efficiency',0,1,1),('tail_efficiency',0,1,1),
                    ('issue_ns',0,5,1000),('issue_service_scale',0,2,1),('receive_ns',0,5,1000)]:
                add(key,name,low,high,scale)
        self.prior, self.lower, self.upper = np.array(prior), np.array(lower), np.array(upper)
        self.prior = np.clip(self.prior,self.lower,self.upper)
        self.gemm = np.array([i for i,s in enumerate(self.stages) if s['key'].startswith('gemm_')])
        self.attention = np.array([i for i,s in enumerate(self.stages) if s['key'].startswith('attention_')])
        self.protocol = np.array([i for i,s in enumerate(self.stages) if s['protocol_key'] in protocol_keys])
        self.features = {name:np.array([s[name] for s in self.stages]) for name in
            ('barrier_ns','startup_ns','initialization_ns','read_startup_ns','flow_ns','compute_ns',
             'epilogue_ns','dram_ns','stream_ns','activation_ns','overlap_limit_ns','preissue_service_ns',
             'late_service_ns','issue_count','issue_service_ns','receive_count')}
        self.gemm_parameters = np.array([[lookup[self.stages[i]['key'],name] for name in
            ('startup_ns','memory_scale','activation_scale','mma_scale')] for i in self.gemm])
        self.attention_parameters = np.array([[lookup[self.stages[i]['key'],name] for name in
            ('startup_ns','body_scale')] for i in self.attention])
        self.protocol_parameters = np.array([[lookup[self.stages[i]['protocol_key'],name] for name in
            ('preissue_efficiency','tail_efficiency','issue_ns','issue_service_scale','receive_ns')]
            for i in self.protocol])
        self.epilogue_parameters = np.array([lookup['native_epilogue_'+ops[self.stages[i]['kind']],'scale']
            for i in self.gemm])
        self.raw_compute = np.array([self.stages[i]['compute_ns']/fits[self.stages[i]['key']]['coefficients']['mma_scale']
            for i in self.gemm])
        self.raw_epilogue = np.array([self.stages[i]['epilogue_ns']/fits['native_epilogue_'+ops[self.stages[i]['kind']]]['coefficients']['scale']
            for i in self.gemm])
        attention_scale = np.array([fits[self.stages[i]['key']]['coefficients']['body_scale'] for i in self.attention])
        self.raw_attention = {name:self.features[name][self.attention]/attention_scale
            for name in ('flow_ns','preissue_service_ns','late_service_ns')}
        if any(f['coefficients'].get('l2_efficiency',0) for f in fits.values()):
            raise ValueError('joint fit requires zero-credit L2 priors')

    def stage_prices(self, x):
        f = self.features
        startup,flow,compute,epilogue,limit,early,late = [f[k].copy() for k in
            ('startup_ns','flow_ns','compute_ns','epilogue_ns','overlap_limit_ns','preissue_service_ns','late_service_ns')]
        g,a,p = self.gemm,self.attention,self.protocol
        v = x[self.gemm_parameters]
        startup[g] = v[:,0]*1000+f['initialization_ns'][g]+f['read_startup_ns'][g]
        activation = f['activation_ns'][g]*v[:,2]
        flow[g] = np.maximum.reduce([f['dram_ns'][g],f['stream_ns'][g]*v[:,1],activation])
        compute[g] = self.raw_compute*v[:,3]
        epilogue[g] = self.raw_epilogue*x[self.epilogue_parameters]
        limit[g] = f['read_startup_ns'][g]+np.maximum(flow[g]-np.maximum(compute[g],activation),0)
        v = x[self.attention_parameters]
        startup[a] = 1000*v[:,0]
        flow[a] = self.raw_attention['flow_ns']*v[:,1];compute[a] = flow[a]
        limit[a] = np.maximum(flow[a]-f['dram_ns'][a],0)
        early[a] = self.raw_attention['preissue_service_ns']*v[:,1]
        late[a] = self.raw_attention['late_service_ns']*v[:,1]
        value = f['barrier_ns']+startup+np.maximum(flow,compute)+epilogue
        v = x[self.protocol_parameters]
        value[p] += (f['issue_count'][p]*1000*v[:,2]+f['issue_service_ns'][p]*v[:,3]+
            f['receive_count'][p]*1000*v[:,4]-np.minimum(limit[p],early[p]*v[:,0]+late[p]*v[:,1]))
        return value

    def predict(self, x):
        values = self.stage_prices(x)
        return np.bincount(self.row_ids,weights=self.weights*values[self.indices],minlength=len(self.rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args();args.out = args.out.resolve()
    if args.out.exists() or not args.out.is_relative_to(ROOT/'runs/sm90r2'):
        raise ValueError('retain previous fits in a new runs/sm90r2 directory')
    data = read(args.observations)
    if data['schema'] != 1:raise ValueError('unsupported joint observations')
    for source in data['sources']:
        if sha(source['path']) != source['sha256']:raise ValueError('training evidence changed')
    if sha(data['target']) != data['target_sha256'] or sha(data['probe']) != data['probe_sha256']:
        raise ValueError('frozen target or native predictor changed')
    target = read(data['target']);cal = target['sm90_dense_decode'];rows = data['rows']
    args.out.mkdir();shutil.copy2(data['probe'],args.out/'price-probe')
    prices = JointPrices(rows,cal)
    initial = prices.stage_prices(prices.prior)
    difference = max(abs(v-s['stage_ns']) for v,s in zip(initial,prices.stages))
    if difference>.05:raise ValueError(f'initial feature equations disagree with native prices: {difference}')
    observed = np.array([row['observed_ns'] for row in rows]);sigma = np.array([row['sigma_ns'] for row in rows])
    if np.any(sigma<=0):raise ValueError('positive observation scale required')
    result = least_squares(lambda x:np.r_[(prices.predict(x)-observed)/sigma,.03*(x-prices.prior)],
        prices.prior,bounds=(prices.lower,prices.upper),loss='soft_l1',max_nfev=1000)
    result.x[np.abs(result.x)<1e-12] = 0
    for (key,name,scale),value in zip(prices.parameters,result.x):
        fit = cal['body_fits'][key];fit['coefficients'][name] = float(value*scale)
        fit['joint_refinement'] = dict(evidence='inferred',method='shared body/operation/protocol joint fit',
            observations_sha256=sha(args.observations),independent_validation=False)
    cal['fit_status'] = 'initial_overlap_fitted'
    cal['paired_training_receipts_sha256'] = data['paired_training_receipts_sha256']
    cal['pending'] = ['fresh class/step/ranking validation after joint recalibration','final measured solver selection']
    write(args.out/'target.json',target)
    predictions = {}
    for stem,item in data['inputs'].items():
        values = json.loads(subprocess.check_output([str(args.out/'price-probe'),str(args.out/'target.json'),
            item['workload'],item['coordinates'],'--request-samples'],text=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='')))
        write(args.out/(stem+'.predictions.json'),values);predictions[stem] = values['predictions']
    fitted = prices.predict(result.x);differences=[];table=[]
    for row,value in zip(rows,fitted):
        def selected(index):
            item=predictions[row['group']][index]
            return item['request']['step_ns'] if row['kind']=='request' else next(
                s['stage_ns'] for s in item['past575']['stages'] if s['kind']==row['kind'])
        native=selected(row['candidate_index'])
        if row['control_index'] is not None:native-=selected(row['control_index'])
        differences.append(abs(native-value))
        table.append(dict(source=row['source'],label=row['label'],observed_ns=row['observed_ns'],
            fitted_ns=float(value),residual_ns=float(value-row['observed_ns']),sigma_ns=row['sigma_ns'],
            quantity='absolute' if row['control_index'] is None else 'delta',independent_validation=False))
    if max(differences)>.05:raise ValueError(f'fitted equations disagree with native predictor: {max(differences)}')
    with(args.out/'training.csv').open('w')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
    write(args.out/'receipt.json',dict(observations=str(args.observations),observations_sha256=sha(args.observations),
        source_target_sha256=data['target_sha256'],target_sha256=sha(args.out/'target.json'),
        probe_sha256=sha(args.out/'price-probe'),script_sha256=sha(__file__),native_price_max_difference_ns=max(differences),
        training_rows=len(rows),parameters=len(prices.parameters),optimization_success=bool(result.success),
        optimization_message=result.message,evaluations=result.nfev,solver_gate_pass=False,
        evidence='inferred joint training fit; no independent validation'))
    print(json.dumps(dict(training_rows=len(rows),native_price_max_difference_ns=max(differences),
        evaluations=result.nfev,optimization_success=bool(result.success),solver_gate_pass=False)))


if __name__=='__main__':main()
