#!/usr/bin/env python3
"""Fit the production SM80 projection issue rate from matched stage arms."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def features(row, sms):
    n = int(row.get('n') or row['padded_n'])
    width, depth, slots, residency = (int(row[key]) for key in ('tn', 'tk', 'slots', 'residency'))
    if row['clipped_output_bitwise_equal'] and 'n' not in row:
        raise ValueError('padded measurements require the logical N extent')
    weights = int(row['logical_weight_bytes'])
    k, remainder = divmod(weights, 2*n)
    if remainder or k <= 0:
        raise ValueError('invalid measured projection weight bytes')
    grid = sms*residency
    tasks = (n+width-1)//width
    waves = (tasks+grid-1)//grid
    active = min(tasks, grid)
    if active != int(row['active_ctas']):
        raise ValueError('recorded active CTAs differ from the measured stage geometry')
    x = (1, math.log2(width/64), math.log2(depth/64), slots-2, math.log2(k/4096),
         int(row['family']=='qkv'), math.log2(waves), max(0, active/sms-1),
         (tasks-(waves-1)*grid)/grid)
    per_cta = 2*width*k*waves
    return x, math.log(per_cta/float(row['stage_span_ns']))


def fit(data):
    size = len(data[0][0])
    gram = [[math.fsum(x[i]*x[j] for x, _ in data) + (.1 if i==j and i else 0)
             for j in range(size)] for i in range(size)]
    vector = [math.fsum(x[i]*y for x, y in data) for i in range(size)]
    for index in range(size):
        top = max(range(index, size), key=lambda j:abs(gram[j][index]))
        gram[index], gram[top] = gram[top], gram[index]
        vector[index], vector[top] = vector[top], vector[index]
        pivot = gram[index][index]
        gram[index] = [value/pivot for value in gram[index]]
        vector[index] /= pivot
        for row in range(size):
            if row == index:
                continue
            scale = gram[row][index]
            gram[row] = [a-scale*b for a,b in zip(gram[row],gram[index])]
            vector[row] -= scale*vector[index]
    return vector


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--widths', required=True, type=Path)
    p.add_argument('--roles', required=True, type=Path)
    p.add_argument('--target', required=True, type=Path)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    target = json.loads(args.target.read_text())
    sms = target['resources']['num_sms']
    if target['arch_tag'] != 'sm_80' or sms != 108:
        raise ValueError('the issue fit binds the A800 SM80 stage measurements')
    roles = [row for row in rows(args.roles) if row['status']=='measured']
    if (not roles or any(float(row['trace_perturbation'])>.05 for row in roles)
            or sum(row['dominant']=='producer_issue_fraction' for row in roles)*2<=len(roles)):
        raise ValueError('require guarded role evidence that producer issue dominates')
    observations = rows(args.widths)
    if any(row['status'] not in ('measured', 'resource_rejected') for row in observations):
        raise ValueError('stage fit needs a resolved status for every arm')
    data = [features(row,sms) for row in observations if row['status']=='measured']
    batches = {int(row['batch']) for row in observations}
    batch, = batches
    errors = []
    for index,(x,y) in enumerate(data):
        weights = fit(data[:index]+data[index+1:])
        errors.append(abs(math.expm1(y-math.fsum(a*b for a,b in zip(weights,x)))))
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git','rev-parse',args.source_commit+'^{commit}'],
                                     cwd=root,text=True).strip()
    headers = ('Sm90DenseGemm.h','Sm90DenseEpilogue.h','Sm90DenseRing.cuh')
    body = {name:hashlib.sha256(subprocess.check_output(
        ['git','show',f'{commit}:include/tilemega/Backend/{name}'],cwd=root)).hexdigest()
        for name in headers}
    profile = dict(schema=1,batch=batch,sms=sms,source='guarded_cold_projection_spans',
        widths_sha256=sha(args.widths),roles_sha256=sha(args.roles),
        source_commit=commit,body_headers=body,
        measured_arms=len(data),resource_rejected_arms=len(observations)-len(data),
        loo_median_absolute_relative_error=statistics.median(errors),
        loo_p95_absolute_relative_error=statistics.quantiles(errors,n=20)[18],
        feature_order=['bias','log2_tn_over_64','log2_tk_over_64','slots_minus_2',
            'log2_k_over_4096','input_norm','log2_waves','dual_cta_pressure','last_wave_fraction'],
        coefficients=fit(data))
    target['sm80_native_constants']['projection_issue_profile']=profile
    with args.out.open('x') as stream:
        json.dump(target,stream,indent=2)
        stream.write('\n')
    print(json.dumps(profile,indent=2))


if __name__ == '__main__':
    main()
