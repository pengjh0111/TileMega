#!/usr/bin/env python3
"""Append stage diagnostics to V1–V4 without turning ratios into gates."""
import argparse
import csv
import json
import math
from pathlib import Path

from sm90d1_common import sha
from sm90r6_merge_analysis import write


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def stage_rows(path, dense):
    rows=read_csv(path);seen=set();result=[]
    expected={(model,batch) for model in (
        ('qwen3_1_7b','llama3_1_8b') if dense else ('qwen3_30b_a3b',)) for batch in (1,16)}
    for row in rows:
        key=row['model'],int(row['batch']);stage=row['stage']
        if key not in expected or (*key,stage) in seen:
            raise ValueError('unexpected or duplicate V1(b) stage')
        seen.add((*key,stage))
        if int(row['processes'])!=(3 if dense else 1):
            raise ValueError('V1(b) changed the specified sampling provenance')
        observed=float(row['measured_span_us' if dense else 'measured_boundary_us'])
        flow=float(row['flow_span_us' if dense else 'flow_space_sum_us'])
        additive=float(row['additive_stage_us'])
        components=json.loads(row['flow_components_per_instance_us' if dense else 'flow_space_medians_us'])
        if not components or not math.isclose(sum(components.values()),flow,rel_tol=1e-7,abs_tol=1e-8):
            raise ValueError('semantic components do not sum to the physical-stage prediction')
        values={'flow_sum_us':flow,'additive_us':additive,
                **{'space.'+kind+'_us':value for kind,value in components.items()}}
        if not math.isfinite(observed) or observed<=0 or any(
                not math.isfinite(value) or value<0 for value in values.values()):
            raise ValueError('invalid stage duration')
        if not all(math.isclose(float(row[name]),value/observed,rel_tol=1e-7,abs_tol=1e-8)
                   for name,value in [('flow_over_measured',flow),('additive_over_measured',additive)]):
            raise ValueError('retained diagnostic ratio disagrees with durations')
        for metric,value in values.items():
            note=('A0; 3 fresh processes; mean semantic spans vs median CTA envelope' if dense else
                  'F; existing R5 marks only; 1 historical process; median semantic spans vs critical-CTA boundary')
            if dense:note+='; D0 GPUs '+row.get('d0_gpus','recorded in raw')
            result.append(dict(check='V1(b)',model=key[0],batch=key[1],tp=1,
                candidate=('A0' if dense else 'F')+':'+stage,metric=metric,value=value,
                reference=observed,ratio=value/observed,status='reported',
                note=note+'; individual-space ratios are components, not full-stage estimates; diagnostics only',
                raw=str(path),raw_sha256=sha(path)))
    if {(model,batch) for model,batch,stage in seen}!=expected:
        raise ValueError('incomplete V1(b) model/batch matrix')
    required={'qkv','o','attention','lm_head'}|({'gate_up','down'} if dense else {'router','expert_fused'})
    for key in expected:
        if not required<={stage for model,batch,stage in seen if (model,batch)==key}:
            raise ValueError('incomplete V1(b) stage families')
    return result


def merge(base,dense,moe,out):
    retained=[row for row in read_csv(base) if row['check']!='V1(b)']
    if 'V1(a)' not in {row['check'] for row in retained} or any(
            not any(row['check'].startswith(prefix) for row in retained) for prefix in ('V2','V3','V4')):
        raise ValueError('base acceptance table is missing existing diagnostics')
    for row in retained:
        row['note']=row['note'].replace('CP-P correctness gates remain incomplete',
            'correctness gate is independent of diagnostic ratios')
    added=stage_rows(dense,True)+stage_rows(moe,False)
    write(out,retained+added)
    return dict(retained_rows=len(retained),stage_rows=len(added))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','dense','moe','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(merge(a.base,a.dense,a.moe,a.out)))
