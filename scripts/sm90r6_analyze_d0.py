#!/usr/bin/env python3
"""Compare dense A0 D0 stage spans with fixed FlowSpace and additive prices."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics as st

from sm90d1_common import read,sha
from sm90r4_inputs import model_dir
from sm90r6_merge_analysis import write


def tsv(path):
    with path.open() as stream:return list(csv.DictReader(stream,delimiter='\t'))


def family(row):
    kind=row['kind_name'];name=row['name']
    if kind=='kEmbedding':return 'embedding'
    if kind=='kArgmaxReduce':return 'argmax'
    if kind=='kFusedAttention':return 'attention'
    if kind=='kAttentionMerge':return 'attention_merge'
    for key,value in [('lm_head','lm_head'),('qkv','qkv'),('o_proj','o'),('gate_up','gate_up'),('down_proj','down')]:
        if key in name:return value
    raise ValueError('unclassified dense runtime stage: '+str(row))


def stage_mapping(data):
    c=data['coordinates'];argmax='argmax' if c.get('argmax','la')=='stage' else 'lm_head'
    mapping=dict(gemm0='qkv',gemm1='o',gemm2='gate_up',gemm3='down',gemm4='lm_head',
                 argmax=argmax,argmax_la=argmax,attention='attention',merge='attention',
                 embedding=argmax if c['emb']=='la' else 'embedding',embedding_la=argmax)
    if c['attention'].get('merge','la')=='stage':mapping['merge']='attention_merge'
    return mapping


def predictions(data):
    mapping=stage_mapping(data)
    flow=defaultdict(float);counts=defaultdict(int)
    for row in data['spaces']:
        k=mapping[row['kind']]
        flow[k]+=row['last_end_ns']-max(0,row['first_start_ns'])
        if row['kind'] not in ('embedding','argmax','merge') or k==row['kind']:counts[k]+=1
    additive=defaultdict(float)
    for row in data['additive_stages']['stages']:
        k=mapping.get(row['kind'],row['kind'])
        additive[k]+=row['instances']*row['stage_ns']
    return flow,additive,counts


def analyze(directory,prediction,bandwidth,compute):
    c=prediction['coordinates'];cell=prediction['input'];model=cell['model'];batch=cell['batch']
    config=read(model_dir(model)/'config.json')
    h=config['hidden_size'];hq=config['num_attention_heads'];hkv=config['num_key_value_heads'];dim=config.get('head_dim') or h//hq
    accepted=read(directory/'checks/result.json')
    if not accepted['passed'] or len(accepted['processes'])!=3:raise ValueError('D0 requires three accepted processes')
    totals=defaultdict(list);spans=defaultdict(list);floors=defaultdict(list);tails=defaultdict(list)
    processes=[];instance_counts={};overheads=[]
    for process in accepted['processes']:
        trial=Path(process['directory']);trace=trial/'trace';run=read(trace/'run.json')
        if (read(trial/'guard/guard_result.json')['code'] or not run['full_token_equivalence'] or
                process['raw_sha256']!=sha(trace/'stage_trace.tsv')):raise ValueError('unaccepted or changed D0 trace')
        source=read(directory/'reference.json')
        if run['reference_sha256']!=sha(source['binary']):raise ValueError('D0 reference changed')
        identity=read(source['binary']+'.identity.json')
        if (identity['binary_sha256']!=run['reference_sha256'] or
                identity['plan']['native_dense']!=c):
            raise ValueError('D0 prediction is not the measured A0 coordinate')
        if read(trace/'trace-tokens.json')!=read(trace/'reference-tokens.json'):
            raise ValueError('retained full D0 tokens differ')
        stages={int(r['stage']):r for r in tsv(trace/'runtime_stages.tsv')};groups=defaultdict(list)
        steps=defaultdict(list)
        for row in tsv(trace/'step_trace.tsv'):
            if int(row['kernel_end']):steps[int(row['iteration'])].append(row)
        iterations=sorted(steps)
        if len(iterations)!=16:raise ValueError('D0 must retain sixteen steady-state steps')
        grid=len(steps[iterations[0]])
        if any(len(v)!=grid or {int(r['worker']) for r in v}!=set(range(grid)) for v in steps.values()):
            raise ValueError('incomplete step ledger')
        # LA embedding skips the prologue on steady-state steps. Its old slot
        # can still contain the initial request's stamp after the ring wraps.
        for row in tsv(trace/'stage_trace.tsv'):
            if int(row['t_release']) and int(row['iteration']) in steps:
                groups[int(row['iteration']),int(row['stage'])].append(row)
        per_type=defaultdict(list)
        for iteration in iterations:
            indices=sorted(s for i,s in groups if i==iteration)
            expected=sorted(stages)
            if c['emb']=='la':expected=[s for s in expected if family(stages[s])!='embedding']
            if indices!=expected:raise ValueError('incomplete D0 stage coverage')
            previous=None;step=defaultdict(float);counts=defaultdict(int)
            for index in indices:
                stage=stages[index];kind=family(stage);rows=groups[iteration,index]
                workers=[int(r['worker']) for r in rows]
                if len(workers)!=grid or set(workers)!=set(range(grid)):raise ValueError('incomplete CTA ledger')
                begin=min(int(r['t_begin']) for r in rows);release=max(int(r['t_release']) for r in rows)
                critical=max(rows,key=lambda r:int(r['t_tasks_end']))
                if any(not 0<int(r['t_begin'])<=int(r['t_tasks_end'])<=int(r['t_release']) for r in rows):
                    raise ValueError('nonmonotonic D0 timestamps')
                if int(critical['past'])<256:raise ValueError('D0 is outside steady state')
                duration=release-(begin if previous is None else previous)
                step[kind]+=duration;counts[kind]+=1
                spans[kind].append(release-begin)
                tails[kind].append(int(critical['t_release'])-int(critical['t_tasks_end']))
                weight=int(stage['weight_bytes']);traffic=weight;flops=batch*weight
                if kind=='attention':
                    traffic=4*batch*hkv*dim*(int(critical['past'])+1)
                    flops=4*batch*hq*dim*(int(critical['past'])+1)
                if kind=='embedding':traffic=2*batch*h;flops=0
                floors[kind].append(max(traffic/bandwidth,flops/(compute*1000)))
                previous=release
            if instance_counts and dict(counts)!=instance_counts:raise ValueError('D0 stage multiplicity changed')
            instance_counts=dict(counts)
            for k,v in step.items():per_type[k].append(v)
        for k,v in per_type.items():totals[k].append(st.median(v))
        overheads.append(run['overhead_pct']);processes.append(str(trial))
    flow,additive,_=predictions(prediction);rows=[]
    components=defaultdict(lambda:defaultdict(float));mapping=stage_mapping(prediction)
    for space in prediction['spaces']:
        components[mapping[space['kind']]][space['kind']]+=(
            space['last_end_ns']-max(0,space['first_start_ns']))/1000
    for kind in sorted(totals):
        count=instance_counts[kind];measured=st.median(spans[kind])/1000
        flow_us=flow.get(kind,0)/count/1000;additive_us=additive.get(kind,0)/count/1000
        boundary=st.median(totals[kind])/1000;floor=st.median(floors[kind])/1000
        rows.append(dict(model=model,batch=batch,stage=kind,instances_per_step=count,processes=3,
            measured_span_us=measured,flow_span_us=flow_us,flow_over_measured=flow_us/measured,
            additive_stage_us=additive_us,additive_over_measured=additive_us/measured,
            flow_components_per_instance_us=json.dumps(
                {k:v/count for k,v in components[kind].items()},sort_keys=True,separators=(',',':')),
            attributed_step_us=boundary,memory_compute_floor_us=floor,
            attributed_excess_us=boundary-count*floor,barrier_tail_us=st.median(tails[kind])/1000,
            instrumentation_overhead_pct=st.median(overheads),evidence='verified measurements; inferred bounds',
            note='median CTA envelope span versus mean per-instance sum of semantic-space spans; fused reducers folded into owning stage; boundary attribution telescopes and may be negative with group release; diagnostics only',
            raw='|'.join(processes)))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--predictions',type=Path,nargs='+',required=True)
    p.add_argument('--peaks',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with a.peaks.open() as stream:peaks=list(csv.DictReader(stream))
    bandwidth=max(float(r['median']) for r in peaks if r['unit']=='GB/s')
    compute=max(float(r['median']) for r in peaks if r['unit']=='TFLOPS')
    rows=[];seen=set()
    for path in a.predictions:
        for line in path.read_text().splitlines():
            if not line.strip():continue
            prediction=json.loads(line);name=prediction['input']['id']
            if name in seen:raise ValueError('duplicate D0 cell')
            seen.add(name);rows.extend(analyze(a.root/name,prediction,bandwidth,compute))
    write(a.out,rows)
    print(json.dumps(dict(cells=len(seen),rows=len(rows))),flush=True)


if __name__=='__main__':main()
