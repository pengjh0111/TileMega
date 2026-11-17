#!/usr/bin/env python3
"""Task-level static list scheduling for the frozen dense B1 structures."""
import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
from sm90_local_global_local import CONFIG, ROOT, SHAPES, sha, write

G=132

class Graph:
    def __init__(self):
        self.nodes=[];self.units=[];self.groups=[];self.group_ids={}
    def event(self,producers):
        key=tuple(sorted(set(producers)))
        if not key:return None
        if key not in self.group_ids:
            self.group_ids[key]=len(self.groups);self.groups.append(list(key))
        return self.group_ids[key]
    def unit(self,name,layer,count,duration,deps):
        unit=len(self.units);ids=[]
        for coordinate in range(count):
            waits=deps(coordinate) if callable(deps) else deps
            waits=sorted(set(e for e in waits if e is not None))
            ids.append(len(self.nodes))
            self.nodes.append(dict(id=ids[-1],unit=unit,kind=name,layer=layer,coordinate=coordinate,
                cost_us=duration/max(1,math.ceil(count/G)),wait_events=waits))
        self.units.append(dict(id=unit,kind=name,layer=layer,tasks=ids))
        return ids

def graph_for(s):
    local=s['structure']=='S_local';g=Graph();c=s['coordinates'];gemms=c['gemms'];att=c['attention']
    scores=s.get('isolated_us',{})
    if not local:
        calibration=json.loads((ROOT/'configs/targets/sm_90_h800.json').read_text())['sm90_native_constants']
        for family,(n,k) in SHAPES.items():
            a=gemms[family];tn=a['tile_n'];tasks=n//tn;per=math.ceil(tasks/G)
            wt=2*n*k;at=tasks*k*2;wc=per*tn*k*2;ac=per*k*2
            kind=dict(qkv='qkv',o='residual',gate_up='gate_up',down='residual',lm_head='lm_head')[family]
            ep=calibration['epilogue_ns'][kind]['wk'+str(a['warps_k'])][str(tn)]['1']
            scores[family]=(calibration['c_bar_ns']['132']+calibration['dram_latency_ns']+ep+
                max(wt/calibration['dram_gbps'],(wt+at)/calibration['l2_gbps'],
                    (wc+ac)/calibration['sm_gbps'],wc*calibration['dram_latency_ns']/(a['slots']*a['slot_bytes'])))/1000
        ac=calibration['attention_ns']['d128:q2:norm1'];keys=math.ceil(576/att['chunks'])
        scores['attention']=(calibration['c_bar_ns']['132']+calibration['l2_latency_ns']+ac['prepare']+
            max(8*575*128*4/calibration['dram_gbps'],math.ceil(8*att['chunks']/G)*
                (keys*128*4/(calibration['dram_gbps']/G)+math.ceil(keys/64)*ac['block']))+
            ac['merge'][str(att['chunks'])])/1000
    def score(name,default):return scores.get(name,default)
    def full(ids):return [g.event(ids)]
    def slices(ids,tn,begin,end):return ids[begin//tn:(end+tn-1)//tn]
    prev=g.unit('embedding',0,1,1.0,[])
    for layer in range(28):
        if local:prev=g.unit('rms_qkv',layer,1,1.0,full(prev))
        qkv=g.unit('qkv',layer,4096//gemms['qkv']['tile_n'],score('qkv',12),full(prev))
        if local:
            prep=g.unit('prepare',layer,8,2.0,lambda bg:full(slices(qkv,gemms['qkv']['tile_n'],bg*512,(bg+1)*512)))
            adeps=lambda t:full([prep[t//att['chunks']]])
        else:
            adeps=lambda t:full(slices(qkv,gemms['qkv']['tile_n'],(t//att['chunks'])*512,(t//att['chunks']+1)*512))
        attention=g.unit('attention',layer,8*att['chunks'],score('attention',12),adeps)
        if att.get('merge')=='stage' and att['chunks']>1:
            context=g.unit('merge',layer,8,2.5,lambda bg:full(attention[bg*att['chunks']:(bg+1)*att['chunks']]))
        else:context=attention
        o=g.unit('o',layer,2048//gemms['o']['tile_n'],score('o',10),full(context))
        prev=o
        if local:
            prev=g.unit('residual_o',layer,8,1.0,lambda t:full(slices(o,gemms['o']['tile_n'],t*256,(t+1)*256)))
            prev=g.unit('rms_gate',layer,1,1.0,full(prev))
        gate=g.unit('gate_up',layer,12288//gemms['gate_up']['tile_n'],score('gate_up',24),full(prev))
        prev=gate
        if local:
            prev=g.unit('swiglu',layer,24,1.5,lambda t:full(slices(gate,gemms['gate_up']['tile_n'],t*512,(t+1)*512)))
        down=g.unit('down',layer,2048//gemms['down']['tile_n'],score('down',18),full(prev))
        prev=down
        if local:prev=g.unit('residual_down',layer,8,1.0,lambda t:full(slices(down,gemms['down']['tile_n'],t*256,(t+1)*256)))
    if local:prev=g.unit('rms_head',0,1,1.0,full(prev))
    g.unit('head',0,151936//gemms['lm_head']['tile_n'],score('lm_head',210),full(prev))
    return g

def parents(g,basic=False):
    result=[]
    for n in g.nodes:
        if basic:
            p=g.units[n['unit']-1]['tasks'] if n['unit'] else []
        else:p=sorted(set(t for e in n['wait_events'] for t in g.groups[e]))
        result.append(p)
    return result

def replay(g,owner,queues,pp,handoff=.42121):
    n=len(g.nodes);successors=[[] for _ in range(n)];indegree=[0]*n
    deps=[set(p) for p in pp]
    for q in queues:
        for a,b in zip(q,q[1:]):deps[b].add(a)
    for t,p in enumerate(deps):
        indegree[t]=len(p)
        for a in p:successors[a].append(t)
    ready=[t for t in range(n) if not indegree[t]];heapq.heapify(ready)
    finish=[0.0]*n;visited=[]
    while ready:
        t=heapq.heappop(ready);visited.append(t)
        data=set(pp[t]);begin=max((finish[p]+(handoff if p in data and owner[p]!=owner[t] else 0)
                                 for p in deps[t]),default=0)
        finish[t]=begin+g.nodes[t]['cost_us']+.3718333333
        for child in successors[t]:
            indegree[child]-=1
            if not indegree[child]:heapq.heappush(ready,child)
    if len(visited)!=n:raise ValueError('dependency/per-track sequence cycle')
    if sorted(t for q in queues for t in q)!=list(range(n)):raise ValueError('task coverage is not unique')
    return max(finish),finish

def basic(g):
    owner=[n['coordinate']%G for n in g.nodes];queues=[[] for _ in range(G)]
    for n in g.nodes:queues[owner[n['id']]].append(n['id'])
    score,_=replay(g,owner,queues,parents(g,True))
    return dict(algorithm='basic_rr_topological_v1',owner=owner,queues=queues,predicted_us=score,aggregate=True)

def selected(g,b,trace_path=None):
    pp=parents(g);n=len(g.nodes);succ=[[] for _ in range(n)]
    for t,deps in enumerate(pp):
        for p in deps:succ[p].append(t)
    critical=[0.0]*n
    for t in reversed(range(n)):
        if any(p>=t for p in pp[t]):raise ValueError('graph is not in semantic topological order')
        critical[t]=g.nodes[t]['cost_us']+max((critical[s] for s in succ[t]),default=0)
    indegree=[len(p) for p in pp]
    frontier=[(-critical[t],t) for t in range(n) if not indegree[t]];heapq.heapify(frontier)
    owner=[-1]*n;finish=[0.0]*n;available=[0.0]*G;queues=[[] for _ in range(G)]
    while frontier:
        _,t=heapq.heappop(frontier)
        by_owner=[0.0]*G
        for p in pp[t]:by_owner[owner[p]]=max(by_owner[owner[p]],finish[p])
        largest=sorted(range(G),key=lambda i:(by_owner[i],i),reverse=True)[:2]
        choices=[]
        for track in range(G):
            foreign=by_owner[largest[1] if track==largest[0] else largest[0]]
            ready=max(by_owner[track],foreign+.42121 if foreign else 0)
            end=max(available[track],ready)+g.nodes[t]['cost_us']+.3718333333
            choices.append((end,track))
        end,track=min(choices);owner[t]=track;finish[t]=end;available[track]=end;queues[track].append(t)
        for child in succ[t]:
            indegree[child]-=1
            if not indegree[child]:heapq.heappush(frontier,(-critical[child],child))
    score,_=replay(g,owner,queues,pp)
    if trace_path:
        write(trace_path,dict(candidate='frontier',gpu_executed=False,owner=owner,queues=queues,
            predicted_us=score,basic_us=b['predicted_us'],critical_priority=critical,
            different_owner_count=sum(x!=y for x,y in zip(owner,b['owner'])),
            rule='priority frontier; predecessor/track/handoff earliest finish; append per-track order; replay'))
    use_basic=b['predicted_us']<=score
    result=dict(b if use_basic else dict(owner=owner,queues=queues,predicted_us=score,aggregate=False))
    result.update(algorithm='critical_frontier_earliest_finish_v1',selected_candidate='basic' if use_basic else 'frontier',
        candidates=dict(basic_us=b['predicted_us'],frontier_us=score),
        cost_model='S_local isolated medians, S_TM native calibrated operator model; divide by waves; fixed untuned auxiliary costs; calibrated cross-owner and publication cost')
    return result

def export_plan(g,p,structure,path):
    # Materialize the exact aggregate events the executor must consume.
    events=[];lookup={}
    def event(producers):
        key=tuple(producers)
        if key not in lookup:lookup[key]=len(events);events.append(dict(producers=list(key),count=len(key)))
        return lookup[key]
    waits=[]
    for node in g.nodes:
        if p['aggregate']:
            producers=g.units[node['unit']-1]['tasks'] if node['unit'] else []
            waits.append([event(producers)] if producers else [])
        else:waits.append([event(g.groups[e]) for e in node['wait_events']])
    publications=[[] for _ in g.nodes]
    for eid,e in enumerate(events):
        for t in e['producers']:publications[t].append(eid)
    # Once acquired, an event stays complete for this decode iteration.
    for q in p['queues']:
        seen=set()
        for t in q:
            original=waits[t];waits[t]=[e for e in original if e not in seen];seen.update(original)
    nodes=[dict(n,owner=p['owner'][n['id']],wait_events=waits[n['id']],publish_events=publications[n['id']]) for n in g.nodes]
    receipt=dict(schema=1,structure_sha256=structure,grid=G,residency=1,cluster=1,units=g.units,
        tasks=nodes,events=events,per_track_sequence=p['queues'],
        planner={k:v for k,v in p.items() if k not in ('owner','queues')},
        audit=dict(unique_task_coverage=True,dependency_and_track_order_acyclic=True,
            head_contract='all assigned head tiles execute in one mapped Run per track; final reduction once per track; global head finish retained',
            storage_lifetimes='norm scalar and temporary GEMM output are reused only after all consumers; qkv/context/activation reuse follows dense input dependencies',
            runtime_consumption='pending generated program and binary audit'))
    write(path,receipt)

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    for file,arms in [('S_local.json',('A','C')),('S_TM.json',('B','D'))]:
        path=CONFIG/file;s=json.loads(path.read_text());g=graph_for(s);b=basic(g);tm=selected(g,b)
        for arm,p in zip(arms,(b,tm)):
            export_plan(g,p,sha(path),out/(arm+'.plan.json'))
            print(arm,len(g.nodes),p['predicted_us'],p.get('selected_candidate','basic'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out.resolve())
