#!/usr/bin/env python3
"""Join elided LA nodes into the runtime DAG and retain temporal overlaps."""
import re,statistics
from collections import defaultdict
from pathlib import Path
from ledger import read

def chain(folder):
    slots=[{k:int(v) for k,v in r.items()} for r in read(folder/'slots.tsv') if int(r['run_begin'])]
    nodes={(r['stage'],r['logical_task']):dict(begin=r['wait_begin'],end=r['publish_end'],worker=r['worker'],virtual=False) for r in slots}
    for r in read(folder/'reducers.tsv') if (folder/'reducers.tsv').exists() else []:
        r={k:int(v) for k,v in r.items()};owner=nodes.get((r['producer_stage'],r['producer_task']))
        nodes[(r['stage'],r['task'])]=dict(begin=owner['begin'] if owner else r['publish_ns'],end=r['publish_ns'],worker=r['worker'],virtual=True,owner=(r['producer_stage'],r['producer_task']))
    stages={int(r['stage']):r for r in read(folder/'runtime_stages.tsv')}
    successors=defaultdict(set);predecessors=defaultdict(set)
    text=(folder/'runtime_dependencies.cuh').read_text()
    edges=re.findall(r'\{(\d+)u,\s*(\d+)u, StageDependency::Map::(\w+),\s*(\d+)u,\s*(-?\d+),\s*(-?\d+),\s*(\d+)u',text)
    for ps,cs,map_,div,scale,offset,count in edges:
        ps,cs,div,scale,offset,count=map(int,(ps,cs,div,scale,offset,count))
        for consumer in [n for n in nodes if n[0]==cs]:
            lo=0 if map_ in ('kAll','kPhase') else (consumer[1]//div)*scale+offset
            hi=int(stages[ps]['active_tasks']) if map_ in ('kAll','kPhase') else lo+count
            for producer in [(ps,t) for t in range(max(0,lo),min(hi,int(stages[ps]['active_tasks']))) if (ps,t) in nodes]:
                successors[producer].add(consumer);predecessors[consumer].add(producer)
    # Worker FIFO is also a dependency. LA executes on its owner's slot and
    # must not be inserted as an independent later queue slot.
    queues=defaultdict(list)
    for r in slots:queues[r['worker']].append(r)
    for rows in queues.values():
        rows.sort(key=lambda r:r['slot'])
        for a,b in zip(rows,rows[1:]):
            source=(a['stage'],a['logical_task']);dest=(b['stage'],b['logical_task'])
            predecessors[dest].add(source);successors[source].add(dest)
    end=max(nodes,key=lambda n:nodes[n]['end']);path=[];seen=set()
    while end not in seen:
        seen.add(end);path.append(end)
        sources=predecessors[end]
        if not sources:break
        end=max(sources,key=lambda n:nodes[n]['end'])
    path.reverse();links=[];previous=None
    for node in path:
        r=nodes[node];links.append(dict(stage=node[0],task=node[1],worker=r['worker'],virtual=r['virtual'],
            begin_ns=r['begin'],complete_ns=r['end'],boundary_ns=r['begin']-previous if previous is not None else None))
        previous=r['end']
    return links,dict(nodes=len(nodes),chain_links=len(path),chain_span_ns=nodes[path[-1]]['end']-nodes[path[0]]['begin'],
        virtual_reducers=sum(n['virtual'] for n in nodes.values()),
        phase_graph='full host edges; negative boundaries denote overlapped K-phase work, not negative execution time')

def pages(path,byte_count,ceilings):
    grouped=defaultdict(list)
    for r in read(path):grouped[int(r['step'])].append({k:int(v) for k,v in r.items()})
    output=[]
    for step,workers in grouped.items():
        workers=[r for r in workers if r['kernel_end_ns']>r['kernel_begin_ns']>0]
        if not workers:continue
        span=max(r['kernel_end_ns'] for r in workers)-min(r['kernel_begin_ns'] for r in workers)
        row=dict(source=str(path),step=step,span_ns=span,bytes=byte_count,workers=len(workers),
            dependency_mean_ns=statistics.mean(r['dependency_wait_ns'] for r in workers),
            page_full_mean_ns=statistics.mean(r['page_full_ns'] for r in workers),
            full_and_dependency_mean_ns=statistics.mean(r['full_and_wait_ns'] for r in workers))
        if all('loader_issue_ns' in r for r in workers):
            row['loader_issue_fraction']=statistics.mean(r['loader_issue_ns']/(r['kernel_end_ns']-r['kernel_begin_ns']) for r in workers)
        for key,bandwidth in ceilings.items():row['span_over_floor_'+key]=span/(byte_count/bandwidth)
        output.append(row)
    return output
