#!/usr/bin/env python3
"""Reconstruct queue transitions and simulated semantic CP from raw dumps.

The simulator's critical_path_ns is its node-duration path along dependency
edges; it does not add FIFO edges or publication/wait overhead to this field.
We report the corresponding attention share, not an arithmetic saturation claim.
"""
import csv,heapq,pathlib,collections,gzip

def rows(path):
    path=pathlib.Path(path);compressed=pathlib.Path(str(path)+'.gz')
    with (gzip.open(compressed,'rt') if compressed.exists() else path.open()) as stream:
        yield from csv.DictReader(stream,delimiter='\t')

def worker_interleaving(path,grid=None):
    workers=collections.defaultdict(list)
    for row in rows(path):
        workers[int(row['worker'])].append((int(row['slot']),int(row['stage'])))
    if grid is not None:
        if any(w<0 or w>=grid for w in workers):raise ValueError('worker outside grid: '+str(path))
        for worker in range(grid):workers[worker]
    output=[]
    for worker,queue in sorted(workers.items()):
        queue.sort()
        if [slot for slot,_ in queue]!=list(range(len(queue))):
            raise ValueError('queue slots are not consecutive: '+str(path))
        adjacent=max(0,len(queue)-1)
        changes=sum(a[1]!=b[1] for a,b in zip(queue,queue[1:]))
        output.append(dict(worker=worker,tasks=len(queue),transitions=changes,
                           adjacent_slots=adjacent,interleaving=changes/adjacent if adjacent else 0))
    return output

def interleaving(path):
    workers=worker_interleaving(path)
    changes=sum(w['transitions'] for w in workers)
    adjacent=sum(w['adjacent_slots'] for w in workers)
    return dict(transitions=changes,adjacent_slots=adjacent,
                interleaving=changes/adjacent if adjacent else 0,
                tasks=sum(w['tasks'] for w in workers))

def attention_path(task_path,edge_path):
    tasks={int(r['node']):r for r in rows(task_path)}
    if set(tasks)!=set(range(len(tasks))):raise ValueError('noncontiguous task nodes')
    duration=[float(tasks[n]['end_ns'])-float(tasks[n]['start_ns']) for n in range(len(tasks))]
    if any(x<0 for x in duration):raise ValueError('negative task duration')
    successors=[[] for _ in tasks];pending=[0]*len(tasks)
    for row in rows(edge_path):
        if row['kind']!='dependency':continue
        p,c=int(row['producer']),int(row['consumer'])
        successors[p].append(c);pending[c]+=1
    ready=[n for n,v in enumerate(pending) if v==0];heapq.heapify(ready)
    distance=duration[:];pred=[-1]*len(tasks);visited=0
    while ready:
        p=heapq.heappop(ready);visited+=1
        for c in successors[p]:
            value=distance[p]+duration[c]
            if value>distance[c] or (value==distance[c] and (pred[c]<0 or p<pred[c])):
                distance[c]=value;pred[c]=p
            pending[c]-=1
            if pending[c]==0:heapq.heappush(ready,c)
    if visited!=len(tasks):raise ValueError('dependency graph has a cycle')
    end=max(range(len(tasks)),key=lambda n:(distance[n],-n));path=[];n=end
    while n>=0:path.append(n);n=pred[n]
    attention=sum(duration[n] for n in path if int(tasks[n]['attention']))
    return dict(cp_ns=distance[end],path_nodes=len(path),attention_ns=attention,
                attention_share=attention/distance[end] if distance[end] else 0)
