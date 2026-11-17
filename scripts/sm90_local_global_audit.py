#!/usr/bin/env python3
"""Static task coverage, event-order and actual emitted-program audit."""
import argparse
import heapq
import json
from pathlib import Path
import subprocess
from sm90_local_global_local import ROOT, CONFIG, sha, write
from sm90_local_global_emit import metadata, aliases, arrays
from sm90_local_global_plan import graph_for, basic, selected, parents

def ordering(plan,graph):
    nodes=plan['tasks'];n=len(nodes);deps=[set() for _ in nodes]
    for node in nodes:
        for e in node['wait_events']:deps[node['id']].update(plan['events'][e]['producers'])
    for q in plan['per_track_sequence']:
        for a,b in zip(q,q[1:]):deps[b].add(a)
    succ=[[] for _ in nodes];degree=[len(d) for d in deps]
    for t,ds in enumerate(deps):
        for d in ds:succ[d].append(t)
    ready=[t for t in range(n) if not degree[t]];heapq.heapify(ready)
    ancestors=[0]*n;logical=parents(graph);visited=0
    while ready:
        t=heapq.heappop(ready);bits=0
        for p in deps[t]:bits|=ancestors[p]|(1<<p)
        for p in logical[t]:
            if not (bits>>p)&1:raise ValueError(f'unenforced data dependency {p}->{t}')
        ancestors[t]=bits;visited+=1
        for c in succ[t]:
            degree[c]-=1
            if degree[c]==0:heapq.heappush(ready,c)
    assert visited==n
    assert sorted(t for q in plan['per_track_sequence'] for t in q)==list(range(n))
    for i,e in enumerate(plan['events']):
        assert e['count']==len(e['producers'])==len(set(e['producers']))
        assert sorted(e['producers'])==[t for t in range(n) if i in nodes[t]['publish_events']]
    return dict(tasks=n,events=len(plan['events']),all_logical_dependencies_enforced=True,unique_coverage=True)

def main(plans,build,out):
    records={};structures={}
    for name in ('S_local','S_TM'):
        s=json.loads((CONFIG/(name+'.json')).read_text());structures[name]=s
        if name=='S_local':assert s['whole_program_measurements_at_freeze']==0 and s['measured_candidates']==196
        g=graph_for(s);b=basic(g)
        selected(g,b,out/(name+'.frontier-candidate.json'))
        chunks=s['coordinates']['attention']['chunks']
        for past in range(64,1087):
            bounds=[min((past+1)*i//chunks,past) for i in range(chunks)]+[past+1]
            assert bounds[0]==0 and bounds[-1]==past+1 and all(x<=y for x,y in zip(bounds,bounds[1:]))
    for arm in 'ABCDE':
        name='S_local' if arm in 'ACE' else 'S_TM';s=structures[name]
        plan_path=plans/(('A' if arm=='E' else arm)+'.plan.json');plan=json.loads(plan_path.read_text())
        assert plan['structure_sha256']==sha(CONFIG/(name+'.json'))
        r=ordering(plan,graph_for(s));text=(build/(arm+'.so.cu')).read_text()
        prefix,_=metadata(s);assert text.startswith(prefix)
        assert all(line in text for line in aliases(s))
        emitted,receipt=arrays(plan);assert all(line in text for line in emitted)
        assert receipt['unique_coverage'] and receipt['head_tasks']==len(plan['units'][-1]['tasks'])
        if arm in 'AB':
            assert plan['planner']['algorithm']=='basic_rr_topological_v1'
            assert all(n['owner']==n['coordinate']%132 for n in plan['tasks'])
        if arm in 'CD':assert plan['planner']['algorithm']=='critical_frontier_earliest_finish_v1'
        # All selected plans here retain aggregate boundaries; these ensure
        # prior unit consumers finish before norm/temp/context storage reuse.
        assert plan['planner']['aggregate']
        compiled=json.loads((build/(arm+'.build.json')).read_text());assert compiled['code']==0
        binary=build/(arm+'.so');assert sha(binary)==compiled['binary_sha256']
        assert sha(build/(arm+'.so.cu'))==compiled['source_sha256']
        symbols=subprocess.check_output(['nm','-C',str(binary)],text=True)
        assert 'Sm90DenseExplicitTasks' in symbols and 'static_entries' in symbols
        assert ('sm90_static_sequence_unit' in symbols)==(arm=='E')
        assert ('sm90_dense_native_kernel' in symbols)==(arm!='E')
        assert ('cudaGraphLaunch' in symbols)==(arm=='E')
        r.update(binary_sha256=sha(binary),source_sha256=compiled['source_sha256'],plan_sha256=sha(plan_path),
            structure_sha256=sha(CONFIG/(name+'.json')),gpu_queue_tables_in_emitted_source=True,
            explicit_task_coordinate_binary_symbols=True,aggregate_lifetime_boundaries=True,
            head_finish_once_per_track=True,execution=receipt)
        records[arm]=r;print(arm,r['tasks'],'audited',flush=True)
    assert records['A']['structure_sha256']==records['C']['structure_sha256']==records['E']['structure_sha256']
    assert records['B']['structure_sha256']==records['D']['structure_sha256']
    # The complete program (not only coordinates) matches for the organization pair.
    a=(build/'A.so.cu').read_text().split('#include <tilemega/Codegen/tasks/Sm90DenseRuntime.cuh>')[0]
    e=(build/'E.so.cu').read_text().split('#define TILEMEGA_NATIVE_KERNEL_SEQUENCE')[0]
    assert a==e
    result=dict(evidence='verified static enumeration, generated source and binary symbol checks; not numerical correctness validation',
        arms=records,paired_structure_signatures=True,AE_identical_task_program=True,
        body_headers={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'include/tilemega').rglob('Sm90Dense*') if p.is_file()},
        dynamic_correctness_tests=0,storage_scope='selected aggregate plans; no unsupported arbitrary-plan race claim')
    write(out/'consistency.json',result)
    prefill=Path('/root/TileMega/runs/sm90r2/build/sm90r3-final-prefill/qwen3_1_7b-B1/prefill.so')
    assert sha(prefill)=='6f6740f70e5e89f439b6eb8466df0372aaa7f100c8583953eb4c1055e95079f8'
    spec=dict(schema=1,model='qwen3_1_7b',batch=1,dtype='BF16',input_tokens=64,output_tokens=1024,
        guard_policy=str(ROOT/'configs/serving/sm90_local_global_ae/guard_policy.json'),
        body_version_resolution='Frozen S_TM semantics with explicit coordinate interface and common static executor; D_test newly measured',
        consistency_sha256=sha(out/'consistency.json'),arms={arm:dict(decode=str(build/(arm+'.so')),
            decode_sha256=records[arm]['binary_sha256'],prefill=str(prefill),prefill_sha256=sha(prefill)) for arm in 'ABCDE'})
    write(out/'spec.json',spec)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plans',type=Path,required=True)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);main(a.plans.resolve(),a.build.resolve(),a.out.resolve())
