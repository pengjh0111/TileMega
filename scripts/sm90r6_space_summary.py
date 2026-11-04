#!/usr/bin/env python3
"""Report formal per-space domains separately from family-tied legal partitions."""
import argparse
from collections import Counter
import csv
import itertools
import json
import math
from pathlib import Path

from sm90r6_merge_analysis import sha,inputs,read


def summary(domains, partitions, same, cells=None, cache_replay=None):
    graphs={r['input']['id']:r for r in read(domains)}
    rows=[]
    for data in read(partitions):
        cell=data['input'];name=cell['id']
        if cells is not None and name not in cells:continue
        graph=graphs[name];batch=cell['batch']
        legal=[r for r in data['residencies'] if r['has_legal_program']]
        families=sorted(legal[0]['families']);axis={f:set() for f in families};axis['attention']=set()
        assignments=set()
        for residency in legal:
            for f in families:axis[f].update(residency['families'][f])
            axis['attention'].update(residency['attention_chunks'])
            for point in itertools.product(*(residency['families'][f] for f in families),residency['attention_chunks']):
                if data['moe'] and [point[families.index('expert_gate_up')],point[families.index('expert_down')]] not in residency['expert_pairs']:
                    continue
                assignments.add(point)
        count={f:len(values) for f,values in axis.items()}
        count.update(embedding=1+(batch>1),argmax=1+(batch>1),attention_merge=1)
        if data['moe']:
            count.update(route=1+len({r['grid'] for r in legal}),combine=count['expert_down'])
        nodes=[n['family'] for n in graph['seed_graph']['spaces']]
        edges={(e['producer'],e['consumer']) for e in graph['seed_graph']['edges']}
        # The union topology includes split and unsplit attention. A merge is
        # present exactly when chunks > 1; it is not an independent axis.
        for i,family in list(enumerate(nodes)):
            if family!='attention':continue
            consumers=[v for u,v in edges if u==i]
            merges=[v for v in consumers if nodes[v]=='attention_merge']
            if merges:
                for merge in merges:
                    edges.update((i,v) for u,v in list(edges) if u==merge)
            elif max(axis['attention'])>1:
                merge=len(nodes);nodes.append('attention_merge');edges.add((i,merge))
                edges.update((merge,v) for v in consumers)
        multiplicity=Counter(nodes)
        formal_product=math.prod(count[family]**amount for family,amount in multiplicity.items())
        endpoint_sum=sum(count[nodes[u]]*count[nodes[v]] for u,v in edges)
        raw=same/name/'measurements.jsonl';measurements=read(raw)
        if not measurements:raise ValueError('missing same-E measurements: '+name)
        inputs(measurements[0]['input'])
        receipt=json.loads(Path(measurements[0]['input']['receipt']).read_text())
        reuse=[r for r in measurements if r['analysis']=='reuse']
        if {r['candidate'] for r in reuse}!=set(range(len(receipt['evaluated']))) or len(reuse)!=len(receipt['evaluated']):
            raise ValueError('reuse replay has not completed: '+name)
        for candidate in receipt['evaluated']:
            c=candidate['coordinates'];point=tuple(c['gemms'][f]['tile_n'] for f in families)+(c['attention']['chunks'],)
            if point not in assignments:raise ValueError('A0 candidate outside the reported joint domain')
        misses=sum(r['derivations'] for r in reuse);hits=sum(r['coupling_hits'] for r in reuse)
        common=dict(model=cell['model'],batch=batch,evidence='verified domains and cache counters; inferred formal bounds',
            raw=str(raw),raw_sha256=sha(raw))
        def emit(metric,value,family='',instances='',values='',note=''):
            rows.append(dict(**common,metric=metric,family=family,instances=instances,value=value,
                             values=values,note=note))
        for family in sorted(count):
            if not multiplicity[family]:continue
            emit('theta_cardinality',count[family],family,multiplicity[family],
                 '|'.join(map(str,sorted(axis.get(family,[])))) or 'derived from realization or another axis',
                 'TN for GEMMs; attention chunks; ownership domains for reducers')
        emit('formal_per_space_product',formal_product,note='independent per-layer/node Cartesian upper bound; family tying and reducer dependencies constrain the legal graph set')
        emit('formal_endpoint_pair_sum',endpoint_sum,note='sum of endpoint-domain products on the union CTG topology; geometric coupling domain bound, excludes release-kind cache multiplicity')
        emit('legal_family_tied_geometry_partitions',len(assignments),note='exact TN/chunks combinations admitted by at least one residency; realization multiplicity excluded; all A0 E partitions checked')
        emit('evaluated_candidates',len(reuse))
        emit('a0_E_replay_derivations',misses,note='cold shared-cache replay of A0 E in discovery order')
        emit('a0_E_replay_hits',hits)
        emit('a0_E_replay_hit_rate',hits/(misses+hits) if misses+hits else 0)
        if cache_replay is not None:
            callback_path=cache_replay/name/'measurements.jsonl'
            callbacks=read(callback_path)
            expected=[(phase,index,candidate) for phase in ('evaluated','shortlist')
                      for index,candidate in enumerate(receipt[phase])]
            if len(callbacks)!=len(expected):raise ValueError('incomplete A0 callback replay: '+name)
            cumulative=0
            for row,(phase,index,candidate) in zip(callbacks,expected):
                if (row['input']!=cell or row['phase']!=phase or row['index']!=index or
                        row['pool_ns']!=candidate['pool_ns']):
                    raise ValueError('A0 callback replay differs from receipt: '+name)
                cumulative+=row['coupling_hits']
                if row['coupling_hits_cumulative']!=cumulative:
                    raise ValueError('cache replay cumulative hits disagree: '+name)
            if (sum(r['derivations'] for r in callbacks if r['phase']=='evaluated')!=misses or
                    sum(r['coupling_hits'] for r in callbacks if r['phase']=='evaluated')!=hits):
                raise ValueError('independent E and callback cache counters disagree: '+name)
            full_misses=sum(r['derivations'] for r in callbacks)
            common.update(raw=str(callback_path),raw_sha256=sha(callback_path))
            emit('a0_analysis_callbacks',len(callbacks),note='all unique E evaluations followed by shortlist replay; coordinate memo hits do not call analysis')
            emit('a0_search_derivations',full_misses,note='verified cold replay of the complete original callback sequence; not a new budget-limited search')
            emit('a0_search_coupling_hits',cumulative)
            emit('a0_search_coupling_hit_rate',cumulative/(full_misses+cumulative) if full_misses+cumulative else 0)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domains',type=Path,required=True)
    p.add_argument('--partitions',type=Path,required=True)
    p.add_argument('--same-E',type=Path,required=True)
    p.add_argument('--cache-replay',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    rows=summary(a.domains,a.partitions,a.same_E,cache_replay=a.cache_replay)
    if len({(r['model'],r['batch']) for r in rows})!=8:raise ValueError('space summary requires all eight cells')
    with a.out.open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(rows=len(rows),cells=8)))
