#!/usr/bin/env python3
"""Report A0 and AB1 partitions using the complete isolated family ledgers."""
import argparse
import json
from pathlib import Path
import statistics

from sm90d1_common import read,sha
from sm90r6_local_theta import geometry_key,measured_choice,traffic_choice
from sm90r6_merge_analysis import write


def selection(path,coordinate,domains):
    data=read(str(path)+'.selection.json')
    if data['a0_sha256']!=sha(coordinate) or data['domains_sha256']!=sha(domains):
        raise ValueError('theta selection input changed')
    if 'measurements' in data and data['measurements_sha256']!=sha(data['measurements']):
        raise ValueError('theta selection measurements changed')
    return read(path),{row['family']:row for row in data['choices']}


def family_rows(root,cell,domain,a0,choices):
    family=domain['family'];directory=root/'ab1/measurements-v1'/cell['input']['id']/family
    result=read(directory/'result.json')
    if (not result['passed'] or result['development'] or not result['same_card'] or
            len(result['processes'])!=3):raise ValueError('incomplete formal family measurement')
    source=directory/'measurements.jsonl'
    records=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    chosen,evidence=measured_choice(domain['geometries'],records,cell['layers'])
    traffic,traffic_evidence=traffic_choice(domain['geometries'],domain['grid'])
    if (choices['AB1a'][family]!={**dict(family=family,grid=domain['grid']),**chosen,**evidence} or
            choices['AB1b'][family]!={**dict(family=family,grid=domain['grid']),**traffic,**traffic_evidence}):
        raise ValueError('retained theta is not reproduced by the formal selection rules')
    guard_paths=set();raw_paths=set();process_sets={};times={}
    for row in records:
        guard_paths.add(row['guard']);raw_paths.add(row['raw'])
        if sha(row['raw'])!=row['raw_sha256'] or read(row['guard'])['code']:
            raise ValueError('unaccepted or changed isolated ledger')
        key=geometry_key(row['geometry'])
        process_sets.setdefault(key,set()).add(row['process_id'])
        times.setdefault(key,[]).append(statistics.median(row['stage_ns']))
    if len({tuple(sorted(value)) for value in process_sets.values()})!=1 or len(guard_paths)!=3:
        raise ValueError('family geometries did not share three fresh processes')
    by_geometry={geometry_key(row['geometry']):row for row in domain['geometries']}
    selected={'A0':a0['gemms'][family],'AB1a':chosen['geometry'],'AB1b':traffic['geometry']}
    rows=[]
    for arm,geometry in selected.items():
        key=geometry_key(geometry);entry=by_geometry[key]
        rows.append(dict(model=cell['input']['model'],batch=cell['input']['batch'],family=family,arm=arm,
            tile_n=geometry['tile_n'],tile_k=geometry.get('tile_k',geometry['slot_bytes']//(2*geometry['tile_n'])),
            warps_n=geometry['warps_n'],warps_k=geometry['warps_k'],slots=geometry['slots'],
            register_stages=geometry.get('register_stages',1),multicast=int(geometry.get('multicast',False)),
            mma=geometry['mma'],tasks=entry['tasks'],traffic_bytes=entry['traffic_bytes'],
            isolated_us=statistics.median(times[key])/1000,gpu=evidence['gpu'],processes=3,
            instances_per_process=cell['layers'],grid=domain['grid'],
            task_window_exception=(traffic_evidence['exception'] or '') if arm=='AB1b' else '',
            window_distance=traffic_evidence['window_distance'] if arm=='AB1b' else '',
            reference_sha256=evidence['reference_sha256'],raw=str(source.relative_to(root)),raw_sha256=sha(source)))
    return rows


def merge(root,measured_root):
    domains=root/'ab5/domains-v1/domains.jsonl';rows=[];seen=set()
    for line in domains.read_text().splitlines():
        if not line.strip():continue
        cell=json.loads(line);name=cell['input']['id']
        if name in seen:raise ValueError('duplicate domain cell')
        seen.add(name)
        version='psel-v2' if name=='olmoe_1b_7b-B16' else 'psel-v1'
        winner=read(root/'a0'/version/name/'result.json')
        binary=root/'a0/finalists-v1'/name/('candidate-'+str(winner['winner_rank']))/'decode.so'
        coordinate=binary.with_name('coordinates.json');a0=read(coordinate)
        if sha(binary)!=winner['winner_binary_sha256']:raise ValueError('A0 binary changed')
        choices={}
        for arm,path in [('AB1a',measured_root/name/'theta.json'),('AB1b',root/'ab1/traffic-v1'/name/'theta.json')]:
            selected,choices[arm]=selection(path,coordinate,domains)
            if any(selected['gemms'][family]!=choice['geometry'] for family,choice in choices[arm].items()):
                raise ValueError('theta coordinate disagrees with its selection receipt')
        expected=set(a0['gemms'])
        if any(set(values)!=expected for values in choices.values()):raise ValueError('incomplete family selection')
        for domain in cell['domains']:
            if domain['family'] in expected and domain['residency']==a0['residency']:
                values=family_rows(root,cell,domain,a0,choices)
                if any(row['reference_sha256']!=sha(binary) for row in values):
                    raise ValueError('isolated operands do not come from selected A0')
                rows.extend(values)
    expected={model+'-B'+str(batch) for model in ('qwen3_1_7b','llama3_1_8b','qwen3_30b_a3b','olmoe_1b_7b') for batch in (1,16)}
    if seen!=expected or len(rows)!=132:raise ValueError('incomplete eight-cell AB1 theta table')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--measured-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=merge(a.root,a.measured_root);write(a.out,rows)
    print(json.dumps(dict(rows=len(rows),cells=8,families=44)))
