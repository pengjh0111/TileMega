#!/usr/bin/env python3
"""Losslessly compact the accepted schema1 forward calibration into schema2."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from dnn1_constants import append_forward_section


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert(source, table, *, allow_subset=False):
    original=json.loads(source.read_text())
    receipt=source.parent/'collection_receipt.json'
    proof=json.loads(receipt.read_text())
    if original['schema']!=1 or (original['completeness']!='complete' and not allow_subset) or original.get('synthetic_cpu_fixture') or original.get('pending'):
        raise ValueError('complete genuine schema1 calibration required')
    if proof['status']!='collected':raise ValueError('accepted collection receipt required')
    root=Path(original['raw_measurement_root']);spec=root/'queue-spec.json'
    if sha(spec)!=proof['source_spec_sha256']:raise ValueError('raw collection specification changed')
    cards=original['by_gpu']
    if set(cards)!={'0','5','7'}:raise ValueError('all three physical cards required')
    # Bind the conversion to the actual accepted measurements, not merely two
    # mutually consistent derived files. No GPU operation is performed here.
    samples={};sessions={};sample_groups={}
    for evidence in proof['evidence']:
        result_path=Path(evidence['result']);session_path=Path(evidence['session'])
        if sha(result_path)!=evidence['result_sha256']:raise ValueError('measurement receipt changed')
        if session_path not in sessions:
            session=json.loads(session_path.read_text())
            guard=json.loads((session_path.parents[1]/'guard_result.json').read_text())
            identity=session_path.parent/'physical-device-identity.json'
            if (session['status']!='complete' or guard['code']!=0 or
                    sha(identity)!=evidence['physical_device_identity_sha256']):
                raise ValueError('unaccepted or changed physical session')
            sessions[session_path]=session
        session=sessions[session_path];result=json.loads(result_path.read_text())
        if not result['valid'] or not result['fresh_executable_process']:
            raise ValueError('invalid measurement receipt')
        for name in ('raw','stdout','stderr'):
            if sha(result[name])!=result[name+'_sha256']:raise ValueError('raw measurement changed')
        for r in result['rows']:
            key=(str(session['gpu']),r['metric'],r['key'])
            samples.setdefault(key,[]).append((result['process_pid'],float(r['median_ns'])))
            sample_groups.setdefault(key,set()).add(result['case']['binary'])
    keys=sorted((metric,key,row['unit']) for metric,values in cards['0'].items() for key,row in values.items())
    if any(sorted((m,k,r['unit']) for m,vs in card.items() for k,r in vs.items())!=keys for card in cards.values()):
        raise ValueError('physical cards contain different constants')
    csv_rows=list(csv.DictReader(table.open()))
    csv_map={(r['gpu'],r['metric'],r['key']):r for r in csv_rows}
    if len(csv_rows)!=len(csv_map) or len(csv_map)!=len(keys)*3:raise ValueError('raw CSV coverage mismatch')
    medians={g:[] for g in cards};clocks={g:{} for g in cards};rows=[]
    for index,(metric,key,unit) in enumerate(keys):
        row={'constant_id':index};clock_group=None
        for gpu in ('0','5','7'):
            value=cards[gpu][metric][key];raw=csv_map[(gpu,metric,key)]
            processes=samples.get((gpu,metric,key),[])
            if (len(processes)!=5 or len({pid for pid,v in processes})!=5 or
                    statistics.median(v for pid,v in processes)!=value['median']):
                raise ValueError('derived median does not match five fresh raw processes')
            if value['processes']!=5 or not math.isfinite(value['median']) or value['median']<=0:
                raise ValueError('five fresh positive finite measurements required')
            for field in ('median','p99','process_range','relative_range'):
                if float(raw[field])!=value[field]:raise ValueError('raw CSV/JSON mismatch')
            if value['unstable']!=(value['relative_range']>.10):raise ValueError('unstable flag mismatch')
            groups=sample_groups[(gpu,metric,key)]
            if len(groups)!=1 or not groups.issubset({'direct','segments'}):
                raise ValueError('measurement binary group differs across processes')
            group=next(iter(groups))
            if clock_group is not None and group!=clock_group:raise ValueError('clock grouping mismatch')
            clock_group=group
            pair={field:raw[field] for field in ('start_sm_clocks_mhz','end_sm_clocks_mhz')}
            if group in clocks[gpu] and clocks[gpu][group]!=pair:raise ValueError('clock metadata differs within binary group')
            clocks[gpu][group]=pair
            medians[gpu].append(value['median'])
            row['gpu'+gpu+'_median']=value['median'];row['gpu'+gpu+'_range']=value['process_range']
        row['clock_group']=clock_group;rows.append(row)
    compact={k:v for k,v in original.items() if k!='by_gpu'}
    compact.update(schema=2,processes=5,keys=keys,median_by_gpu=medians,clock_groups=clocks,
        csv_schema='constant_id indexes keys; GPU columns retain exact medians and absolute process ranges; clock_group indexes clock_groups[gpu]',
        raw_sources={p.name:{'path':str(p),'sha256':sha(p)} for p in (source,table,receipt)},
        statistics_source='p99 and full per-process samples remain in SHA-bound raw sources; no rounding/interpolation/fitting')
    assert all(compact['median_by_gpu'][g][i]==cards[g][m][k]['median'] for g in cards for i,(m,k,u) in enumerate(keys))
    return compact,rows



def supplement(compact, rows, source):
    extra, added = convert(source, source.parent/'DNN1_constants.csv', allow_subset=True)
    old_keys={tuple(key) for key in compact['keys']}
    if any(tuple(key) in old_keys for key in extra['keys']):
        raise ValueError('supplements must never replace an existing measured key')
    offset=len(compact['keys']);number=len(compact.get('supplements',[]))+1
    before={gpu:list(values) for gpu,values in compact['median_by_gpu'].items()}
    compact['keys'].extend(extra['keys'])
    for gpu in ('0','5','7'):
        compact['median_by_gpu'][gpu].extend(extra['median_by_gpu'][gpu])
        for group,clocks in extra['clock_groups'][gpu].items():
            compact['clock_groups'][gpu][f'supplement{number}/{group}']=clocks
        assert compact['median_by_gpu'][gpu][:offset]==before[gpu]
    for row in added:
        row['constant_id']+=offset
        row['clock_group']=f'supplement{number}/'+row['clock_group']
    rows.extend(added);compact['unstable'].extend(extra['unstable'])
    metadata={k:v for k,v in extra.items() if k not in (
        'schema','keys','median_by_gpu','clock_groups','unstable','historical_reference',
        'csv_schema','statistics_source','omitted_ready','resource_infeasible')}
    if metadata['header_sha256']==compact['header_sha256']:
        del metadata['header_sha256'];metadata['header_source']='base header_sha256; exact dictionary equality verified'
    metadata['key_indices']=list(range(offset,len(compact['keys'])))
    metadata['scope']='only newly required exact geometries; existing medians immutable'
    compact.setdefault('supplements',[]).append(metadata)
    return compact,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True)
    p.add_argument('--supplement',type=Path,action='append',default=[],help='accepted subset collection; append new exact keys only')
    p.add_argument('--csv',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();before=a.target.read_bytes();constants,rows=convert(a.source,a.source.parent/'DNN1_constants.csv')
    for source in a.supplement:constants,rows=supplement(constants,rows,source)
    result=append_forward_section(a.target,constants)
    a.csv.parent.mkdir(parents=True,exist_ok=True)
    with a.csv.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    after=a.target.read_bytes();prefix=before.rstrip()[:-1].rstrip()
    assert after.startswith(prefix)
    proof=dict(result,entries=len(rows)*3,rows=len(rows),exact_medians=True,original_target_prefix_unchanged=True,
        section_bytes=len(json.dumps(constants,separators=(',',':')).encode()),csv_bytes=a.csv.stat().st_size,
        target_sha256=sha(a.target),csv_sha256=sha(a.csv),raw_sha256=sha(a.source))
    a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
