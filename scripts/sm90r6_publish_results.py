#!/usr/bin/env python3
"""Export compact result CSVs with immutable, full-precision canonical copies."""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal,InvalidOperation
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    if not any(c in value for c in '.eE'):return value
    try:d=Decimal(value)
    except InvalidOperation:return value
    if not d.is_finite():return value
    if d==d.to_integral_value():return format(d.to_integral_value(),'f')
    text=format(d,'.9g')
    if 'e' not in text.lower() and '.' in text:text=text.rstrip('0').rstrip('.')
    return text


def compact(rows,name,root):
    original=list(rows[0])
    if 'record' in original or 'measured_record' in original:
        raise ValueError('compact input cannot replace its full canonical evidence')
    if name=='SM90R6_hw_bound.csv':
        grouped=defaultdict(dict)
        for index,row in enumerate(rows,2):
            key=tuple(row[k] for k in ('system','model','tp','batch'))
            if row['basis'] in grouped[key]:raise ValueError('duplicate hardware-bound basis')
            grouped[key][row['basis']]=(index,row)
        result=[]
        same=['system','model','tp','batch','bytes_step','flops_step','tpot_ms',
              'communication_bound_ms','timing_source','timing_acceptance','timing_conditions']
        for parts in grouped.values():
            if set(parts)!={'measured','nominal'}:raise ValueError('both hardware-bound bases are required')
            i,m=parts['measured'];j,n=parts['nominal']
            if any(m[k]!=n[k] for k in same):raise ValueError('hardware-bound bases disagree on the measured cell')
            row={k:m[k] for k in same}
            row['timing_source']=row['timing_source'].removeprefix(str(root)+'/')
            row.update(bound_measured_ms=m['bound_ms'],ratio_measured=m['bound_over_measured'],
                bound_nominal_ms=n['bound_ms'],ratio_nominal=n['bound_over_measured'],
                measured_record=str(i),nominal_record=str(j))
            result.append(row)
        mapping='both bases in one row; intermediate bound terms and complete provenance remain in canonical records'
    else:
        drop={k for k in original if k.endswith('_sha256') or k.startswith('raw') or
              k in ('note','evidence','budget_raw','search_receipt','measurement_raw','pair_raw','input_state')}
        if name in ('SM90R6_correctness.csv','SM90R6_decisions.csv','SM90R6_hw_support.csv'):drop=set()
        drop.discard('candidate_sha256');result=[];prefixes={}
        for index,row in enumerate(rows,2):
            value={k:v for k,v in row.items() if k not in drop}
            if 'candidate_sha256' in value:
                full=value.pop('candidate_sha256');prefix=full[:16]
                if prefix in prefixes and prefixes[prefix]!=full:raise ValueError('candidate SHA prefix collision')
                prefixes[prefix]=full;value['candidate_id']=prefix
            value['record']=str(index);result.append(value)
        mapping=dict(archived_columns=sorted(drop),candidate_id='unique 16-hex prefix; full SHA remains in the canonical record')
    for row in result:
        for key,value in row.items():
            if not any(part in key for part in ('sha256','source','raw','candidate','note','config','checkpoint')):
                row[key]=number(value)
    return result,mapping


def publish(source,out,root):
    out.mkdir(parents=True,exist_ok=False)
    archive=out/'canonical';archive.mkdir();compact_dir=out/'compact';compact_dir.mkdir()
    manifest=dict(schema=1,precision='nine significant digits for nonintegral numeric display; decisions computed before export; canonical values unchanged',
        record='1-based CSV record including the header as record 1; not a physical text line',files={})
    for path in sorted(source.glob('SM90R6*.csv')):
        with path.open() as stream:rows=list(csv.DictReader(stream))
        if not rows:raise ValueError('empty result table: '+str(path))
        destination=archive/path.name;shutil.copyfile(path,destination)
        values,mapping=compact(rows,path.name,root)
        compact_path=compact_dir/path.name
        with compact_path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,list(values[0]),lineterminator='\n');writer.writeheader();writer.writerows(values)
        manifest['files'][path.name]=dict(canonical=str(destination.resolve()),canonical_sha256=sha(destination),
            canonical_rows=len(rows),rows=len(values),compact_sha256=sha(compact_path),mapping=mapping)
    if not manifest['files']:raise ValueError('no canonical R6 result CSVs')
    (out/'SM90R6_result_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return dict(files=len(manifest['files']),canonical_bytes=sum(p.stat().st_size for p in archive.iterdir()),
        compact_bytes=sum(p.stat().st_size for p in compact_dir.iterdir())+(out/'SM90R6_result_provenance.json').stat().st_size)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args();print(json.dumps(publish(a.source,a.out,a.root)))
