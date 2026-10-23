#!/usr/bin/env python3
"""Share identical immutable C3 payload storage; preserve every sample path."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import uuid

from as1_cleanup import file_stamp, plain_path, process_alive, project_paths


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def link_identical(source,target,roots,source_stamp,target_stamp,sha):
    source=plain_path(source,roots);target=plain_path(target,roots)
    if file_stamp(source)!=source_stamp or file_stamp(target)!=target_stamp:
        raise ValueError('Artifact identity changed')
    if source.stat().st_dev!=target.stat().st_dev or digest(source)!=sha or digest(target)!=sha:
        raise ValueError('Payload mismatch or filesystem boundary')
    if file_stamp(source)!=source_stamp or file_stamp(target)!=target_stamp:
        raise ValueError('Artifact changed while hashing')
    temp=target.with_name('.as2-dedup-'+uuid.uuid4().hex)
    try:
        os.link(source,temp)
        if file_stamp(source)!=source_stamp or file_stamp(target)!=target_stamp:
            raise ValueError('Artifact changed before replacement')
        os.replace(temp,target)
    finally:
        if temp.exists():temp.unlink()
    if not os.path.samefile(source,target) or digest(target)!=sha:
        raise RuntimeError('Deduplicated artifact verification failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();runs,_,_=project_paths('as2');roots=[runs]
    audit=plain_path(a.audit,roots);out=plain_path(a.out,roots)
    if out.exists():raise FileExistsError(out)
    proof=json.loads(audit.read_text())
    if proof['status']!='pass' or proof['processes_passed']<5 or not proof['payload_kv_tokens_controls_bitwise_equal']:
        raise ValueError('Require a completed five-process request C3 audit')
    manifest=runs/'manifest.jsonl';registered={}
    for line in manifest.read_text().splitlines():
        entry=json.loads(line);registered[entry['path']]=entry
    directories=[]
    for record in proof['records']:
        job=plain_path(Path(record['job'])/'job.json',roots);receipt=json.loads(job.read_text())
        if receipt['project']!='as2' or receipt['status']!='passed' or receipt['executable_sha256']!=proof['binary_sha256']:
            raise ValueError('C3 job identity changed')
        if process_alive(dict(pid=receipt['pid'],start_ticks=receipt['start_ticks'])):
            raise ValueError('Original producer is still alive')
        if receipt['command'][1]!='request':raise ValueError('Require request artifacts')
        root=plain_path(Path(receipt['command'][3]),roots)
        directories.extend(sorted(root.glob('request_*')))
    canonical={};items=[]
    for directory in directories:
        for target in sorted(directory.rglob('*.bin')):
            if not target.stem.isdecimal() or target.stat().st_size<1<<20:continue
            target=plain_path(target,roots);entry=registered.get(str(target))
            if not entry or not entry.get('completed') or 'active_refs' not in entry or any(process_alive(r) for r in entry['active_refs']):
                raise ValueError('Missing completed manifest entry or active reference: '+str(target))
            stamp=file_stamp(target)
            if stamp!=entry['stamp']:raise ValueError('Manifest identity changed: '+str(target))
            key=str(target.relative_to(directory));sha=digest(target)
            if file_stamp(target)!=stamp:raise ValueError('Artifact changed while hashing')
            if key not in canonical:
                canonical[key]=(target,stamp,sha);continue
            source,source_stamp,expected=canonical[key]
            if sha!=expected:raise ValueError('C3 payload no longer matches: '+str(target))
            if os.path.samefile(source,target):continue
            reclaim=target.stat().st_blocks*512 if target.stat().st_nlink==1 else 0
            if a.apply:
                link_identical(source,target,roots,source_stamp,stamp,sha)
                updated=dict(entry,stamp=file_stamp(target),deduplicated_from=str(source),sha256=sha)
                with manifest.open('a') as f:f.write(json.dumps(updated)+'\n')
            items.append(dict(source=str(source),target=str(target),sha256=sha,
                bytes=stamp['size'],previous_unique_allocated_bytes=reclaim))
    result=dict(schema='tilemega.as2.c3_dedup.v1',mode='apply' if a.apply else 'dry-run',
        time=dt.datetime.now(dt.timezone.utc).isoformat(),audit=str(audit),audit_sha256=digest(audit),
        changed_files=len(items) if a.apply else 0,eligible_files=len(items),
        previous_unique_allocated_bytes=sum(i['previous_unique_allocated_bytes'] for i in items),
        paths_and_payloads_preserved=True,items=items)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='items'}))


if __name__=='__main__':main()
