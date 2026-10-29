#!/usr/bin/env python3
"""Share immutable output storage; preserve every sample path and payload."""
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


def completed_request_directories(jobs,roots,expected_binary=None):
    if not 1<=len(jobs)<=8:raise ValueError('Require one to eight completed request jobs')
    directories=[];identities=set();common=None
    for directory in jobs:
        job=plain_path(Path(directory)/'job.json',roots);receipt=json.loads(job.read_text())
        command=receipt['command']
        if (receipt['project']!='as2' or receipt['status']!='passed' or receipt.get('exit_code')!=0 or
                receipt.get('device_locked') is not False or len(command)!=8 or command[1]!='request'):
            raise ValueError('Require completed, successful AS2 request receipts')
        if expected_binary and receipt['executable_sha256']!=expected_binary:
            raise ValueError('Request binary identity changed')
        identity=(receipt['pid'],receipt['start_ticks'])
        if identity in identities:raise ValueError('Duplicate producer identity')
        identities.add(identity)
        if process_alive(dict(pid=receipt['pid'],start_ticks=receipt['start_ticks'])):
            raise ValueError('Original producer is still alive')
        binding=plain_path(Path(command[2]),roots)
        key=(str(binding),receipt['executable_sha256'])
        if common is not None and key!=common:raise ValueError('Mixed bindings or binaries')
        common=key;root=plain_path(Path(command[3]),roots)
        outputs,warmup,repetitions,diagnostic=map(int,command[4:])
        if not 2<=outputs<=1024 or warmup<0 or repetitions<1 or warmup+repetitions>16 or diagnostic not in (0,1,2):
            raise ValueError('Invalid completed request scope')
        for repetition in range(-warmup,repetitions):
            path=plain_path(root/('request_'+str(repetition)),roots)
            if not path.is_dir():raise ValueError('Missing completed request output')
            directories.append(path)
    if len(directories)>16:raise ValueError('Bounded storage sharing batch required')
    return directories


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--audit',type=Path)
    source.add_argument('--completed-jobs',type=Path,nargs='+')
    p.add_argument('--apply',action='store_true')
    p.add_argument('--min-bytes',type=int,default=1<<20)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();runs,_,_=project_paths('as2');roots=[runs]
    if a.min_bytes<65536:p.error('Deduplication is limited to numeric payloads of at least64KiB')
    out=plain_path(a.out,roots);audit=plain_path(a.audit,roots) if a.audit else None
    if out.exists():raise FileExistsError(out)
    if audit:
        proof=json.loads(audit.read_text())
        if proof['status']!='pass' or proof['processes_passed']<5 or not proof['payload_kv_tokens_controls_bitwise_equal']:
            raise ValueError('Require a completed five-process request C3 audit')
        jobs=[Path(record['job']) for record in proof['records']]
        directories=completed_request_directories(jobs,roots,proof['binary_sha256'])
    else:
        jobs=[plain_path(path,roots) for path in a.completed_jobs]
        directories=completed_request_directories(jobs,roots)
    manifest=runs/'manifest.jsonl';registered={}
    for line in manifest.read_text().splitlines():
        entry=json.loads(line);registered[entry['path']]=entry
    canonical={};items=[];different=0
    for directory in directories:
        for target in sorted(directory.rglob('*.bin')):
            if target.name in ('trace.bin','profile.bin') or target.stat().st_size<a.min_bytes:continue
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
            if sha!=expected:
                if audit:raise ValueError('C3 payload no longer matches: '+str(target))
                different+=1;continue
            if os.path.samefile(source,target):continue
            reclaim=target.stat().st_blocks*512 if target.stat().st_nlink==1 else 0
            if a.apply:
                link_identical(source,target,roots,source_stamp,stamp,sha)
                updated=dict(entry,stamp=file_stamp(target),deduplicated_from=str(source),sha256=sha)
                with manifest.open('a') as f:f.write(json.dumps(updated)+'\n')
            items.append(dict(source=str(source),target=str(target),sha256=sha,
                bytes=stamp['size'],previous_unique_allocated_bytes=reclaim))
    result=dict(schema='tilemega.as2.c3_dedup.v1' if audit else 'tilemega.as2.completed_request_storage.v1',mode='apply' if a.apply else 'dry-run',
        scope='Byte-preserving storage sharing; no new numerical, synchronization or performance admission',
        time=dt.datetime.now(dt.timezone.utc).isoformat(),audit=str(audit) if audit else None,
        audit_sha256=digest(audit) if audit else None,completed_jobs=[str(p) for p in jobs],different_payloads_retained=different,
        min_payload_bytes=a.min_bytes,
        changed_files=len(items) if a.apply else 0,eligible_files=len(items),
        previous_unique_allocated_bytes=sum(i['previous_unique_allocated_bytes'] for i in items),
        paths_and_payloads_preserved=True,items=items)
    out.write_text(json.dumps(result,indent=2)+'\n')
    if a.apply:
        summary={k:v for k,v in result.items() if k not in ('items','completed_jobs')}
        summary.update(operation='share_immutable_completed_payloads',report=str(out))
        with (runs/'cleanup_history.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='items'}))


if __name__=='__main__':main()
