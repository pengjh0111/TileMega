#!/usr/bin/env python3
"""Losslessly retain completed profile payloads before manifest-based cleanup."""
import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from as1_cleanup import file_stamp,plain_path,process_alive,project_paths


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile-summary',type=Path,nargs='+',default=[])
    p.add_argument('--schedule',type=Path,nargs='+',default=[])
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    if not 1<=len(a.profile_summary)+len(a.schedule)<=16:p.error('Bounded archive batch required')
    runs,work,roots=project_paths('as2');manifest=runs/'manifest.jsonl'
    latest={v['path']:v for v in map(json.loads,manifest.read_text().splitlines())}
    targets={}
    for summary in a.profile_summary:
        summary=plain_path(summary.absolute(),[runs]);value=json.loads(summary.read_text())
        if value['schema']!='tilemega.as2.native_body_profile.v1' or value['status']!='measured_body_only':
            raise ValueError('Require already parsed, successful Body profiling evidence')
        job_path=plain_path(Path(value['job'])/'job.json',[runs]);job=json.loads(job_path.read_text());cmd=job['command']
        if job['status']!='passed' or len(cmd)!=8 or cmd[1]!='request' or cmd[7]!='2':
            raise ValueError('Only completed successful profile requests may be archived')
        path=plain_path(Path(cmd[3])/'request_0'/value['phase']/'profile.bin',[runs])
        row=latest.get(str(path))
        if not row or not row.get('completed') or row.get('category')=='failure' or row.get('stamp')!=file_stamp(path):
            raise ValueError('Require unchanged completed manifest registration: '+str(path))
        if 'active_refs' not in row or any(process_alive(ref) for ref in row['active_refs']):
            raise ValueError('Active/unknown project reference')
        if path.stat().st_nlink!=1 or digest(path)!=value['profile_sha256']:
            raise ValueError('Profile payload identity or link count differs')
        targets[str(path)]=(path,summary,value,row)
    for path in a.schedule:
        path=plain_path(path.absolute(),[runs]);summary=path.parent/'import.json'
        if path.name not in {'schedule.json','eft_candidate.json','structural_candidate.json','uncompressed_schedule.json'}:
            raise ValueError('Only reproducible verbose schedule evidence is admitted')
        imported=json.loads(summary.read_text())
        if not imported.get('native_plan_materialized') or not (path.parent/'plan/metadata.bin').exists():
            raise ValueError('Require the completed native plan and importer provenance')
        row=latest.get(str(path))
        if not row or not row.get('completed') or row.get('category')=='failure' or row.get('stamp')!=file_stamp(path):
            raise ValueError('Require an unchanged completed schedule registration')
        if 'active_refs' not in row or any(process_alive(ref) for ref in row['active_refs']) or path.stat().st_nlink!=1:
            raise ValueError('Active/unknown reference or shared link')
        targets[str(path)]=(path,summary,dict(profile_sha256=digest(path)),row)
    print(json.dumps(dict(mode='apply' if a.apply else 'dry-run',
        sources=[dict(path=k,bytes=v[0].stat().st_size) for k,v in targets.items()])),flush=True)
    if not a.apply:return
    for raw,(source,summary,value,row) in targets.items():
        archive=source.with_name(source.name+'.gz');receipt=source.with_name(source.name+'.archive.json')
        if archive.exists() or receipt.exists():raise FileExistsError('Never overwrite an existing archive')
        with source.open('rb') as src,gzip.GzipFile(filename=str(archive),mode='xb',compresslevel=1,mtime=0) as dst:
            shutil.copyfileobj(src,dst,1<<20)
        with gzip.open(archive,'rb') as stream:uncompressed=hashlib.file_digest(stream,'sha256').hexdigest()
        if uncompressed!=value['profile_sha256'] or row['stamp']!=file_stamp(source):
            raise ValueError('Archive roundtrip/source stability check failed; original retained')
        detail=dict(schema='tilemega.as2.profile_archive.v1',source=raw,bytes=source.stat().st_size,
            source_sha256=uncompressed,archive=str(archive),archive_sha256=digest(archive),
            profile_summary=str(summary),profile_summary_sha256=digest(summary),
            roundtrip_verified=True,source_removal='only through separate manifest cleanup dry-run/apply')
        receipt.write_text(json.dumps(detail,indent=2)+'\n')
        stamp=dt.datetime.now(dt.timezone.utc).isoformat()
        records=[dict(path=str(path),stage='AS2-K',category='result',keep=True,regenerable=False,
                     completed=True,active_refs=[],created_at=stamp,stamp=file_stamp(path)) for path in (archive,receipt)]
        records.append(dict(row,category='archived_trace',keep=False,regenerable=True,selected=False,
            active_refs=[],created_at=stamp,restore_from=str(archive),archive_receipt=str(receipt),
            reason='Lossless compressed payload retained and independently decompressed SHA256 verified'))
        with manifest.open('a') as stream:
            for record in records:stream.write(json.dumps(record)+'\n')
        print(json.dumps(detail),flush=True)


if __name__=='__main__':main()
