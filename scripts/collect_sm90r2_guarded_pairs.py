#!/usr/bin/env python3
"""Collect complete accepted pairs without using either half of a rejected run."""
import argparse
import csv
from pathlib import Path
from sm90d1_common import read,sha
from run_sm90r2_pair import measurement,validate_pair


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--states',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=[]
    for state in a.states:
        for name,row in read(state).items():
            if row['status']!='done':continue
            directory=Path(row['out']);path=directory/'pair/paired.json'
            if not path.exists():continue
            guard=read(directory/'guard_result.json');receipt=read(path)
            if guard['code'] or receipt['status']!='completed':
                raise ValueError('a completed paired job lacks accepted full-pair evidence')
            root=path.parent;before,after=measurement(root/'reference'),measurement(root/'candidate')
            validate_pair(before,after)
            first=read(root/'reference/job/selected-config.json')
            second=read(root/'candidate/job/selected-config.json')
            if receipt['baseline_pid']!=before['pid'] or receipt['candidate_pid']!=after['pid']:
                raise ValueError('paired process identity changed')
            baseline_ms=before['tpot_seconds']*1000;candidate_ms=after['tpot_seconds']*1000
            rows.append(dict(pair=name.rsplit('-pair-f',1)[0],fresh_process=receipt['fresh_process'],
                baseline_ms=baseline_ms,candidate_ms=candidate_ms,
                change_pct=(candidate_ms/baseline_ms-1)*100,
                token_equality=read(root/'reference/job/measurements/reference_tokens.json')==
                    read(root/'candidate/job/measurements/reference_tokens.json'),
                uuid=receipt['physical_uuid'],baseline_sha256=first['decode_sha256'],
                candidate_sha256=second['decode_sha256'],baseline_tag=receipt['baseline_tag'],
                candidate_tag=receipt['candidate_tag'],baseline_run=str(root/'reference'),
                candidate_run=str(root/'candidate'),baseline_pid=before['pid'],candidate_pid=after['pid'],
                paired_guard=True,guard_sha256=sha(directory/'guard_result.json'),
                ownership_verified=bool(before['ownership_verified'] and after['ownership_verified'])))
    if not rows:raise ValueError('no accepted complete guarded pairs')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'{len(rows)} accepted complete guarded pairs')


if __name__=='__main__':
    main()
