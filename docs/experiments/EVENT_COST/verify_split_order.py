#!/usr/bin/env python3
"""Audit complete process matrices and optionally reuse cached symbolic counts."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser()
    p.add_argument('raw',type=Path)
    p.add_argument('--projection',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads((a.raw/'build_manifest.json').read_text())
    config=manifest['arguments']
    if config['runs']<50:
        raise RuntimeError('fewer than 50 fresh processes requested')
    if not (a.raw/'status.txt').read_text().startswith('COMPLETE;'):
        raise RuntimeError('matrix is not complete; no partial acceptance')
    expected=set(itertools.product(range(config['runs']),config['models'],
        config['splits'],config['states'],config['seqs'],config['pasts']))
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    for build in manifest['builds']:
        binary=a.raw/'bin'/Path(build['command'][-1]).name
        if sha(binary)!=build['binary_sha256']:
            raise RuntimeError(f'binary changed: {binary}')
    projection={}
    if a.projection:
        for row in csv.DictReader(a.projection.open(),delimiter='\t'):
            if int(row['kappa'])!=1:
                raise RuntimeError('this runner has kappa1; projection differs')
            key=(row['model'],int(row['split']),int(row['seq']),int(row['past']))
            if key in projection:
                raise RuntimeError('duplicate symbolic cell')
            projection[key]=row
    found=set()
    counts={}
    log_hashes={}
    counter_comparisons=0
    for row in csv.DictReader((a.raw/'correctness.tsv').open(),delimiter='\t'):
        r,m,k,s,q,past=(int(row['round']),row['model'],int(row['split']),
                       int(row['order']),int(row['seq']),int(row['past']))
        key=(r,m,k,s,q,past)
        if key not in expected or key in found:
            raise RuntimeError(f'unexpected or duplicate process: {key}')
        found.add(key)
        log=a.raw/'logs'/f'{m}_k{k}_order{s}_s{q}_p{past}_r{r}.txt'
        text=log.read_text()
        def record(prefix):
            lines=[line for line in text.splitlines() if line.startswith(prefix+' ')]
            if len(lines)!=1: raise RuntimeError(f'ambiguous log: {log} {prefix}')
            return dict(re.findall(r'(\w+)=([^\s]+)',lines[0]))
        hashes=record('E2E_HASH')
        schedule=record('E2E_SCHEDULE')
        for field in ('task_refs','waits'):
            if row[field]!=schedule[field]:
                raise RuntimeError(f'CSV/log mismatch: {log} {field}')
        passed='RESULT status=PASS' in text
        if int(row['pass'])!=int(passed):
            raise RuntimeError(f'CSV/log PASS mismatch: {log}')
        if s and (not passed or len({hashes[f] for f in ('l05','l1','l2')})!=1):
            raise RuntimeError(f'repaired correctness failed: {log}')
        cell=(m,k,s,q,past)
        entry=counts.setdefault(cell,[0,0])
        entry[0]+=int(passed); entry[1]+=1
        if a.projection and s:
            symbolic=projection[(m,k,q,past)]
            for field in ('task_refs','waits'):
                if row[field]!=symbolic[field]:
                    raise RuntimeError(f'A2 exact count mismatch: {log} {field}')
                counter_comparisons+=1
        log_hashes[str(log.relative_to(a.raw))]=sha(log)
    if found!=expected:
        raise RuntimeError(f'missing {len(expected-found)} fresh-process logs')
    report=dict(scope='requested frozen build/fixtures only; not future header edits',
        processes=len(found),symbolic_counter_comparisons=counter_comparisons,
        cells=[dict(model=m,split=k,order=s,seq=q,past=past,passes=v[0],processes=v[1])
               for (m,k,s,q,past),v in sorted(counts.items())],
        build_manifest_sha256=sha(a.raw/'build_manifest.json'),log_sha256=log_hashes)
    if a.out.exists(): raise RuntimeError('refusing to overwrite completed evidence')
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(f'VERIFIED processes={len(found)} counter_comparisons={counter_comparisons}')


if __name__=='__main__': main()
