#!/usr/bin/env python3
"""Merge verified completed CPU queries after interruption, never rerun GPU logs."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('parts',type=Path,nargs='+')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise RuntimeError('refusing to overwrite merged evidence')
    rows=[]; keys=set(); sources={}; binaries=set()
    for part in a.parts:
        manifest=json.loads((part/'manifest.json').read_text())
        binaries.add(manifest['tool_sha256'])
        for command in manifest['commands']:
            if command.get('returncode')!=0: raise RuntimeError('failed query cannot enter exact gate')
        part_rows=list(csv.DictReader((part/'counts.tsv').open(),delimiter='\t'))
        if len(part_rows)!=15*len(manifest['commands']):
            raise RuntimeError('incomplete query output in interrupted prefix')
        for row in part_rows:
            key=(row['model'],int(row['split']),int(row['seq']),int(row['past']))
            if key in keys or row['comparison']!='EXACT_RUNTIME_MATCH':
                raise RuntimeError('duplicate or unverified count row')
            keys.add(key); rows.append(row)
        for path in part.iterdir():
            if path.is_file(): sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    expected=set(itertools.product(('gqa2','mha4'),(1,2,4,8,16),(1,4,128,512,2048),(0,3,512)))
    if keys!=expected or len(binaries)!=1:
        raise RuntimeError('full 150-cell coverage or frozen tool identity differs')
    a.out.mkdir(parents=True)
    with (a.out/'counts.tsv').open('w') as stream:
        writer=csv.DictWriter(stream,rows[0].keys(),delimiter='\t',lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    (a.out/'manifest.json').write_text(json.dumps(dict(
        cells=len(keys),tool_sha256=next(iter(binaries)),sources=sources,
        note='Completed exact symbolic queries only; initial batch interrupted after query eight.'),indent=2)+'\n')


if __name__=='__main__': main()
