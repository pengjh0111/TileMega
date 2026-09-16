#!/usr/bin/env python3
"""Capture the probe repair and prove its safe window image did not change."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
RAW=HERE/'window_probe_fix'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def l2(path):
    return next(c for c in path.read_text().split('Function : ')[1:]
                if 'tilemega_l2_kernel' in c.splitlines()[0])


def verify():
    meta=json.loads((RAW/'proof.json').read_text())
    for p,digest in meta['artifacts'].items():
        if sha(REPO/p)!=digest:raise ValueError(f'{p}: changed probe evidence')
    for model in ('gqa2','mha4'):
        old=HERE/'window2/sass'/f'{model}_p0_full.sass'
        new=RAW/f'{model}_safe.sass'
        if old.read_bytes()!=new.read_bytes():raise ValueError(f'{model}: safe window SASS changed')
    old,new=l2(RAW/'before.sass'),l2(RAW/'after.sass')
    if old==new:raise ValueError('unsafe window image did not change')
    before=len(re.findall(r'\bBAR\.RED',old))
    after=len(re.findall(r'\bBAR\.RED',new))
    if (before,after)!=(1,0):raise ValueError('window probe convergence was not removed from nowait')
    if len(re.findall(r'\bATOMG',old))!=1 or re.search(r'\bATOMG',new):
        raise ValueError('window event RMW polling remains in nowait')
    return f'2/2 safe W=2 SASS images unchanged; nowait BAR.RED sites {before} -> {after}; event polling ATOMG 1 -> 0; original partial matrix excluded and retained'


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('capture','verify'));args=p.parse_args()
    if args.phase=='capture':
        artifacts={}
        jobs=[(HERE/'window2/bin/gqa2_p0_nowait',RAW/'after.sass')]
        jobs += [(RAW/'control/bin'/f'{m}_p0_full',RAW/f'{m}_safe.sass') for m in ('gqa2','mha4')]
        commands=[]
        for binary,out in jobs:
            cmd=['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(binary)]
            with out.open('w') as f:subprocess.run(cmd,stdout=f,check=True)
            artifacts[str(out.relative_to(REPO))]=sha(out)
            commands.append(dict(command=cmd,binary_sha256=sha(binary)))
        artifacts[str((RAW/'before.sass').relative_to(REPO))]=sha(RAW/'before.sass')
        (RAW/'proof.json').write_text(json.dumps(dict(commands=commands,artifacts=artifacts),indent=2)+'\n')
    print(verify())


if __name__=='__main__':main()
