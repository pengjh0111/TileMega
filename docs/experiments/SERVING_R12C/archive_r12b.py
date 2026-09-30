#!/usr/bin/env python3
"""D-0: preserve original diagnostic inputs without copying code or binaries."""
import argparse, hashlib, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent

def main():
    out=HERE/'r12b_archive'; out.mkdir(parents=True,exist_ok=True)
    paths=set()
    old=ROOT/'docs/experiments/SERVING_R12B'
    for name in ('s1.json','s1b.json','defaults.json'):
        paths.add(old/name)
    patterns=('measurements.json','step_times.tsv','*.plan.json','*.classes.tsv','*.top3*.tsv','*.materializations.tsv','*.resources.tsv','*.search.tsv')
    for base in [old/'controls',*ROOT.glob('runs/r12b-*')]:
        if not base.exists(): continue
        for pattern in patterns:
            paths.update(base.rglob(pattern))
        for name in ('plans.json','cache.json','config.json'):
            if (base/name).exists():paths.add(base/name)
        paths.update(base.glob('commands/*/command.json'))
        cache=base/'cache.json'
        if cache.exists():
            for event in json.loads(cache.read_text()):
                if event.get('layer')=='plan' and event.get('key'):
                    plan=Path.home()/'.cache/tilemega/plans'/event['key']
                    for pattern in ('record.json','options.json','floor.json','plan.so.plan.json',*[f'plan.so.{n}.tsv' for n in ('top3','top3_measured','top3_measure_rounds','classes','materializations','resources','search')]):
                        paths.update(plan.glob(pattern))
    paths.update(old.glob('queue_run*/progress.tsv'))
    s0=Path('/root/r12_work/r12b_s0')
    for pattern in ('smoke64/smoke.json','measure/measurements.json','*.plan.json','*.classes.tsv'):
        paths.update(s0.rglob(pattern))
    manifest=[]; missing=[]
    for p in sorted(paths):
        if not p.is_file():missing.append(str(p));continue
        relative=p.relative_to('/root')
        dst=out/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        manifest.append((str(p),str(dst.relative_to(HERE)),hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size))
    (out/'MANIFEST.tsv').write_text('source\tarchived\tsha256\tbytes\n'+''.join('\t'.join(map(str,row))+'\n' for row in manifest))
    (out/'missing.json').write_text(json.dumps(missing,indent=2)+'\n')
    print(f'D-0: archived {len(manifest)} files, {sum(r[3] for r in manifest)} bytes; missing={len(missing)}')
if __name__=='__main__':main()
