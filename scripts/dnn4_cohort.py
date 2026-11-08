#!/usr/bin/env python3
"""Run fresh sequential children inside one guarded GPU reservation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from tilemega.build.identity import sha


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def archive_sources(out, frozen):
    folder=out/'sources';folder.mkdir()
    sources={}
    for path,digest in frozen.items():
        path=Path(path)
        if path.suffix!='.py':
            continue
        target=folder/(digest+'-'+path.name)
        shutil.copyfile(path,target)
        if sha(target)!=digest:
            raise ValueError('cohort source changed while archiving')
        sources[str(path)]=dict(sha256=digest,copy=str(target))
    save(folder/'index.json',sources)


def execute(plan_path, out):
    from tilemega.serving.baseline_common import require_guard
    require_guard()
    plan = json.loads(plan_path.read_text())
    frozen = plan['frozen']
    if any(sha(path)!=digest for path,digest in frozen.items()):
        raise ValueError('cohort inputs differ before execution')
    out.mkdir(parents=True, exist_ok=False)
    processes = []
    for step in plan['steps']:
        folder = out/step['name']
        folder.mkdir()
        command = [word.replace('{attempt_dir}',str(folder)) for word in step['command']]
        with (folder/'process.log').open('w') as log:
            child = subprocess.Popen(command,cwd=step['cwd'],env=dict(os.environ,**step['env']),
                                     stdout=log,stderr=subprocess.STDOUT)
            try:
                code = child.wait(timeout=step['timeout_s'])
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        processes.append(dict(name=step['name'],pid=child.pid,exit_code=code,
                              out=str(folder),command=command))
        save(out/'result.json',dict(status='partial',processes=processes))
        if code==75:
            # The per-process sampler can detect occupancy after preflight.
            # Retry the whole reservation, never an incomplete A/B sequence.
            raise SystemExit(75)
        if code:
            raise RuntimeError('fresh cohort child failed: '+step['name'])
    if len({p['pid'] for p in processes})!=len(processes):
        raise ValueError('cohort did not use distinct fresh process IDs')
    if any(sha(path)!=digest for path,digest in frozen.items()):
        raise ValueError('cohort inputs changed during execution')
    save(out/'result.json',dict(status='passed',processes=processes,
        reservation='one unchanged guard preflight;all children sequential and monitored',
        gpu=int(os.environ['TILEMEGA_DEVICE_INDEX'])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    execute(args.plan,args.out)


if __name__=='__main__':
    main()
