#!/usr/bin/env python3
"""Fresh R9 control solve and ten processes. CPU golden is not this gate."""
import argparse, hashlib, json, os, pathlib, re, shutil, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parents[3]
def run(cmd,log,env=None):
    started=time.time_ns()
    with log.open('w') as f:
        p=subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
    log.with_suffix('.json').write_text(json.dumps(dict(command=list(map(str,cmd)),started_ns=started,elapsed_ns=time.time_ns()-started,exit_code=p.returncode),indent=2)+'\n')
    return p.returncode

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bridge',type=pathlib.Path,required=True);ap.add_argument('--fixture',type=pathlib.Path,required=True)
    ap.add_argument('--seq',type=int,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--compiler',type=pathlib.Path,default=ROOT/'build-portable/tools/tilemega-compile')
    ap.add_argument('--domain',type=pathlib.Path);ap.add_argument('--capacity',type=int,default=8)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    source=a.out/'selected.cu'; need=8192;free=shutil.disk_usage(a.out).free//2**20
    print(f'DISK NEED_MIB={need} FREE_MIB={free}',flush=True)
    if free<need: raise RuntimeError('disk budget')
    env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
    cmd=[a.compiler,a.bridge,source,'--solver','legacy','--solve',ROOT/'docs/experiments/COSTMODEL/event_fit/target.json','--seq',a.seq,'--past',3,'--dump-cg',a.out/'selected.mlir','--hop-curve',ROOT/'docs/experiments/SIMULATOR/hop_ns.tsv']
    cmd+=['--search-capacity',a.capacity]
    if a.domain:cmd+=['--search-domain',a.domain]
    if run(cmd,a.out/'solve.log',env): raise RuntimeError('legacy solve failed')
    cmd=['python3',ROOT/'docs/experiments/SOLVER_V2/measure.py','--source',source,'--fixture',a.fixture]
    if run(cmd,a.out/'measure.log',env):raise RuntimeError('legacy measurement failed')
if __name__=='__main__':main()
