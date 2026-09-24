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
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    source=a.out/'selected.cu'; need=8192;free=shutil.disk_usage(a.out).free//2**20
    print(f'DISK NEED_MIB={need} FREE_MIB={free}',flush=True)
    if free<need: raise RuntimeError('disk budget')
    env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
    cmd=[a.compiler,a.bridge,source,'--solver','legacy','--solve',ROOT/'docs/experiments/COSTMODEL/event_fit/target.json','--seq',a.seq,'--past',3,'--dump-cg',a.out/'selected.mlir','--hop-curve',ROOT/'docs/experiments/SIMULATOR/hop_ns.tsv']
    if run(cmd,a.out/'solve.log',env): raise RuntimeError('legacy solve failed')
    cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-arch=sm_89','-lineinfo','-Xptxas=-v','-DTILEMEGA_MIDPOINT_REFINE=0']
    for sub in ['include','third_party/cutlass/include','third_party/cutlass/tools/util/include','third_party/cutlass/test']:cmd+=['-I'+str(ROOT/sub)]
    cmd += [source,ROOT/'build-portable/libtilemega.a','-L/usr/local/cuda/lib64','-lcudart','-o',a.out/'selected']
    if run(cmd,a.out/'build.log',env): raise RuntimeError('legacy build failed')
    env.update(TILEMEGA_WARMUP='2',TILEMEGA_REPEAT='10')
    for i in range(10):
        log=a.out/f'process_{i:02}.log';run([a.out/'selected',a.fixture],log,env)
        text=log.read_text();m=re.search(r'E2E_HASH l05=(\w+) l1=(\w+) l2=(\w+)',text)
        ok=bool(m and len(set(m.groups()))==1 and 'l1_vs_l05_mismatch=0' in text and 'l2_vs_l1_mismatch=0' in text)
        print(f'INTERNAL round={i} pass={ok}',flush=True)
        if not ok:raise RuntimeError('internal equality failed')
if __name__=='__main__':main()
