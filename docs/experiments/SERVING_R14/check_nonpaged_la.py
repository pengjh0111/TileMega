#!/usr/bin/env python3
"""Bounded host, architecture and numerical checks; never emits timing data."""
import argparse,json,subprocess
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write
from tilemega.build.identity import sha
OUT=HERE/'raw/nonpaged_la_checks'
def run(cmd,name,timeout=1800):
    OUT.mkdir(parents=True,exist_ok=True);write(OUT/(name+'.command.json'),list(map(str,cmd)))
    with (OUT/(name+'.log')).open('w') as f:r=subprocess.run(list(map(str,cmd)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if r.returncode:raise SystemExit(r.returncode)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('compile','numerical'));a=p.parse_args()
    if a.action=='compile':
        run(['cmake','--build','build-phase12','--target','tilemega','tilemega-unit','monotonic_last_arriver_test','-j','6'],'build',5400)
        run(['ctest','--test-dir','build-phase12','-R','handoff_|serving_attention_legality','--output-on-failure'],'host',1800)
        binaries={}
        for arch in (80,89,90,100,120):
            dst=OUT/f'monotonic_sm{arch}';cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3',f'-arch=sm_{arch}','-Iinclude','test/unit/monotonic_last_arriver_test.cu']
            if arch!=89:cmd+=['-c'];dst=dst.with_suffix('.o')
            run(cmd+['-o',dst],f'compile_sm{arch}',600);binaries[str(arch)]=dict(path=str(dst),sha256=sha(dst))
        write(OUT/'binaries.json',binaries);print('Nonpaged LA host and five-architecture compilation PASS')
    else:
        binary=json.loads((OUT/'binaries.json').read_text())['89']
        if sha(binary['path'])!=binary['sha256']:raise ValueError('binary changed')
        run([binary['path']],'numerical',120);print('Monotonic reduction numerical check PASS; 50-process validation remains pending')
if __name__=='__main__':main()
