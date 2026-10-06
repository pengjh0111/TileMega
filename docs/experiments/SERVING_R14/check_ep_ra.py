#!/usr/bin/env python3
"""Validate EP-1 and RA-1 without using their timings for selection."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'docs/experiments/SERVING_R14/raw/ep_ra_checks'
def run(cmd,name,timeout=900):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/(name+'.command.json')).write_text(json.dumps(list(map(str,cmd)),indent=2)+'\n')
    with (OUT/(name+'.log')).open('w') as f:r=subprocess.run(list(map(str,cmd)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if r.returncode:raise SystemExit(r.returncode)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('compile','numerical'));a=p.parse_args()
    from tilemega.build.identity import sha
    if a.action=='compile':
        binary={};flags=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','--expt-relaxed-constexpr','-DTILEMEGA_EP_PARALLEL_ARGMAX=1','-DTILEMEGA_ATTENTION_NOINLINE=1','-Iinclude','-I/root/TileMega/third_party/cutlass/include','-Xptxas=-v']
        for arch in (80,89,90,100,120):
            for test in ('serving_argmax_rows','serving_epilogue','attention_layout','paged_gemm'):
                dst=OUT/f'{test}_sm{arch}';cmd=flags+[f'-arch=sm_{arch}',f'test/unit/{test}_test.cu']
                if arch!=89:cmd+=['-c'];dst=dst.with_suffix('.o')
                run(cmd+['-o',dst],f'{test}_sm{arch}')
                if arch==89:binary[test]=dict(path=str(dst),sha256=sha(dst))
        provenance=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),binaries=binary)
        (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print('EP/RA five-architecture compilation PASS')
    else:
        for test,b in json.loads((OUT/'provenance.json').read_text())['binaries'].items():
            if sha(b['path'])!=b['sha256']:raise ValueError('changed numerical binary')
            run([b['path']],test+'_numerical',300)
        print('EP/RA position-coded, argmax and epilogue numerical checks PASS')
if __name__=='__main__':main()
