#!/usr/bin/env python3
"""Compile the new family and validate all supported numerical cases first."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'docs/experiments/SERVING_R14/raw/gemv_checks'
def main():
    from tilemega.build.identity import sha
    p=argparse.ArgumentParser();p.add_argument('action',choices=('compile','numerical'));a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.action=='compile':
        records={}
        for arch in (80,89,90,100,120):
            dst=OUT/f'gemv_sm{arch}';cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','--expt-relaxed-constexpr',f'-arch=sm_{arch}','-Iinclude','-I/root/TileMega/third_party/cutlass/include','-Xptxas=-v','test/unit/serving_gemv_test.cu']
            if arch!=89:cmd+=['-c'];dst=dst.with_suffix('.o')
            cmd+=['-o',str(dst)];(OUT/f'sm{arch}.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
            with (OUT/f'sm{arch}.log').open('w') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=900)
            if r.returncode:raise SystemExit(r.returncode)
            records[str(arch)]=dict(path=str(dst),sha256=sha(dst))
        (OUT/'binaries.json').write_text(json.dumps(records,indent=2)+'\n');print('GEMV five-architecture compilation PASS')
    else:
        b=json.loads((OUT/'binaries.json').read_text())['89']
        if sha(b['path'])!=b['sha256']:raise ValueError('numerical binary changed')
        with (OUT/'numerical.log').open('w') as f:r=subprocess.run([b['path']],stdout=f,stderr=subprocess.STDOUT,timeout=300)
        if r.returncode:raise SystemExit(r.returncode)
        print('GEMV numerical matrix PASS; full plan integration remains pending')
if __name__=='__main__':main()
