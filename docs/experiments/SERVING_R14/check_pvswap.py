#!/usr/bin/env python3
"""Compile and numerically validate the isolated AT-3a implementation."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'docs/experiments/SERVING_R14/raw/AT3a_checks'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(cmd,name,timeout=900):
    (OUT/(name+'.command.json')).write_text(json.dumps(list(map(str,cmd)),indent=2)+'\n')
    with (OUT/(name+'.log')).open('w') as log:
        result=subprocess.run(list(map(str,cmd)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:raise SystemExit(result.returncode)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('compile','numerical'));a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.action=='compile':
        sources=[ROOT/'include/tilemega/Backend/ServingAttentionPVSwap.h',ROOT/'include/tilemega/Codegen/tasks/PagedAttentionTaskBody.h',ROOT/'test/unit/serving_pvswap_test.cu',ROOT/'test/unit/attention_layout_test.cu']
        provenance=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={str(x):sha(x) for x in sources})
        binary={};flags=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','--expt-relaxed-constexpr','-DTILEMEGA_ATTENTION_PVSWAP=1','-Xptxas=-v','-Iinclude','-I/root/TileMega/third_party/cutlass/include']
        for arch in (80,89,90,100,120):
            for test in ('serving_pvswap','attention_layout'):
                out=OUT/f'{test}_sm{arch}';cmd=flags+[f'-arch=sm_{arch}',f'test/unit/{test}_test.cu']
                if arch!=89:cmd+=['-c'];out=out.with_suffix('.o')
                run(cmd+['-o',out],f'{test}_sm{arch}')
                if arch==89:binary[test]=dict(path=str(out),sha256=sha(out))
        provenance['binaries']=binary;(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        print('AT-3a five-architecture compilation PASS')
    else:
        provenance=json.loads((OUT/'provenance.json').read_text())
        for name,b in provenance['binaries'].items():
            if sha(b['path'])!=b['sha256']:raise ValueError('numerical binary changed')
            run([b['path']],name+'_numerical',600)
        print('AT-3a fragment mapping and attention layout numerics PASS')
if __name__=='__main__':main()
