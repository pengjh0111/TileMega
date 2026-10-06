#!/usr/bin/env python3
"""Compile production dispatch separately from standalone GEMV arithmetic."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'docs/experiments/SERVING_R14/raw/gemv_integration'
def run(command,name,limit):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/(name+'.command.json')).write_text(json.dumps(list(map(str,command)),indent=2)+'\n')
    with (OUT/(name+'.log')).open('w') as log:
        r=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=limit)
    if r.returncode:raise SystemExit(r.returncode)
def main():
    run(['cmake','--build','build-phase12','--target','tilemega','tilemega-unit','-j','6'],'build',5400)
    run(['ctest','--test-dir','build-phase12','-R','serving_pruning|variant_resource|operator_classes|serving_task_index|handoff_access','--output-on-failure'],'host',600)
    for arch in (80,89,90,100,120):
        run(['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','--expt-relaxed-constexpr',
            f'-arch=sm_{arch}',f'-DTILEMEGA_ARCH_ID={arch*10}','-Iinclude','-Ithird_party/cutlass/include',
            '-Ithird_party/cutlass/tools/util/include','-c','test/unit/serving_gemv_dispatch_test.cu',
            '-o',str(OUT/f'dispatch_sm{arch}.o')],f'dispatch_sm{arch}',600)
    print('GEMV production dispatch: host checks and five-architecture compilation PASS')
if __name__=='__main__':main()
