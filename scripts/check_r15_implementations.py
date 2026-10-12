#!/usr/bin/env python3
"""Validate a frozen workload/attention checkpoint without collecting timings."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from check_full_checkpoint import validate

TARGETS=['tilemega','tilemega-unit','attention_layout_test','attention_layout_pvswap_test',
         'attention_layout_pvswap2_test','attention_layout_simt_test',
         'serving_prefill_attention_test','serving_prefill_attention_128_test']


def execute(command,out,label,timeout=10800):
    (out/(label+'.command.json')).write_text(json.dumps(command)+'\n')
    with (out/(label+'.log')).open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=timeout)
        except BaseException:
            from tilemega.bench.gpu_guard import stop
            stop(process)
            raise
    if code:raise RuntimeError(f'{label} failed with {code}; see {out}/{label}.log')
    print('PASS '+label,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('build','attention-build','architectures','numeric'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();record=validate(out)
    root=Path(record['tree']);build=root/'build-r15'
    os.environ['PYTHONPATH']=str(root/'python')
    if args.action=='numeric':
        if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1':raise RuntimeError('numerical suite needs bench guard')
        for process in range(5):
            execute(['ctest','--test-dir',str(build),'--output-on-failure','-j','1','--timeout','600',
                '-R','^(attention_layout(_pvswap|_pvswap2|_simt)?|attention_pvswap2_bitwise|serving_prefill_attention(_128)?)$'],
                out,f'numeric_process{process}')
    else:
        with Path(os.environ['TILEMEGA_GPU_LOCK']).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if args.action in ('build','attention-build'):
                command=['cmake','-S',str(root),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
                    '-DTILEMEGA_TARGET_ARCH=sm_89','-DTILEMEGA_BUILD_TESTS=ON','-DTILEMEGA_BUILD_VERIFY=OFF',
                    '-DTILEMEGA_BUILD_EXPERIMENTAL=OFF','-DMLIR_DIR=/root/toolchains/mlir-23a60f15/lib/cmake/mlir',
                    '-DPython3_EXECUTABLE='+sys.executable]
                for dep in ('ISL','POLYLIB','BARVINOK'):
                    command.append('-DTILEMEGA_'+dep+'_BUILD_DIR='+str(Path(record['root'])/('build-'+dep.lower())))
                execute(command,out,'configure',600)
                targets=TARGETS if args.action=='build' else TARGETS[2:]
                execute(['cmake','--build',str(build),'--target',*targets,'-j','3'],out,'build',28800)
            else:
                for arch in (80,89,90,100,120):
                    for impl in (0,1,2,3):
                        label=f'attention_impl{impl}_sm{arch}'
                        execute([os.getenv('NVCC','/usr/local/cuda/bin/nvcc'),'-std=c++17','-O3',
                            '--expt-relaxed-constexpr','-arch=sm_'+str(arch),
                            '-DTILEMEGA_ARCH_ID='+str(arch*10),'-DTILEMEGA_ATTENTION_IMPL='+str(impl),
                            '-I'+str(root/'include'),'-I'+str(root/'third_party/cutlass/include'),
                            '-c',str(root/'test/unit/attention_impl_compile.cu'),'-o',str(out/(label+'.o'))],out,label,1800)
    validate(out)
    (out/(args.action+'.done.json')).write_text(json.dumps(dict(head=record['head'],pass_rate='5/5' if args.action=='numeric' else None))+'\n')

if __name__=='__main__':main()
