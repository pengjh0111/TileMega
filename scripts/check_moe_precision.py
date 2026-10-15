#!/usr/bin/env python3
"""Validate both exported routing precisions in a frozen checkpoint."""
from target_arch import nvcc_flag

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

from check_full_checkpoint import validate
from check_r15_implementations import execute


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('build','numeric','architectures'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();record=validate(out)
    root=Path(record['tree']);build=root/'build-r15'
    if args.action=='numeric':
        if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1':
            raise RuntimeError('numerical checks require the GPU guard')
        for process in range(5):
            execute(['ctest','--test-dir',str(build),'--output-on-failure','--timeout','600',
                     '-R','^dm_moe_(routing_combine_[01]|rms_stats)$'],out,f'numeric_{process}')
    else:
        with Path(os.environ['TILEMEGA_GPU_LOCK']).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if args.action=='build':
                command=['cmake','-S',str(root),'-B',str(build),'-G','Ninja',
                    '-DCMAKE_BUILD_TYPE=Release','-DTILEMEGA_TARGET_ARCH=sm_89',
                    '-DTILEMEGA_BUILD_TESTS=ON','-DTILEMEGA_BUILD_VERIFY=OFF',
                    '-DTILEMEGA_BUILD_EXPERIMENTAL=OFF',
                    '-DMLIR_DIR=/root/toolchains/mlir-23a60f15/lib/cmake/mlir',
                    '-DPython3_EXECUTABLE='+sys.executable]
                for dep in ('ISL','POLYLIB','BARVINOK'):
                    command.append('-DTILEMEGA_'+dep+'_BUILD_DIR='+str(Path(record['root'])/('build-'+dep.lower())))
                execute(command,out,'configure',600)
                execute(['cmake','--build',str(build),'--target','tilemega','tilemega-unit',
                    'dm_moe_routing_combine_0_test','dm_moe_routing_combine_1_test',
                    'dm_moe_rms_stats_test','-j','3'],out,'build',21600)
                execute(['ctest','--test-dir',str(build),'--output-on-failure','-R',
                         '^(moe_region_(pattern|plan)|r15_moe_full)$'],out,'host',600)
            else:
                for arch in (80,89,90,100,120):
                    label=f'moe_fp32_sm{arch}'
                    execute([os.getenv('NVCC','/usr/local/cuda/bin/nvcc'),'-std=c++17','-O3',
                        '--expt-relaxed-constexpr',nvcc_flag(arch),
                        '-DTILEMEGA_ARCH_ID='+str(arch*10),'-DDM_TEST_STAGE_WRAPPERS=1','-UNDEBUG',
                        '-I'+str(root/'include'),'-I'+str(root/'third_party/cutlass/include'),
                        '-c',str(root/'test/unit/dm_moe_routing_combine_test.cu'),
                        '-o',str(out/(label+'.o'))],out,label,3600)
    validate(out)
    (out/(args.action+'.done.json')).write_text(json.dumps(dict(head=record['head'],
        pass_rate='5/5' if args.action=='numeric' else None))+'\n')


if __name__=='__main__':main()
