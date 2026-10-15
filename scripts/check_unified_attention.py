#!/usr/bin/env python3
"""Build and numerically check the first single-stack attention checkpoint."""
from target_arch import nvcc_flag

import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path

TARGETS = ['attention_layout_test', 'serving_pvswap_test',
           *[f'dm_qperkv8_attention_mode{mode}_test' for mode in range(5)],
           'dm_qperkv8_prefill_test']


def run(command, out, label):
    with (out / (label + '.log')).open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'run'))
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.out = args.out.resolve()
    args.build_dir = args.build_dir.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.action == 'build':
        lock = Path(os.environ['TILEMEGA_GPU_LOCK'])
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            run(['cmake', '--build', str(args.build_dir), '--target', *TARGETS, '-j', '3'], args.out, 'build')
            common = [os.getenv('NVCC', '/usr/local/cuda/bin/nvcc'), '-std=c++17', '-O3',
                      '--expt-relaxed-constexpr', '-I' + str(root / 'include'),
                      '-I' + str(args.build_dir / 'include'),
                      '-I' + str(root / 'third_party/cutlass/include')]
            arches = []
            for arch in (80, 89, 90, 100, 120):
                command = [*common, nvcc_flag(arch), '-c',
                    str(root / 'test/unit/serving_pvswap_test.cu'),
                    '-o', str(args.out / f'pvswap_sm{arch}.o')]
                run(command, args.out, f'compile_sm{arch}')
                arches.append(arch)
            run([*common, '-arch=sm_89', '-DTILEMEGA_ATTENTION_PVSWAP=1',
                 str(root / 'test/unit/attention_layout_test.cu'),
                 '-o', str(args.out / 'attention_layout_pvswap')], args.out, 'build_layout_pvswap')
            (args.out / 'build.json').write_text(json.dumps(dict(arches=arches, targets=TARGETS)) + '\n')
    else:
        if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
            raise RuntimeError('numerical checks must run under the scheduler GPU guard')
        run(['ctest', '--test-dir', str(args.build_dir), '--output-on-failure',
             '-R', '^(attention_layout|serving_pvswap|dm_qperkv8_attention_mode[0-4]|dm_qperkv8_prefill)$'], args.out, 'ctest')
        run([str(args.out / 'attention_layout_pvswap')], args.out, 'layout_pvswap')
        (args.out / 'numeric.json').write_text(json.dumps(dict(status='pass',
            scope='attention transport and PV templates; G-REG-U is still pending')) + '\n')
    print(f'unified attention {args.action} PASS', flush=True)


if __name__ == '__main__':
    main()
