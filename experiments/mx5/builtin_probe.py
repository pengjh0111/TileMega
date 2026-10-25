#!/usr/bin/env python3
"""MX-5 S0: inventory and compile the installed MACA synchronization builtins."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


CALLS = {
    'arrive': '__builtin_mxc_arrive(0x3040);',
    'arrive_bsmcnt': '__builtin_mxc_arrive_bsmcnt(0);',
    'arrive_gvmcnt': '__builtin_mxc_arrive_gvmcnt(0);',
    'barrier': '__builtin_mxc_barrier();',
    'barrier_and_wait1': '__builtin_mxc_barrier_and_wait1(1, memcpy_async<4>(s, p));',
    'barrier_and_wait2': '__builtin_mxc_barrier_and_wait2(1, memcpy_async<8>(s, p));',
    'barrier_and_wait4': '__builtin_mxc_barrier_and_wait4(1, memcpy_async<16>(s, p));',
    'barrier_ex': '__builtin_mxc_barrier_ex(4);',
    'barrier_inst': '__builtin_mxc_barrier_inst();',
    'barrier_shared': '__builtin_mxc_barrier_shared();',
    'barrier_warp': '__builtin_mxc_barrier_warp();',
    'fence': '__builtin_mxc_fence(__ATOMIC_ACQ_REL, "block");',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path,
                        default=Path('/opt/maca/tools/cu-bridge/bin/cucc'))
    parser.add_argument('--clang', type=Path,
                        default=Path('/opt/maca/mxgpu_llvm/bin/mxcc'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    binary = args.clang.read_bytes()
    inventory = sorted({s.decode() for s in re.findall(
        rb'__builtin_mxc_[A-Za-z0-9_]+', binary)})
    (args.out / 'all_builtins.txt').write_text('\n'.join(inventory) + '\n')
    related = [s for s in inventory if any(k in s for k in
                                          ('barrier', 'fence', 'arrive', 'wait'))]
    missing = set(related) - {'__builtin_mxc_' + s for s in CALLS}
    if missing:
        raise RuntimeError('unverified synchronization builtin: ' + ', '.join(sorted(missing)))
    rows = []
    for suffix, expression in CALLS.items():
        source = args.out / (suffix + '.cu')
        source.write_text('#include <cuda_runtime.h>\n'
                          '__global__ void probe(unsigned* p) {\n'
                          '  extern __shared__ unsigned s[];\n'
                          '  p[threadIdx.x] = threadIdx.x;\n  ' + expression + '\n'
                          '  p[threadIdx.x] += 1;\n}\n')
        command = [str(args.compiler), '-std=c++17', '-O3', '-fno-strict-aliasing',
                   '--device-obj', str(source), '-o', str(args.out / (suffix + '.o')),
                   '-mllvm', '-print-after=stack-frame-layout']
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (args.out / (suffix + '.mir')).write_bytes(process.stdout)
        row = dict(builtin='__builtin_mxc_' + suffix, expression=expression,
                   returncode=process.returncode, command=command)
        rows.append(row)
        print(row['builtin'], 'PASS' if not process.returncode else 'UNAVAILABLE', flush=True)
    result = dict(compiler=str(args.compiler), clang=str(args.clang),
                  clang_sha256=hashlib.sha256(binary).hexdigest(),
                  builtin_count=len(inventory), probes=rows)
    (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
