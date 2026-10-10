#!/usr/bin/env python3
"""Compile CG-generated sparse dependency descriptors on all DM-1 targets."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from identity_dm import resources, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    unit = args.root / 'build-dm/tilemega-unit'
    cu = args.out / 'bound.cu'
    subprocess.run([str(unit), 'bound_dependency_codegen', '--emit', str(cu)], check=True)
    nvcc = '/usr/local/cuda/bin/nvcc'
    common = [nvcc, '-std=c++17', '-O2', '--expt-relaxed-constexpr',
              '-DTILEMEGA_SERVING_PAST_LO=0', '-DTILEMEGA_SERVING_PAST_HI=0',
              '-I' + str(args.root / 'include'),
              '-I' + str(args.root / 'third_party/cutlass/include'),
              '-I' + str(args.root / 'third_party/cutlass/tools/util/include')]
    listing = subprocess.check_output(common + ['-arch=sm_80', '-DTILEMEGA_ARCH_ID=800', '-M', str(cu)], text=True)
    inputs = {Path(word).resolve() for word in shlex.split(listing.replace('\\\n', ' '))[1:] if Path(word).is_file()}
    inputs.update([Path(__file__).resolve(), unit, args.root / 'test/unit/bound_dependency_codegen_test.cpp'])
    hashes = {str(path): sha(path) for path in sorted(inputs)}
    source = dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.root, text=True).strip(),
                  diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=args.root)).hexdigest())
    result = dict(evidence='verified', passed=False, source=source, inputs=hashes,
                  cu_sha256=sha(cu), unit_sha256=sha(unit), builds=[],
                  scope='CG-generated table descriptors and fixed-batch ABI compile checks only; synthetic semantic fixture is not numerically executed',
                  compiler_version=subprocess.check_output([nvcc, '--version'], text=True), compiler_sha256=sha(nvcc))
    try:
        for arch in (80, 89, 90, 100, 120):
            obj = args.out / f'bound-sm_{arch}.o'
            log = args.out / f'bound-sm_{arch}.log'
            command = common + [f'-arch=sm_{arch}', f'-DTILEMEGA_ARCH_ID={arch*10}', '-Xptxas=-v', '-c', str(cu), '-o', str(obj)]
            with log.open('w') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            kernels = resources(log.read_text())
            if not kernels: raise RuntimeError('missing kernel resources')
            result['builds'].append(dict(arch=arch, command=command, object_sha256=sha(obj),
                log_sha256=sha(log), kernels=kernels, spill=any(row['spill'] for row in kernels.values())))
            Path(str(obj) + '.identity.json').write_text(json.dumps(dict(source=source, cu_sha256=sha(cu),
                object_sha256=sha(obj), command=command, architecture=f'sm_{arch}', execution='compile-only',
                implementations=['CG table waits', 'ServingGemmTaskBody::RunDm'], kernels=kernels), indent=2) + '\n')
        log = args.out / 'wrong-batch.log'
        command = common + ['-arch=sm_89', '-DTILEMEGA_ARCH_ID=890', '-DTILEMEGA_SERVING_BATCH_HI=8', '-c', str(cu), '-o', str(args.out/'wrong-batch.o')]
        with log.open('w') as stream:
            rejected = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode != 0
        if not rejected or 'exact task dependencies bind one batch per plan' not in log.read_text():
            raise RuntimeError('fixed-batch ABI mismatch was not rejected at its contract')
        result['wrong_batch_rejected'] = True
        if hashes != {str(path): sha(path) for path in sorted(inputs)}:
            raise RuntimeError('bound codegen inputs changed during validation')
        result['passed'] = True
    finally:
        (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Bound codegen: five targets compiled; incompatible batch rejected', flush=True)


if __name__ == '__main__':
    main()
