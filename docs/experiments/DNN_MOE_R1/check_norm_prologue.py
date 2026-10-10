#!/usr/bin/env python3
"""Verify the repaired prologue without weakening its bitwise criterion."""
import argparse
import json
from pathlib import Path
import subprocess
from identity_dm import resources, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = ['test/unit/norm_prologue_gemm_test.cu',
             'include/tilemega/Codegen/tasks/PagedGemmTaskBody.h',
             'include/tilemega/Codegen/tasks/ServingRMSNormTaskBody.h']
    hashes = {name: sha(args.root / name) for name in names}
    nvcc = '/usr/local/cuda/bin/nvcc'
    result = dict(evidence='verified', passed=False, inputs=hashes, builds=[],
                  fresh_processes=[], execution_architecture=89,
                  criterion='24 geometries: fused prologue versus standalone RMSNorm plus page GEMM; memcmp unchanged')
    try:
        for arch in (80, 89, 90, 100, 120):
            binary = args.out / f'norm-sm_{arch}'
            command = [nvcc, '-std=c++17', '-O2', '--expt-relaxed-constexpr',
                       f'-arch=sm_{arch}', f'-DTILEMEGA_ARCH_ID={arch * 10}', '-Xptxas=-v',
                       '-I' + str(args.root / 'include'),
                       '-I' + str(args.root / 'third_party/cutlass/include'),
                       str(args.root / names[0]), '-o', str(binary)]
            log = args.out / f'norm-sm_{arch}.log'
            with log.open('w') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            kernels = resources(log.read_text())
            result['builds'].append(dict(arch=arch, command=command,
                binary_sha256=sha(binary), kernels=kernels,
                spill=any(row['spill'] for row in kernels.values())))
        for process in range(50):
            log = args.out / f'process-{process:02d}.log'
            with log.open('w') as stream:
                status = subprocess.run([str(args.out / 'norm-sm_89')],
                    stdout=stream, stderr=subprocess.STDOUT).returncode
            result['fresh_processes'].append(dict(process=process, exit_code=status,
                                                  log_sha256=sha(log)))
            if status:
                raise RuntimeError(f'process {process} failed; see {log}')
        if hashes != {name: sha(args.root / name) for name in names}:
            raise RuntimeError('normalization inputs changed during verification')
        result['passed'] = True
    finally:
        result['pass_rate'] = dict(passes=sum(row['exit_code'] == 0 for row in result['fresh_processes']),
                                  attempts=len(result['fresh_processes']), required=50)
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Norm prologue: five targets compiled; 50/50 fresh processes passed', flush=True)


if __name__ == '__main__':
    main()
