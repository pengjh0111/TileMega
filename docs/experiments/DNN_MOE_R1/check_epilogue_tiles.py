#!/usr/bin/env python3
"""Compile finite tile epilogues and verify them in 50 fresh processes."""
import argparse
import hashlib
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
    names = ['test/unit/dm_epilogue_tile_test.cu',
             'test/fixtures/dnn/epilogue_tiles.bin',
             'include/tilemega/Backend/ServingDmEpilogue.h',
             'include/tilemega/Backend/DmEpilogueValue.h',
             'include/tilemega/Backend/DmTensorAddress.h',
             'include/tilemega/Codegen/DmDescriptors.h',
             'include/tilemega/Codegen/executor/ComputeGroup.cuh']
    hashes = {name: sha(args.root / name) for name in names}
    nvcc = '/usr/local/cuda/bin/nvcc'
    result = dict(evidence='verified', passed=False, inputs=hashes, builds=[],
                  execution_architecture=89, fresh_processes=[],
                  head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                               cwd=args.root, text=True).strip(),
                  diff_sha256=hashlib.sha256(subprocess.check_output(
                      ['git', 'diff', '--binary', 'HEAD'], cwd=args.root)).hexdigest(),
                  compiler_version=subprocess.check_output([nvcc, '--version'], text=True),
                  scope='tile-local epilogue storage and synchronization; not inter-stage statistics or any §8.A path')
    try:
        for arch in (80, 89, 90, 100, 120):
            binary = args.out / f'tiles-sm_{arch}'
            command = [nvcc, '-std=c++17', '-O2', '--expt-relaxed-constexpr',
                       f'-arch=sm_{arch}', f'-DTILEMEGA_ARCH_ID={arch * 10}', '-Xptxas=-v',
                       '-I' + str(args.root / 'include'),
                       '-I' + str(args.root / 'third_party/cutlass/include'),
                       str(args.root / names[0]), '-o', str(binary)]
            log = args.out / f'tiles-sm_{arch}.log'
            with log.open('w') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            kernels = resources(log.read_text())
            result['builds'].append(dict(arch=arch, command=command,
                binary_sha256=sha(binary), kernels=kernels,
                spill=any(row['spill'] for row in kernels.values())))
        for process in range(50):
            log = args.out / f'process-{process:02d}.log'
            with log.open('w') as stream:
                status = subprocess.run([str(args.out / 'tiles-sm_89'), str(args.root / names[1])],
                                        stdout=stream, stderr=subprocess.STDOUT).returncode
            result['fresh_processes'].append(dict(process=process, exit_code=status,
                                                  log_sha256=sha(log)))
            if status:
                raise RuntimeError(f'process {process} failed; see {log}')
        if hashes != {name: sha(args.root / name) for name in names}:
            raise RuntimeError('epilogue inputs changed during validation')
        result['passed'] = True
    finally:
        passes = sum(row['exit_code'] == 0 for row in result['fresh_processes'])
        result['pass_rate'] = dict(passes=passes, attempts=len(result['fresh_processes']), required=50)
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Tile epilogues: five targets compiled; 50/50 fresh processes passed', flush=True)


if __name__ == '__main__':
    main()
