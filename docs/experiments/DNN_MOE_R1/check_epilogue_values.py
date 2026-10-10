#!/usr/bin/env python3
"""Check numerical value functions and compile their five-target variants."""
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
    source = args.root / 'test/unit/dm_epilogue_value_test.cu'
    fixture = args.root / 'test/fixtures/dnn/epilogue_values.bin'
    nvcc = '/usr/local/cuda/bin/nvcc'
    builds = []
    for arch in (80, 89, 90, 100, 120):
        binary = args.out / f'epilogue-sm_{arch}'
        command = [nvcc, '-std=c++17', '-O2', '--expt-relaxed-constexpr',
                   f'-arch=sm_{arch}', '-Xptxas=-v',
                   '-I' + str(args.root / 'include'),
                   '-I' + str(args.root / 'third_party/cutlass/include'),
                   str(source), '-o', str(binary)]
        log = args.out / f'epilogue-sm_{arch}.log'
        with log.open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        kernels = resources(log.read_text())
        builds.append(dict(arch=arch, command=command, binary_sha256=sha(binary),
                           kernels=kernels,
                           spill=any(row['spill'] for row in kernels.values())))
    output = subprocess.check_output([str(args.out / 'epilogue-sm_89'), str(fixture)], text=True)
    result = dict(evidence='verified', passed=True, output=output,
                  source_sha256=sha(source), fixture_sha256=sha(fixture),
                  headers={name: sha(args.root / name) for name in (
                      'include/tilemega/Backend/DmEpilogueValue.h',
                      'include/tilemega/Codegen/DmDescriptors.h')},
                  head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                               cwd=args.root, text=True).strip(),
                  diff_sha256=hashlib.sha256(subprocess.check_output(
                      ['git', 'diff', '--binary', 'HEAD'], cwd=args.root)).hexdigest(),
                  compiler_version=subprocess.check_output([nvcc, '--version'], text=True),
                  execution_architecture=89, builds=builds,
                  scope='value functions and finite compile-time chains; no TaskBody or synchronization claim')
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(output, end='', flush=True)


if __name__ == '__main__':
    main()
