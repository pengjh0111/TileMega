#!/usr/bin/env python3
"""Compile descriptor probes for five targets and compare the legacy ABI."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.root / 'test/unit/dm_descriptor_device_test.cu'
    nvcc = '/usr/local/cuda/bin/nvcc'
    records = []

    def build(root, name, arches, dm):
        executable = args.out / name
        command = [nvcc, '-std=c++17', '-O2', '--expt-relaxed-constexpr',
                   '-Xptxas=-v', '-I' + str(root / 'include'),
                   '-I' + str(root / 'third_party/cutlass/include'),
                   '-I' + str(root / 'third_party/cutlass/tools/util/include')]
        if dm:
            command.append('-DTILEMEGA_DM_SUPPORT=1')
        for arch in arches:
            command += ['-gencode', f'arch=compute_{arch},code=sm_{arch}']
        command += [str(source), '-o', str(executable)]
        with (args.out / (name + '.log')).open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        record = dict(name=name, command=command, architectures=arches,
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      binary_sha256=hashlib.sha256(executable.read_bytes()).hexdigest())
        records.append(record)
        return executable

    baseline = build(args.reference_root, 'reference-legacy', [89], False)
    candidate = build(args.root, 'candidate-legacy', [89], False)
    old = json.loads(subprocess.check_output([str(baseline), '--layout'], text=True))
    new = json.loads(subprocess.check_output([str(candidate), '--layout'], text=True))
    if old != new:
        raise ValueError(f'legacy device ABI changed: {old} != {new}')
    extended = build(args.root, 'candidate-dm', [80, 89, 90, 100, 120], True)
    output = subprocess.check_output([str(extended)], text=True)
    result = dict(evidence='verified', passed=True, legacy_layout=old,
                  architectures=[80, 89, 90, 100, 120], execution_architecture=89,
                  device_output=output.strip(), builds=records,
                  scope='descriptor ABI and field interpretation; no TaskBody or synchronization claim')
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(output.strip(), flush=True)


if __name__ == '__main__':
    main()
