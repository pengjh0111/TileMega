#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Build the CPU semantic-class case writer against an immutable TileMega build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-build', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build, out = args.snapshot_build.absolute(), args.out.absolute()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).absolute().with_name('r16_dnn_class_cases.cpp')
    commands = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands', 'tilemega'], text=True).splitlines()
    original = shlex.split(next(row for row in commands if 'lib/Codegen/Codegen.cpp' in row and ' -c ' in row))
    compile_command, position = [], 0
    while position < len(original):
        value = original[position]
        if value in ('-MT', '-MF', '-o'):
            position += 2
            continue
        if value == '-MD':
            position += 1
            continue
        if value.endswith('/lib/Codegen/Codegen.cpp'):
            value = str(source)
        compile_command.append(value)
        position += 1
    compile_command += ['-o', str(out/'class_cases.o')]
    original = shlex.split(commands[-1].split('&&')[1].strip())
    link_command, position = [], 0
    while position < len(original):
        value = original[position]
        if value.endswith('.o'):
            position += 1
            continue
        if value == '-o':
            link_command += ['-o', str(out/'class_cases')]
            position += 2
            continue
        link_command.append(value)
        position += 1
    link_command.insert(1, str(out/'class_cases.o'))
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    receipt = dict(gpu_executed=False, build=str(build), source=str(source),
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   compile=compile_command, link=link_command, steps=[])
    for stage, command in (('compile', compile_command), ('link', link_command)):
        with (out/f'{stage}.log').open('w') as stream:
            result = subprocess.run(command, cwd=build, env=env, stdout=stream, stderr=subprocess.STDOUT)
        receipt['steps'].append({'stage':stage, 'returncode':result.returncode})
        (out/'build-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        if result.returncode:
            raise SystemExit(result.returncode)
    receipt['binary_sha256'] = hashlib.sha256((out/'class_cases').read_bytes()).hexdigest()
    (out/'build-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
