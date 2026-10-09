#!/usr/bin/env python3
"""Build a frozen TaskBody correctness fixture and record its full identity."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arch', type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'framework'))
    from capture_macros_dm import capture
    from identity_dm import resources, sha
    prepared = json.loads((root / 'preparation.json').read_text())
    def unchanged():
        for name, value in prepared['inputs'].items():
            if sha(name) != value:
                raise ValueError('frozen TaskBody input changed: ' + name)
    unchanged()
    binary = root / f'task-sm_{args.arch}'
    source = root / 'fixture.cu'
    command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O2', '-lineinfo',
               '--expt-relaxed-constexpr', '-UNDEBUG', f'-arch=sm_{args.arch}',
               f'-DTILEMEGA_ARCH_ID={args.arch * 10}',
               '-I' + str(root / 'include'),
               '-I' + str(root / 'third_party/cutlass/include'),
               '-I' + str(root / 'third_party/cutlass/tools/util/include'),
               '-Xptxas=-v,-warn-spills', str(source), '-o', str(binary)]
    log = root / f'task-sm_{args.arch}.build.log'
    with log.open('w') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    macro_dir = root / f'task-sm_{args.arch}.macros'
    capture(command, macro_dir)
    unchanged()
    identity = dict(schema='tilemega.dm1.native-test.identity.v1', evidence='verified',
        scope=prepared['scope'], source=prepared, cu_sha256=sha(source),
        binary_sha256=sha(binary), compiler=dict(path=command[0], sha256=sha(command[0]),
            version=subprocess.check_output([command[0], '--version'], text=True)),
        command=command, target_arch=f'sm_{args.arch}', execution=dict(phase='body_unit'),
        implementations=prepared['implementations'], resources=resources(log.read_text()),
        complete_macros=dict(path=str(macro_dir / 'capture.json'),
                             sha256=sha(macro_dir / 'capture.json')))
    identity['artifact_id'] = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    Path(str(binary) + '.identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(dict(event='task_body_build_complete', arch=args.arch,
                         artifact_id=identity['artifact_id'])), flush=True)


if __name__ == '__main__':
    main()
