#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    command = ['flock', os.environ.get('TILEMEGA_GPU_LOCK', '/root/r14_work/gpu.lock'),
               'ctest', '--test-dir', str(args.root / 'build-dm'),
               '--output-on-failure', '-j', '4']
    with args.out.with_suffix('.log').open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    args.out.write_text(json.dumps(dict(exit_code=code, command=command)) + '\n')
    print('ctest complete ' + str(args.root) + ' exit=' + str(code), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
