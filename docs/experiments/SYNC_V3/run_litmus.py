#!/usr/bin/env python3
"""Independent sensitivity controls; every invocation creates a CUDA context."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RAW = HERE / 'litmus_v3'
ARMS = {'cache': ('per_writer', 'thread0_fence', 'no_fence'),
        'skew': ('per_writer', 'thread0_fence', 'no_barrier')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('build', 'pilot', 'scan', 'sass'))
    args = parser.parse_args()
    for sub in ('bin', 'log'):
        (RAW / sub).mkdir(parents=True, exist_ok=True)
    binary = RAW / 'bin/litmus'
    if args.phase == 'sass':
        with (RAW / 'litmus.sass').open('w') as f:
            subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump', '--dump-sass', str(binary)], stdout=f, check=True)
        return
    if args.phase == 'build':
        free = shutil.disk_usage(RAW).free // 2**20
        print(f'DISK NEED_MIB=512 FREE_MIB={free}', flush=True)
        if free < 512:
            raise RuntimeError('insufficient disk before compilation')
        cmd = ['/usr/local/cuda/bin/nvcc', '-O2', '-std=c++17', '-arch=sm_89', '-lineinfo',
               '--ptxas-options=-v', f'-I{REPO}/test/harness', str(HERE / 'litmus.cu'), '-o', str(binary)]
        with (RAW / 'log/build.log').open('w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        (RAW / 'build.json').write_text(json.dumps(dict(command=cmd,
            source_sha256=hashlib.sha256((HERE / 'litmus.cu').read_bytes()).hexdigest(),
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            fixed_skew_cycles=2000000, cache_acquire=0, skew_acquire=1), indent=2)+'\n')
        return
    count = 1 if args.phase == 'pilot' else 50
    failures = []
    for suite, arms in ARMS.items():
        for grid in (64, 128, 256):
            for tile in (1024, 4096):
                tallies = {arm: [] for arm in arms}
                for round_ in range(count):
                    for index in range(len(arms)):
                        arm = arms[(round_ + index) % len(arms)]
                        directory = RAW / args.phase / suite / f'g{grid}_t{tile}' / arm
                        directory.mkdir(parents=True, exist_ok=True)
                        cmd = [str(binary), '--release', arm, '--grid', str(grid), '--tile', str(tile),
                               '--writer-skew-cycles', '2000000' if suite == 'skew' else '0']
                        if suite == 'cache':
                            cmd += ['--no-acquire-fence']
                        try:
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                            output, rc = result.stdout+result.stderr, result.returncode
                        except subprocess.TimeoutExpired as e:
                            output, rc = str(e), 124
                        (directory / f'r{round_}.log').write_text(output)
                        (directory / f'r{round_}.json').write_text(json.dumps(dict(
                            command=cmd, exit_code=rc, time_ns=time.time_ns(), round=round_, order=index))+'\n')
                        statuses = re.findall(r'^RESULT status=(\S+)', output, re.M)
                        status = statuses[0] if len(statuses) == 1 else 'ERROR'
                        if rc not in (0, 1):
                            status = 'ERROR'
                        tallies[arm].append(status)
                for arm, statuses in tallies.items():
                    expected = 'pass' if arm in ('per_writer', 'thread0_fence') else 'MISMATCH'
                    passed = statuses.count(expected)
                    print(f'LITMUS {args.phase} {suite} g{grid} t{tile} {arm} {expected}={passed}/{count}', flush=True)
                    if passed != count:
                        failures.append((suite, grid, tile, arm, statuses))
    (RAW / f'{args.phase}_status.json').write_text(json.dumps(dict(failures=failures), indent=2)+'\n')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
