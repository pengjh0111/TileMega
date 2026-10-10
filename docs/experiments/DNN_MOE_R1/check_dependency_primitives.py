#!/usr/bin/env python3
"""Verify dependency primitives on five targets and 50 fresh sm_89 processes."""
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
    nvcc = '/usr/local/cuda/bin/nvcc'
    source = args.root / 'test/unit/dm_dependency_test.cu'
    common = [nvcc, '-std=c++17', '-O2', '--expt-relaxed-constexpr', '-I' + str(args.root / 'include')]
    listing = subprocess.check_output(common + ['-arch=sm_80', '-M', str(source)], text=True)
    dependencies = {Path(__file__).resolve(), Path(nvcc)}
    for word in shlex.split(listing.replace('\\\n', ' '))[1:]:
        path = Path(word).resolve()
        if path.is_file():
            dependencies.add(path)
    hashes = {str(path): sha(path) for path in sorted(dependencies)}
    result = dict(evidence='verified', passed=False, inputs=hashes, builds=[], processes=[],
                  source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.root, text=True).strip(),
                  diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=args.root)).hexdigest(),
                  compiler_version=subprocess.check_output([nvcc, '--version'], text=True), compiler_sha256=sha(nvcc),
                  scope='synthetic sparse table waits, counted release/acquire with runtime target permutation, weighted last-arriver and zero contribution; 32 epochs each; full executor/body gates pending')
    try:
        for arch in (80, 89, 90, 100, 120):
            binary = args.out / f'dependency-sm_{arch}'
            command = common + [f'-arch=sm_{arch}', f'-DTILEMEGA_ARCH_ID={arch * 10}', '-Xptxas=-v',
                                str(source), '-o', str(binary)]
            log = args.out / f'build-sm_{arch}.log'
            with log.open('w') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            kernels = resources(log.read_text())
            result['builds'].append(dict(arch=arch, command=command, binary_sha256=sha(binary),
                                         log_sha256=sha(log), kernels=kernels,
                                         spill=any(row['spill'] for row in kernels.values())))
        for process in range(50):
            log = args.out / f'process-{process:02d}.log'
            with log.open('w') as stream:
                status = subprocess.run([str(args.out / 'dependency-sm_89')], stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=60).returncode
            result['processes'].append(dict(process=process, exit_code=status, log_sha256=sha(log)))
            if status:
                raise RuntimeError(f'fresh process {process} failed; see {log}')
        if hashes != {str(path): sha(path) for path in sorted(dependencies)}:
            raise RuntimeError('transitive inputs changed during validation')
        result['passed'] = True
    finally:
        passes = sum(row['exit_code'] == 0 for row in result['processes'])
        result['pass_rates'] = {path: dict(passes=passes, attempts=len(result['processes']), required=50)
                               for path in ('table', 'counted', 'weighted_la')}
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Dependency primitives: five targets compiled; table/counted/weighted-LA each passed 50/50 fresh processes', flush=True)


if __name__ == '__main__':
    main()
