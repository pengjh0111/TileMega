#!/usr/bin/env python3
"""Freeze a compiler and run the CPU-only LLM/DM switch generation checks."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, out, tree):
    if out.exists():
        raise FileExistsError('feature checkpoint already exists')
    out.mkdir(parents=True)
    head = git(root, 'rev-parse', 'HEAD')
    subprocess.run(['git', '-C', str(root), 'worktree', 'add', '--detach',
                    '--no-checkout', str(tree), head], check=True)
    subprocess.run(['git', '-C', str(tree), 'sparse-checkout', 'init', '--cone'], check=True)
    subprocess.run(['git', '-C', str(tree), 'sparse-checkout', 'set',
                    'include', 'lib', 'tools', 'cmake', 'python', 'scripts', 'test', 'configs'], check=True)
    subprocess.run(['git', '-C', str(tree), 'read-tree', '-mu', 'HEAD'], check=True)
    third_party = tree / 'third_party'
    if third_party.is_dir() and not any(third_party.iterdir()):
        third_party.rmdir()
    third_party.symlink_to(root / 'third_party', target_is_directory=True)
    shutil.copy2(__file__, out / 'runner.py')
    record = dict(root=str(root), tree=str(tree), head=head,
                  runner_sha256=sha(out / 'runner.py'),
                  cutlass_head=git(root / 'third_party/cutlass', 'rev-parse', 'HEAD'))
    (out / 'source.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Prepared frozen source-generation checks at ' + head)


def validate(out):
    record = json.loads((out / 'source.json').read_text())
    tree = Path(record['tree'])
    if sha(out / 'runner.py') != record['runner_sha256'] or git(tree, 'rev-parse', 'HEAD') != record['head']:
        raise ValueError('feature checkpoint changed')
    if git(tree, 'diff', '--name-only', 'HEAD'):
        raise ValueError('feature checkpoint source modified')
    if git(Path(record['root']) / 'third_party/cutlass', 'rev-parse', 'HEAD') != record['cutlass_head']:
        raise ValueError('CUTLASS changed')
    return record


@contextlib.contextmanager
def locked():
    with Path(os.environ['TILEMEGA_GPU_LOCK']).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def execute(command, cwd, log, timeout, env=None):
    Path(str(log) + '.command.json').write_text(json.dumps(command) + '\n')
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'exit {result.returncode}: {log}')


def build(out):
    record = validate(out)
    tree, root = Path(record['tree']), Path(record['root'])
    build_dir = tree / 'build-r15'
    command = ['cmake', '-S', str(tree), '-B', str(build_dir), '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', '-DTILEMEGA_TARGET_ARCH=sm_89',
        '-DTILEMEGA_BUILD_TESTS=OFF', '-DTILEMEGA_BUILD_VERIFY=OFF',
        '-DTILEMEGA_BUILD_EXPERIMENTAL=OFF',
        '-DMLIR_DIR=/root/toolchains/mlir-23a60f15/lib/cmake/mlir']
    for dep in ('ISL', 'POLYLIB', 'BARVINOK'):
        command += ['-DTILEMEGA_' + dep + '_BUILD_DIR=' + str(root / ('build-' + dep.lower()))]
    with locked():
        execute(command, tree, out / 'configure.log', 600)
        execute(['cmake', '--build', str(build_dir), '--target', 'tilemega', '-j', '6'],
                tree, out / 'compiler.log', 7200)
    (out / 'compiler.json').write_text(json.dumps(dict(
        head=record['head'], path=str(build_dir / 'tools/tilemega'),
        sha256=sha(build_dir / 'tools/tilemega')), indent=2) + '\n')
    print('PASS frozen feature compiler build')


def check(out):
    record = validate(out)
    compiler = json.loads((out / 'compiler.json').read_text())
    if sha(compiler['path']) != compiler['sha256']:
        raise ValueError('frozen feature compiler changed')
    tree = Path(record['tree'])
    env = dict(os.environ, TILEMEGA_BIN=compiler['path'], PYTHONPATH=str(tree / 'python'),
               TILEMEGA_FEATURE_RESULTS=str(out / 'generated'))
    with locked():
        execute([sys.executable, '-m', 'unittest', 'discover', '-s', str(tree / 'test/python'),
                 '-p', 'test_generated_features.py', '-v'], tree, out / 'checks.log', 14400, env)
    print('PASS actual LLM/DM feature generation and rejection checks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'build', 'check'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--tree', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.tree is None:
            parser.error('prepare requires --tree')
        prepare(args.root.resolve(), args.out.resolve(), args.tree.resolve())
    elif args.action == 'build':
        build(args.out.resolve())
    else:
        check(args.out.resolve())


if __name__ == '__main__':
    main()
