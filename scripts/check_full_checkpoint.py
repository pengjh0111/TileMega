#!/usr/bin/env python3
"""Build and run the complete CTest suite from a frozen R15 worktree."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


FIXTURES = [
    'docs/experiments/SOLVER_R9B/fit/target.json',
    'docs/experiments/SERVING_R12C/target_r12b.json',
    'docs/experiments/DNN_MOE_R1/inputs/regression/llama_B1/prefill/target.json',
    'docs/experiments/MODELS/covered_llama/auto.cu.export.json',
    'docs/experiments/E2E_GEN/raw/export_bridge.json',
    'docs/experiments/P3_ISL/crosslink_probe.cpp',
    'docs/experiments/SEQSCAN/raw/export/gqa2.json',
    'docs/experiments/SEQSCAN/raw/export/mha4.json',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def prepare(root, out, tree):
    if out.exists():
        raise FileExistsError('full checkpoint already exists')
    out.mkdir(parents=True)
    head = git(root, 'rev-parse', 'HEAD')
    subprocess.run(['git', '-C', str(root), 'worktree', 'add', '--detach', '--no-checkout',
                    str(tree), head], check=True)
    subprocess.run(['git', '-C', str(tree), 'sparse-checkout', 'init', '--cone'], check=True)
    subprocess.run(['git', '-C', str(tree), 'sparse-checkout', 'set',
                    'include', 'lib', 'tools', 'cmake', 'python', 'scripts', 'test', 'configs'], check=True)
    subprocess.run(['git', '-C', str(tree), 'read-tree', '-mu', 'HEAD'], check=True)
    third = tree / 'third_party'
    if third.exists() and not any(third.iterdir()):
        third.rmdir()
    third.symlink_to(root / 'third_party', target_is_directory=True)
    files = set(FIXTURES)
    for folder in ('docs/experiments/V_I', 'docs/experiments/SERVING_R11',
                   'docs/experiments/DNN_MOE_R1'):
        files.update(name for name in git(root, 'ls-files', folder).splitlines()
                     if Path(name).parent == Path(folder) and Path(name).suffix in ('.py', '.sh', '.txt'))
    files.add('docs/experiments/V_I/CMakeLists.txt')
    for name in files:
        destination = tree / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(subprocess.check_output(['git', '-C', str(root), 'show', head + ':' + name]))
        destination.chmod((root / name).stat().st_mode)
    shutil.copy2(__file__, out / 'runner.py')
    (out / 'inputs.json').write_text(json.dumps(dict(root=str(root), tree=str(tree), head=head,
        runner_sha256=sha(out / 'runner.py'), fixtures={name:sha(tree / name) for name in files},
        cutlass_head=git(root / 'third_party/cutlass', 'rev-parse', 'HEAD')), indent=2) + '\n')
    print('Prepared complete CTest checkpoint at ' + head)


def validate(out):
    record = json.loads((out / 'inputs.json').read_text())
    tree = Path(record['tree'])
    if sha(out / 'runner.py') != record['runner_sha256'] or git(tree, 'rev-parse', 'HEAD') != record['head']:
        raise ValueError('frozen CTest checkpoint changed')
    if git(tree, 'diff', '--name-only', 'HEAD'):
        raise ValueError('frozen CTest source changed')
    if any(sha(tree / name) != expected for name,expected in record['fixtures'].items()):
        raise ValueError('frozen historical test fixture changed')
    if git(Path(record['root']) / 'third_party/cutlass', 'rev-parse', 'HEAD') != record['cutlass_head']:
        raise ValueError('CUTLASS changed')
    return record


def execute(command, out, label, timeout, env=None):
    (out / (label + '.command.json')).write_text(json.dumps(command) + '\n')
    with (out / (label + '.log')).open('w') as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{label} exited {result.returncode}; see {out}/{label}.log')


def build(out):
    record = validate(out)
    root, tree = Path(record['root']), Path(record['tree'])
    build_dir = tree / 'build-r15'
    command = ['cmake', '-S', str(tree), '-B', str(build_dir), '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', '-DTILEMEGA_TARGET_ARCH=sm_89',
        '-DTILEMEGA_BUILD_TESTS=ON', '-DTILEMEGA_BUILD_VERIFY=ON',
        '-DTILEMEGA_BUILD_EXPERIMENTAL=OFF',
        '-DMLIR_DIR=/root/toolchains/mlir-23a60f15/lib/cmake/mlir',
        '-DPython3_EXECUTABLE=' + sys.executable]
    for dep in ('ISL', 'POLYLIB', 'BARVINOK'):
        command.append('-DTILEMEGA_' + dep + '_BUILD_DIR=' + str(root / ('build-' + dep.lower())))
    with Path(os.environ['TILEMEGA_GPU_LOCK']).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        execute(command, out, 'configure', 600)
        execute(['cmake', '--build', str(build_dir), '-j', '3'], out, 'build', 43200)
        execute(['ctest', '--test-dir', str(build_dir), '--show-only=json-v1'], out, 'tests', 60)
    validate(out)
    print('PASS complete CTest build', flush=True)


def run(out):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise RuntimeError('full CTest requires the scheduler GPU guard')
    record = validate(out)
    tree = Path(record['tree'])
    env = dict(os.environ, PYTHONPATH=str(tree / 'python'),
               TILEMEGA_BIN=str(tree / 'build-r15/tools/tilemega'))
    execute(['ctest', '--test-dir', str(tree / 'build-r15'), '--output-on-failure',
             '--timeout', '1200', '-j', '1'], out, 'ctest', 21600, env)
    validate(out)
    print('PASS complete CTest suite', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'build', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tree', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.tree is None:
            parser.error('prepare requires --tree')
        prepare(Path(__file__).resolve().parents[1], args.out.resolve(), args.tree.resolve())
    else:
        {'build':build, 'run':run}[args.action](args.out.resolve())


if __name__ == '__main__':
    main()
