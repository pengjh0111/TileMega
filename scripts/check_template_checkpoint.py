#!/usr/bin/env python3
"""Freeze and check a template checkpoint while RF-1 uses old binaries."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


CASES = [
    ('epilogue', 'serving_epilogue_test.cu', [], []),
    ('gemm', 'serving_gemm_task_body_test.cu', [], []),
    ('gemm_trace', 'serving_gemm_task_body_test.cu', ['TILEMEGA_TRACE_TASK=1'], []),
    ('paged_gemm', 'paged_gemm_test.cu', [], []),
    ('paged_gemm_pipe', 'paged_gemm_test.cu', ['TILEMEGA_MMA_REG_PIPE=1'], []),
    ('dm_epilogue_value', 'dm_epilogue_value_test.cu', [], ['epilogue_values.bin']),
    ('dm_epilogue_tile', 'dm_epilogue_tile_test.cu', [], ['epilogue_tiles.bin']),
    ('dm_chain', 'dm_gemm_chain_test.cu', [], ['gemm_chains.bin']),
    ('dm_chain_pipe', 'dm_gemm_chain_test.cu', ['TILEMEGA_MMA_REG_PIPE=1'], ['gemm_chains.bin']),
    ('dm_chain_trace', 'dm_gemm_chain_test.cu', ['TILEMEGA_TRACE_TASK=1'], ['gemm_chains.bin']),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(out):
    record = json.loads((out / 'source.json').read_text())
    if sha(out / 'runner.py') != record['runner_sha256']:
        raise ValueError('frozen checkpoint runner changed')
    for name, expected in record['files'].items():
        if sha(out / 'source' / name) != expected:
            raise ValueError('frozen input changed: ' + name)
    return record


def execute(command, out, label):
    (out / (label + '.command.json')).write_text(json.dumps(command) + '\n')
    with (out / (label + '.log')).open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise RuntimeError(f'{label} failed ({result.returncode}); see {out / (label + ".log")}')


def prepare(root, out):
    source = out / 'source'
    if source.exists():
        raise FileExistsError('checkpoint inputs already exist')
    shutil.copytree(root / 'include', source / 'include')
    (source / 'test/unit').mkdir(parents=True)
    for name in {row[1] for row in CASES}:
        shutil.copy2(root / 'test/unit' / name, source / 'test/unit' / name)
    shutil.copytree(root / 'test/fixtures/dnn', source / 'test/fixtures/dnn')
    shutil.copy2(Path(__file__), out / 'runner.py')
    record = dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        runner_sha256=sha(out / 'runner.py'),
        files={str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()},
        cutlass=str(root / 'third_party/cutlass'),
        cutlass_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=root / 'third_party/cutlass', text=True).strip())
    (out / 'source.json').write_text(json.dumps(record, indent=2) + '\n')


def build(out):
    record = validate(out)
    source = out / 'source'
    cutlass = Path(record['cutlass'])
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cutlass, text=True).strip() != record['cutlass_head']:
        raise ValueError('CUTLASS revision changed after checkpoint preparation')
    common = [os.getenv('NVCC', '/usr/local/cuda/bin/nvcc'), '-std=c++17', '-O3',
        '--expt-relaxed-constexpr', '-UNDEBUG', '-Xptxas=-v',
        '-I' + str(source / 'include'), '-I' + str(cutlass / 'include'),
        '-I' + str(cutlass / 'tools/util/include')]
    executables = []
    lock = Path(os.environ['TILEMEGA_GPU_LOCK'])
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for label, file, definitions, fixtures in CASES:
            binary = out / label
            execute([*common, '-arch=sm_89', '-DTILEMEGA_ARCH_ID=890',
                *['-D' + value for value in definitions], str(source / 'test/unit' / file),
                '-o', str(binary)], out, 'build_' + label)
            executables.append(dict(label=label, binary=str(binary), sha256=sha(binary),
                args=[str(source / 'test/fixtures/dnn' / name) for name in fixtures]))
        for arch in (80, 89, 90, 100, 120):
            execute([*common, '-arch=sm_' + str(arch), '-DTILEMEGA_ARCH_ID=' + str(arch * 10),
                '-c', str(source / 'test/unit/dm_gemm_chain_test.cu'),
                '-o', str(out / f'dm_chain_sm{arch}.o')], out, f'compile_sm{arch}')
    validate(out)
    (out / 'build.json').write_text(json.dumps(dict(source_head=record['head'],
        executables=executables, arches=[80, 89, 90, 100, 120]), indent=2) + '\n')


def run(out):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise RuntimeError('numerical checks require the scheduler GPU guard')
    validate(out)
    build_record = json.loads((out / 'build.json').read_text())
    passed = []
    for row in build_record['executables']:
        if sha(row['binary']) != row['sha256']:
            raise ValueError('checkpoint binary changed: ' + row['label'])
        execute([row['binary'], *row['args']], out, 'numeric_' + row['label'])
        passed.append(row['label'])
    (out / 'numeric.json').write_text(json.dumps(dict(passed=passed,
        source_head=build_record['source_head'], scope='component regression; full G-REG-U/G-DM-U pending')) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'build', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.action == 'prepare':
        prepare(Path(__file__).resolve().parents[1], out)
    elif args.action == 'build':
        build(out)
    else:
        run(out)
    print(f'template checkpoint {args.action} PASS', flush=True)


if __name__ == '__main__':
    main()
