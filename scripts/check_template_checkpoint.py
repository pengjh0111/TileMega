#!/usr/bin/env python3
"""Freeze and check a template checkpoint while RF-1 uses old binaries."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
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
ATTENTION_CASES = [
    ('layout', 'attention_layout_test.cu', [], []),
    ('layout_pvswap', 'attention_layout_test.cu', ['TILEMEGA_ATTENTION_PVSWAP=1'], []),
    ('pvswap', 'serving_pvswap_test.cu', [], []),
    *[(f'q8_mode{mode}', 'dm_qperkv8_attention_test.cu',
       [f'DM_TEST_ATTN_MODE={mode}', 'TILEMEGA_DM_SUPPORT=1'], []) for mode in range(5)],
    ('q8_prefill', 'dm_qperkv8_prefill_test.cu', [], []),
    ('epoch', 'dm_epoch_last_arriver_test.cu', [], []),
    ('monotonic', 'monotonic_last_arriver_test.cu', [], []),
    ('counted', 'dm_counted_threshold_test.cu', [], []),
]
RUNTIME_CASES = [
    ('combine', 'dm_nonpaged_handoff_test.cu', [], []),
    ('combine_decode', 'dm_nonpaged_handoff_test.cu', ['DM_TEST_PHASE=1'], []),
    ('attention_merge', 'dm_nonpaged_attention_handoff_test.cu', [], []),
    ('small_combine', 'dm_small_native_gemm_test.cu', [], []),
    ('binding_stream', 'dm_binding_stream_test.cu', [], []),
    ('binding_pages', 'dm_binding_pages_test.cu', [], []),
]
GEMV_CASES = [
    ('epilogues', 'serving_gemv_test.cu', [], []),
    ('access', 'serving_gemv_access_test.cu', [], []),
    ('profile', 'serving_task_profile_test.cu', ['TILEMEGA_TRACE_TASK=1'], []),
]
DM_CASES = [(name, 'dm_' + name + '_test.cu',
             ['TILEMEGA_DM_SUPPORT=1'] if name == 'descriptor_device' else [], []) for name in (
    'conv_operands', 'conv_task', 'norm_embedding_layout', 'pool_global',
    'depthwise', 'dwpw_fused', 'encoder_attention', 'descriptor_device',
    'small_task_gemm', 'small_gemm', 'multipage_gemm', 'scaled_dense',
    'stage_dependencies', 'dependency', 'counted_stage_threshold',
    'moe_rms_stats', 'moe_expert_gemm', 'moe_routing_combine',
    'moe_counted_publish')]
DM_FORWARD_CASES = [
    (f'forward_paged{paged}_split{split}', 'dm_dnn_runtime_test.cu',
     [f'DM_TEST_PAGED={paged}', f'DM_TEST_SPLIT={split}'], [])
    for paged in (0, 1) for split in (1, 5)
] + [
    ('forward_reject_split2', 'dm_dnn_runtime_test.cu', ['DM_TEST_SPLIT=2'], [])
] + [
    (f'forward_embedding_paged{paged}', 'dm_dnn_runtime_test.cu',
     [f'DM_TEST_PAGED={paged}', 'DM_TEST_EMBEDDING=1'], []) for paged in (0, 1)
] + [
    (f'{name}_phase{phase}', 'dm_' + name + '_test.cu',
     [f'DM_TEST_PHASE={phase}'], [])
    for name in ('paged_phase', 'paged_multipage_phase') for phase in (0, 2)
]
SUITES = {'gemm': CASES, 'attention': ATTENTION_CASES, 'runtime': RUNTIME_CASES,
          'gemv': GEMV_CASES, 'dm': DM_CASES, 'dm_forward': DM_FORWARD_CASES}
LINKED_SUITES = {'runtime', 'dm', 'dm_forward'}
ARCH_SOURCES = {'attention': ['serving_pvswap_test.cu', 'dm_qperkv8_prefill_test.cu'],
                'runtime': ['dm_nonpaged_attention_handoff_test.cu', 'dm_binding_stream_test.cu'],
                'gemm': ['dm_gemm_chain_test.cu'],
                'gemv': ['dm_gemv_compile.cu', 'serving_task_profile_test.cu'],
                'dm': ['dm_moe_counted_publish_test.cu', 'dm_dwpw_fused_test.cu',
                       'dm_stage_dependencies_test.cu'],
                'dm_forward': ['dm_dnn_runtime_test.cu']}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(out):
    record = json.loads((out / 'source.json').read_text())
    if sha(out / 'runner.py') != record['runner_sha256']:
        raise ValueError('frozen checkpoint runner changed')
    for name, expected in record['files'].items():
        if sha(out / 'source' / name) != expected:
            raise ValueError('frozen input changed: ' + name)
    for name, expected in record.get('link_inputs', {}).items():
        if sha(name) != expected:
            raise ValueError('checkpoint link input changed: ' + name)
    return record


def execute(command, out, label):
    (out / (label + '.command.json')).write_text(json.dumps(command) + '\n')
    with (out / (label + '.log')).open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise RuntimeError(f'{label} failed ({result.returncode}); see {out / (label + ".log")}')


def prepare(root, out, suite, build_dir, labels, arches):
    source = out / 'source'
    if source.exists():
        raise FileExistsError('checkpoint inputs already exist')
    shutil.copytree(root / 'include', source / 'include')
    (source / 'test/unit').mkdir(parents=True)
    selected = set(labels.split(',')) if labels else {row[0] for row in SUITES[suite]}
    cases = [row for row in SUITES[suite] if row[0] in selected]
    if {row[0] for row in cases} != selected:
        raise ValueError('unknown checkpoint case labels')
    names = {row[1] for row in cases}
    if arches:
        names.update(ARCH_SOURCES[suite])
    for name in names:
        shutil.copy2(root / 'test/unit' / name, source / 'test/unit' / name)
    shutil.copytree(root / 'test/fixtures/dnn', source / 'test/fixtures/dnn')
    shutil.copy2(Path(__file__), out / 'runner.py')
    record = dict(suite=suite, cases=cases, arches=arches,
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        runner_sha256=sha(out / 'runner.py'),
        files={str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()},
        cutlass=str(root / 'third_party/cutlass'),
        cutlass_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=root / 'third_party/cutlass', text=True).strip())
    if suite in LINKED_SUITES:
        build_dir = build_dir.resolve()
        line = subprocess.check_output(['ninja', '-C', str(build_dir), '-t',
            'commands', 'serving_epilogue_test'], text=True).splitlines()[-1]
        tokens = shlex.split(line)
        command = tokens[tokens.index('&&') + 1:tokens.index('&&', tokens.index('&&') + 1)]
        link_inputs = {}
        for i, token in enumerate(command):
            if token.endswith('.o'):
                command[i] = '{object}'
            elif token == '-o':
                command[i + 1] = '{binary}'
            elif token.endswith(('.a', '.so')):
                path = (build_dir / token).resolve()
                if path.name == 'libtilemega.a':
                    copy = out / path.name
                    shutil.copy2(path, copy)
                    path = copy
                command[i] = str(path)
                link_inputs[str(path)] = sha(path)
        record.update(link_command=command, link_inputs=link_inputs)
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
    executables, failures = [], []
    lock = Path(os.environ['TILEMEGA_GPU_LOCK'])
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for label, file, definitions, fixtures in record.get('cases', SUITES[record['suite']]):
            binary = out / label
            command = [*common, '-arch=sm_89', '-DTILEMEGA_ARCH_ID=890',
                *['-D' + value for value in definitions], str(source / 'test/unit' / file),
                '-o', str(binary)]
            try:
                if record['suite'] in LINKED_SUITES:
                    obj = binary.with_suffix('.o')
                    command[-1] = str(obj)
                    execute([*command, '-c'], out, 'compile_' + label)
                    execute([str(obj) if token == '{object}' else str(binary) if token == '{binary}'
                        else token for token in record['link_command']], out, 'link_' + label)
                else:
                    execute(command, out, 'build_' + label)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                failures.append(dict(label=label, error=str(error)))
                continue
            executables.append(dict(label=label, binary=str(binary), sha256=sha(binary),
                processes=5 if record['suite'] in LINKED_SUITES or label in ('epoch', 'monotonic', 'counted', 'access') else 1,
                args=[str(source / 'test/fixtures/dnn' / name) for name in fixtures]))
        for arch in record.get('arches', [80, 89, 90, 100, 120]):
            for file in ARCH_SOURCES[record['suite']]:
                label = f'{Path(file).stem}_sm{arch}'
                try:
                    execute([*common, '-arch=sm_' + str(arch), '-DTILEMEGA_ARCH_ID=' + str(arch * 10),
                        '-c', str(source / 'test/unit' / file),
                        '-o', str(out / (label + '.o'))], out, 'compile_' + label)
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    failures.append(dict(label=label, error=str(error)))
    validate(out)
    (out / 'build.json').write_text(json.dumps(dict(source_head=record['head'],
        executables=executables, failures=failures,
        arches=record.get('arches', [80, 89, 90, 100, 120])), indent=2) + '\n')
    if failures:
        raise RuntimeError(f'{len(failures)} compile checks failed; see {out / "build.json"}')


def run(out):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise RuntimeError('numerical checks require the scheduler GPU guard')
    validate(out)
    build_record = json.loads((out / 'build.json').read_text())
    passed, failures = [], list(build_record.get('failures', []))
    for row in build_record['executables']:
        if sha(row['binary']) != row['sha256']:
            raise ValueError('checkpoint binary changed: ' + row['label'])
        successes = 0
        for process in range(row.get('processes', 1)):
            try:
                execute([row['binary'], *row['args']], out, f'numeric_{row["label"]}_{process}')
                successes += 1
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                failures.append(dict(label=row['label'], process=process, error=str(error)))
        passed.append(dict(label=row['label'], passed=successes,
                           processes=row.get('processes', 1)))
    (out / 'numeric.json').write_text(json.dumps(dict(passed=passed, failures=failures,
        source_head=build_record['source_head'], scope='component regression; full G-REG-U/G-DM-U pending')) + '\n')
    if failures:
        raise RuntimeError(f'{len(failures)} numerical processes failed; see {out / "numeric.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'build', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--suite', choices=SUITES, default='gemm')
    parser.add_argument('--build-dir', type=Path, default=Path('build-phase12'))
    parser.add_argument('--labels', default='', help='prepare only these comma-separated cases')
    parser.add_argument('--arches', default='80,89,90,100,120',
                        help='compile architectures; empty for a targeted numerical retry')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.action == 'prepare':
        arches = [int(value) for value in args.arches.split(',') if value]
        if any(value not in (80, 89, 90, 100, 120) for value in arches):
            parser.error('unsupported compile architecture')
        prepare(Path(__file__).resolve().parents[1], out, args.suite, args.build_dir, args.labels, arches)
    elif args.action == 'build':
        build(out)
    else:
        run(out)
    print(f'template checkpoint {args.action} PASS', flush=True)


if __name__ == '__main__':
    main()
