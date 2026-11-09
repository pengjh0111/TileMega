#!/usr/bin/env python3
"""Build and guard focused F3 convolution, GAP, gather and split-K gates.

This covers small adversarial shapes, not the complete §6.8 P-UNIT table.
All subprocesses in a cohort hold the same scheduler/guard reservation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(os.environ.get('TILEMEGA_DNN4_REPO', Path(__file__).resolve().parents[1]))
SOURCES = dict(ffn1_tma256=ROOT / 'test/unit/dnn4_store_bert_test.cu',
               ffn1_tma128=ROOT / 'test/unit/dnn4_store_bert_test.cu',
               ffn2_reg64=ROOT / 'test/unit/dnn4_store_bert_test.cu',
               ffn1_reg64=ROOT / 'test/unit/dnn4_store_bert_test.cu',
               ffn2_store=ROOT / 'test/unit/dnn4_store_test.cu',
               ffn2_store_bert=ROOT / 'test/unit/dnn4_store_bert_test.cu',
               pp_store=ROOT / 'test/unit/dnn4_store_test.cu',
               pp_store_bert=ROOT / 'test/unit/dnn4_store_bert_test.cu',
               repeat_odd=ROOT / 'test/unit/dnn4_repeat_test.cu',
               partial=ROOT / 'test/unit/dnn4_bert_test.cu',
               repeat_alias=ROOT / 'test/unit/dnn4_repeat_test.cu',
               bert_side_norm=ROOT / 'test/unit/dnn4_bert_test.cu',
               pingpong128=ROOT / 'test/unit/dnn4_repeat_test.cu',
               pingpong=ROOT / 'test/unit/dnn4_repeat_test.cu',
               repeat=ROOT / 'test/unit/dnn4_repeat_test.cu',
               multicast=ROOT / 'test/unit/dnn4_multicast_test.cu',
               store_bert=ROOT / 'test/unit/dnn4_store_bert_test.cu',
               store=ROOT / 'test/unit/dnn4_store_test.cu',
               stem=ROOT / 'test/unit/dnn4_stem_test.cu',
               body=ROOT / 'test/unit/dnn4_f3_test.cu',
               attention=ROOT / 'test/unit/dnn4_attention_test.cu',
               bert=ROOT / 'test/unit/dnn4_bert_test.cu',
               registers=ROOT / 'test/unit/dnn4_register_test.cu',
               legal_registers=ROOT / 'test/unit/dnn4_legal_register_test.cu',
               pipeline=ROOT / 'test/unit/dnn4_pipeline_test.cu')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def inputs(suite):
    paths = [Path(__file__).resolve(), SOURCES[suite], *sorted((ROOT / 'include').rglob('*.h')),
             *sorted((ROOT / 'include').rglob('*.cuh'))]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def execute(options):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1' or not os.environ.get('TILEMEGA_RUN_NONCE'):
        raise ValueError('the cohort requires the GPU scheduler and guard')
    out = options.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    build = json.loads(options.build.read_text())
    binary = Path(build['binary'])
    if sha(binary) != build['binary_sha256'] or any(sha(path) != value for path, value in build['inputs'].items()):
        raise ValueError('unit binary or source differs from its build receipt')
    result = dict(evidence='verified', build=str(options.build), passed=False,
                  scope='focused F3 ' + options.case + ' regressions; not complete P-UNIT',
                  gpu=int(os.environ['TILEMEGA_DEVICE_INDEX']), processes=[], started=time.time())
    hashes = []

    def run(name, command):
        log = out / (name + '.log')
        with log.open('w') as stream:
            done = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        row = dict(name=name, command=command, exit_code=done.returncode, log=str(log), sha256=sha(log))
        result['processes'].append(row)
        save(out / 'result.json', result)
        return done.returncode, log.read_text()

    for index in range(5):
        code, text = run(f'unit-{index+1}', [str(binary), options.case])
        if code:
            result['stop'] = 'P-UNIT_failed'
            save(out / 'result.json', result)
            return 1
        match = re.findall(r'^output_hash=([0-9a-f]+)$', text, re.MULTILINE)
        if len(match) != 1:
            raise ValueError('unit process lacks one complete output hash')
        hashes += match
    if len(set(hashes)) != 1:
        result['stop'] = 'P-UNIT_fresh_process_bitwise_mismatch'
        save(out / 'result.json', result)
        return 1
    result['unit_passes'] = '5/5'
    result['output_hash'] = hashes[0]
    cases = ((options.suite,) if options.suite != 'body' else
             ('gather', 'pool', 'split', 'conv', 'swap', 'small') if options.case == 'all'
             else (options.case,))
    for case in cases:
        for tool, count in (('memcheck', 1), ('initcheck', 1), ('racecheck', 5), ('synccheck', 5)):
            for index in range(count):
                name = f'{case}-{tool}-{index+1}'
                raw = out / (name + '.sanitizer.log')
                command = ['/root/TileMega-dnn3/runs/dnn3/sanitizer_12_9/unpacked/cuda_sanitizer_api-linux-x86_64-12.9.79-archive/bin/compute-sanitizer', '--tool', tool, '--error-exitcode', '86',
                           '--log-file', str(raw), str(binary), case]
                code, _ = run(name, command)
                row = result['processes'][-1]
                row.update(sanitizer_log=str(raw), sanitizer_sha256=sha(raw) if raw.exists() else None)
                text = raw.read_text() if raw.exists() else ''
                clean = (code == 0 and (('ERROR SUMMARY: 0 errors' in text) or
                         (tool == 'racecheck' and '0 hazards displayed' in text)))
                if not clean:
                    result['stop'] = ('U3_gather_protocol' if case == 'gather' and
                                      tool in ('racecheck', 'synccheck') else 'P-SAN_failed')
                    save(out / 'result.json', result)
                    return 1
    if any(sha(path) != value for path, value in build['inputs'].items()) or sha(binary) != build['binary_sha256']:
        raise ValueError('unit inputs changed during the cohort')
    result.update(passed=True, finished=time.time())
    save(out / 'result.json', result)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cpu-set', default='120-127', help='NUMA0 functional worker cores')
    parser.add_argument('--build', type=Path)
    parser.add_argument('--suite', choices=tuple(SOURCES), default='body')
    parser.add_argument('--register-budget', type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument('--cyclic-slots', action='store_true')
    parser.add_argument('--coalesced-partials', action='store_true')
    parser.add_argument('--matrix-store', action='store_true')
    parser.add_argument('--attention-vector-store', action='store_true')
    parser.add_argument('--repeat-arithmetic', action='store_true')
    parser.add_argument('--case', choices=('all', 'gather', 'pool', 'split', 'conv', 'swap', 'small'), default='all')
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args()
    if options.register_budget and options.suite not in ('repeat', 'repeat_odd', 'repeat_alias', 'bert_side_norm', 'pp_store', 'pp_store_bert', 'ffn2_store', 'ffn2_store_bert', 'ffn1_reg64', 'ffn2_reg64', 'ffn1_tma128', 'ffn1_tma256'):
        parser.error('--register-budget requires a Repeat or BERT side-normalization suite')
    if options.repeat_arithmetic and options.suite not in ('repeat', 'repeat_odd', 'repeat_alias'):
        parser.error('--repeat-arithmetic requires a Repeat suite')
    if options.attention_vector_store and options.suite != 'attention':
        parser.error('--attention-vector-store requires the attention suite')
    if options.execute:
        return execute(options)
    out = options.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = out / 'dnn4_f3_test'
    log = out / 'ptxas.log'
    before = inputs(options.suite)
    source = out / 'source'
    source.mkdir()
    shutil.copytree(ROOT / 'include', source / 'include')
    frozen_unit = source / SOURCES[options.suite].name
    shutil.copyfile(SOURCES[options.suite], frozen_unit)
    frozen_runner = source / 'cohort.py'
    shutil.copyfile(Path(__file__), frozen_runner)
    frozen = {str(path): sha(path) for path in source.rglob('*') if path.is_file()}
    if before != inputs(options.suite):
        raise ValueError('unit inputs changed while freezing')
    command = ['taskset', '-c', '136-143', 'nvcc', '-std=c++17', '-O3',
               '-gencode', 'arch=compute_90a,code=sm_90a', '--expt-relaxed-constexpr',
               '-Xptxas', '-v', '-Xptxas', '--device-function-maxrregcount=232',
               '-I' + str(source / 'include'), '-I' + str(ROOT / 'third_party/cutlass/include'),
               str(frozen_unit), '-o', str(binary)]
    if options.suite in ('pingpong', 'pingpong128'):
        command.insert(-4, '-DDNN4_PP_TEST=1')
    if options.suite == 'pingpong128':
        command.insert(-4, '-DDNN4_PP_TILE_M=256')
    if options.suite in ('pp_store', 'pp_store_bert'):
        command.insert(-4, '-DDNN4_PP_STORE_TEST=1')
        command.insert(-4, '-DTILEMEGA_F3_SIDE_NORM=1')
    if options.suite in ('ffn2_store', 'ffn2_store_bert'):
        command.insert(-4, '-DDNN4_FFN2_STORE_TEST=1')
        command.insert(-4, '-DTILEMEGA_F3_SIDE_NORM=1')
    if options.suite == 'ffn1_tma256':
        command.insert(-4, '-DDNN4_FFN1_TMA256_TEST=1')
        command.insert(-4, '-DTILEMEGA_F3_SIDE_NORM=1')
    if options.suite == 'ffn1_tma128':
        command.insert(-4, '-DDNN4_FFN1_TMA128_TEST=1')
        command.insert(-4, '-DTILEMEGA_F3_SIDE_NORM=1')
    if options.suite == 'ffn1_reg64':
        command.insert(-4, '-DDNN4_FFN1_REG64_TEST=1')
        command.insert(-4, '-DTILEMEGA_F3_SIDE_NORM=1')
    if options.suite == 'ffn2_reg64':
        command.insert(-4, '-DDNN4_FFN2_REG64_TEST=1')
        command.insert(-4, '-DTILEMEGA_F3_SIDE_NORM=1')
    if options.suite == 'repeat_odd':
        command.insert(-4, '-DDNN4_ODD_REPEAT_TEST=1')
    if options.suite == 'repeat_alias':
        command.insert(-4, '-DDNN4_ALIAS_REPEAT_TEST=1')
    if options.register_budget:
        command.insert(-4, '-DTILEMEGA_F3_REGISTER_BUDGET=' + str(options.register_budget))
    if options.cyclic_slots:
        command.insert(-4, '-DTILEMEGA_F3_CYCLIC_SLOTS=1')
    if options.coalesced_partials:
        command.insert(-4, '-DTILEMEGA_F3_COALESCED_PARTIALS=1')
    if options.matrix_store:
        command.insert(-4, '-DTILEMEGA_F3_MATRIX_STORE=1')
    if options.attention_vector_store:
        command.insert(-4, '-DTILEMEGA_F3_ATTENTION_VECTOR_STORE=1')
    if options.suite == 'partial':
        command.insert(-4, '-DDNN4_PARTIAL_LAYOUT_TEST=1')
    if options.repeat_arithmetic:
        command.insert(-4, '-DTILEMEGA_F3_REPEAT_ARITHMETIC=1')
    if options.suite == 'bert_side_norm':
        command.insert(-4, '-DTILEMEGA_F3_SIDE_NORM=1')
    with log.open('w') as stream:
        done = subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                              stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
    text = log.read_text()
    spills = [int(x) for x in re.findall(r'(\d+) bytes spill (?:stores|loads)', text)]
    registers = [int(x) for x in re.findall(r'Used (\d+) registers', text)]
    if done.returncode or before != inputs(options.suite) or not spills or any(spills) or not registers or max(registers) > 232 or re.search(r'C750[47]', text) is not None:
        raise RuntimeError('F3 unit build failed DT-SPILL/source stability; retain ' + str(log))
    build = out / 'build.json'
    save(build, dict(binary=str(binary), binary_sha256=sha(binary), inputs=frozen, original_inputs=before,
                     command=command, log=str(log), log_sha256=sha(log), registers=registers,
                     spill_bytes=max(spills), cutlass_commit=subprocess.check_output(
                         ['git', '-C', str(ROOT / 'third_party/cutlass'), 'rev-parse', 'HEAD'], text=True).strip()))
    queue = out / 'queue'
    queue.mkdir()
    gpu = str(options.device)
    command = ['taskset', '-c', options.cpu_set, sys.executable, str(frozen_runner),
               '--execute', '--build', str(build), '--out', '{attempt_dir}/cohort', '--case', options.case,
               '--suite', options.suite]
    environment = dict(CUDA_VISIBLE_DEVICES=gpu, TILEMEGA_DEVICE_INDEX=gpu,
                       TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', TILEMEGA_NUMA_NODE='0',
                       TILEMEGA_GPU_PID_MODE='namespace_limited', TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
                       PYTHONPATH=str(ROOT / 'python'), TILEMEGA_DNN4_REPO=str(ROOT))
    save(queue / 'queue_f3.json', [dict(name='dnn4-focused-f3-gates', gpu=True, timing=False,
         timeout_s=3600, needs_free_mib=4096, command=command, cwd=str(ROOT), env=environment)])
    scheduler = [str(ROOT / 'scripts/dnn4_guard_python.py'), '-m', 'tilemega.bench.scheduler',
                 '--queue-dir', str(queue), '--out', str(out / 'scheduler'),
                 '--policy', str(ROOT / 'configs/bench/sm90r2_guard_policy.json'), '--deadline-hours', '6']
    with (out / 'dispatch.log').open('w') as stream:
        subprocess.run(scheduler, cwd=ROOT, env=dict(os.environ, **environment), stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    state = json.loads((out / 'scheduler/state.json').read_text())['dnn4-focused-f3-gates']
    print(json.dumps(state, indent=2))
    return 0 if state['status'] == 'done' else 1


if __name__ == '__main__':
    raise SystemExit(main())
