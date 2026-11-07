#!/usr/bin/env python3
"""Build and guard the focused F3 GAP/residual, gather and split-K gates.

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
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'test/unit/dnn3_f3_test.cu'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def inputs():
    paths = [SOURCE, *sorted((ROOT / 'include').rglob('*.h')),
             *sorted((ROOT / 'include').rglob('*.cuh'))]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def execute(options):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1' or not os.environ.get('TILEMEGA_RUN_NONCE'):
        raise ValueError('the cohort requires the GPU scheduler and guard')
    out = options.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    build = json.loads(options.build.read_text())
    binary = Path(build['binary'])
    if sha(binary) != build['binary_sha256'] or inputs() != build['inputs']:
        raise ValueError('unit binary or source differs from its build receipt')
    result = dict(evidence='verified', build=str(options.build), passed=False,
                  scope='focused F3 GAP/residual, gather and split-K regressions; not complete P-UNIT',
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
        code, text = run(f'unit-{index+1}', [str(binary), 'all'])
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
    for case in ('gather', 'pool', 'split'):
        for tool, count in (('memcheck', 1), ('initcheck', 1), ('racecheck', 5), ('synccheck', 5)):
            for index in range(count):
                name = f'{case}-{tool}-{index+1}'
                raw = out / (name + '.sanitizer.log')
                command = ['compute-sanitizer', '--tool', tool, '--error-exitcode', '86',
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
    result.update(passed=True, finished=time.time())
    save(out / 'result.json', result)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args()
    if options.execute:
        return execute(options)
    out = options.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = out / 'dnn3_f3_test'
    log = out / 'ptxas.log'
    before = inputs()
    command = ['taskset', '-c', '136-143', 'nvcc', '-rdc=true', '-std=c++17', '-O3',
               '-gencode', 'arch=compute_90a,code=sm_90a', '--expt-relaxed-constexpr',
               '-Xptxas', '-v', '-Xptxas', '--device-function-maxrregcount=224',
               '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'third_party/cutlass/include'),
               str(SOURCE), '-o', str(binary)]
    with log.open('w') as stream:
        done = subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                              stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
    text = log.read_text()
    spills = [int(x) for x in re.findall(r'(\d+) bytes spill (?:stores|loads)', text)]
    registers = [int(x) for x in re.findall(r'Used (\d+) registers', text)]
    if done.returncode or before != inputs() or not spills or any(spills) or not registers or max(registers) > 224:
        raise RuntimeError('F3 unit build failed DT-SPILL/source stability; retain ' + str(log))
    build = out / 'build.json'
    save(build, dict(binary=str(binary), binary_sha256=sha(binary), inputs=before,
                     command=command, log=str(log), log_sha256=sha(log), registers=registers,
                     spill_bytes=max(spills), cutlass_commit=subprocess.check_output(
                         ['git', '-C', str(ROOT / 'third_party/cutlass'), 'rev-parse', 'HEAD'], text=True).strip()))
    queue = out / 'queue'
    queue.mkdir()
    gpu = str(options.device)
    command = ['taskset', '-c', '128-135', sys.executable, str(Path(__file__).resolve()),
               '--execute', '--build', str(build), '--out', '{attempt_dir}/cohort']
    environment = dict(CUDA_VISIBLE_DEVICES=gpu, TILEMEGA_DEVICE_INDEX=gpu,
                       TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', TILEMEGA_NUMA_NODE='0',
                       TILEMEGA_GPU_PID_MODE='namespace_limited', TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
                       PYTHONPATH=str(ROOT / 'python'))
    save(queue / 'queue_f3.json', [dict(name='dnn3-focused-f3-gates', gpu=True, timing=False,
         timeout_s=3600, needs_free_mib=4096, command=command, cwd=str(ROOT), env=environment)])
    scheduler = [str(ROOT / 'scripts/dnn1_idle_floor_python.py'), '-m', 'tilemega.bench.scheduler',
                 '--queue-dir', str(queue), '--out', str(out / 'scheduler'),
                 '--policy', str(ROOT / 'configs/bench/sm90r2_guard_policy.json'), '--deadline-hours', '6']
    with (out / 'dispatch.log').open('w') as stream:
        subprocess.run(scheduler, cwd=ROOT, env=dict(os.environ, **environment), stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    state = json.loads((out / 'scheduler/state.json').read_text())['dnn3-focused-f3-gates']
    print(json.dumps(state, indent=2))
    return 0 if state['status'] == 'done' else 1


if __name__ == '__main__':
    raise SystemExit(main())
