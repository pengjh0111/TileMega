#!/usr/bin/env python3
"""Build NEW DNN forward artifacts on the explicit bounded DNN CPU lane."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

LOCK = '/tmp/tilemega_dnn1_cpu_build.lock'
AFFINITY = {88, 89, 90, 91}
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def unchanged(hashes):
    return all(Path(p).is_file() and sha(p) == value for p, value in hashes.items())


def reserve(output):
    if output.suffix != '.so': raise ValueError('output must be a new .so artifact')
    output.parent.mkdir(parents=True, exist_ok=True)
    if list(output.parent.glob(output.name + '*')):
        raise ValueError('retain artifact evidence; choose a new output prefix')
    path = Path(str(output) + '.cpu-build.json')
    with path.open('x') as f: f.write('{}\n')
    return path


def nvcc_command(nvcc, root, output, manifest):
    if manifest.get('phase') != 'forward' or 'native_forward' not in manifest:
        raise ValueError('DNN CPU lane requires a native forward manifest')
    registers = manifest['register_contract']
    limit = registers['callee_limit']
    if registers['threads'] != 160 or type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError('invalid SM90 native forward register contract')
    # cuda_occupancy.h uses a 256-register allocation ceiling for SM90; the
    # manifest supplies the computed
    # occupancy-constrained ceiling, matching native_forward.cpp.
    args = [str(nvcc), '-rdc=true', '-std=c++17', '-O3', '-gencode',
            'arch=compute_90a,code=sm_90a', '--expt-relaxed-constexpr', '-shared',
            '-Xcompiler', '-fPIC', '-cudart', 'shared', '-Xptxas', '-v',
            '--split-compile', '1', '-I' + str(root / 'include'),
            '-I' + str(root / 'third_party/cutlass/include'),
            str(output) + '.cu', '-o', str(output)]
    if limit < 256: args += ['-Xptxas', '--device-function-maxrregcount=' + str(limit)]
    return args


def bounded(command, receipt, log, env, root):
    helper = ROOT / 'scripts/run_sm90r1_build.py'
    args = [sys.executable, str(helper), '--receipt', str(receipt), '--max-gib', '48',
            '--timeout', '5400', '--lock', LOCK, '--', *command]
    with Path(log).open('w') as stream:
        code = subprocess.run(args, env=env, cwd=root, stdout=stream,
                              stderr=subprocess.STDOUT, check=False).returncode
    record = json.loads(Path(receipt).read_text())
    return code, record


def bounded_success(code, receipt, command):
    return (code == 0 and receipt.get('status') == 'completed' and
            receipt.get('exit_code') == 0 and receipt.get('command') == command and
            receipt.get('lock') == LOCK and receipt.get('max_address_space_gib') == 48 and
            receipt.get('timeout_s') == 5400 and
            receipt.get('classification') == 'compilation; not a numerical test')


def identity_call(root, args, env):
    subprocess.run([sys.executable, '-m', 'tilemega.build.identity', '--root', str(root),
                    *args], cwd=root, env=env, check=True)


def finish_identity(output, root, snapshot, env, allowed):
    if not allowed: raise RuntimeError('compiler or input validation failed; no artifact identity generated')
    identity_call(root, ['--so', str(output), '--snapshot', str(snapshot)], env)
    subprocess.run([sys.executable, '-c',
                    'import sys; from tilemega.build.identity import verify; verify(sys.argv[1])',
                    str(output)], cwd=root, env=env, check=True)


def compile_artifact(a):
    output = a.output.resolve(); root = a.source_root.resolve()
    compiler = a.compiler.resolve(); nvcc = a.nvcc.resolve(); reject = a.reject_cuda.resolve()
    inputs = [a.input.resolve(), a.coordinates.resolve(), a.target.resolve()]
    for p in [compiler, nvcc, reject, *inputs]:
        if not p.is_file(): raise ValueError('missing input: ' + str(p))
    receipt_path = reserve(output)
    record = dict(schema=1, classifier='dnn1_bounded_cpu_lane_v1', status='preparing',
                  gpu_executed=False, lock=LOCK, cpu_affinity=sorted(AFFINITY),
                  max_address_space_gib=48, timeout_s=5400, started=time.time(),
                  source_root=str(root), compiler=str(compiler), output=str(output),
                  scheduling='new DNN CPU mutex; existing global queue unchanged')
    write(receipt_path, record)
    snapshot = Path(str(output) + '.source.json')
    hashes = {str(p): sha(p) for p in [compiler, nvcc, reject, *inputs,
                Path(__file__).resolve(), ROOT / 'scripts/run_sm90r1_build.py']}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', LD_PRELOAD=str(reject),
               TILEMEGA_FORWARD_CONSTANTS_FILE=str(inputs[2]), PYTHONPATH=str(root / 'python'))
    for key in ('TILEMEGA_DEVICE_INDEX', 'TILEMEGA_GPU_LOCK_HELD', 'TILEMEGA_DEVICE_GROUP'):
        env.pop(key, None)
    try:
        archives = {}
        for suffix, original in [('cpu-build-driver.py', Path(__file__).resolve()),
                                 ('cpu-build-limits.py', ROOT / 'scripts/run_sm90r1_build.py')]:
            archive = Path(str(output) + '.' + suffix)
            archive.write_bytes(original.read_bytes())
            hashes[str(archive)] = sha(archive)
            if hashes[str(archive)] != hashes[str(original)]:
                raise RuntimeError('build driver changed during archival')
            archives[suffix] = dict(path=str(archive), sha256=hashes[str(archive)])
        record['driver_archives'] = archives
        write(receipt_path, record)
        os.sched_setaffinity(0, AFFINITY)
        if os.sched_getaffinity(0) != AFFINITY: raise RuntimeError('CPU affinity not applied')
        identity_call(root, ['--capture', '--compiler', str(compiler), '--snapshot', str(snapshot)], env)
        source = json.loads(snapshot.read_text())
        for p, value in source['source_files'].items():
            path = Path(p); path = path if path.is_absolute() else root / path
            key = str(path)
            if key in hashes and hashes[key] != value: raise RuntimeError('conflicting input binding')
            hashes[key] = value
        hashes[str(snapshot)] = sha(snapshot)
        hashes[str(snapshot) + '.patch'] = sha(str(snapshot) + '.patch')
        if not unchanged(hashes): raise RuntimeError('inputs changed before source generation')
        cu = Path(str(output) + '.cu'); cg = Path(str(output) + '.tmexec.mlir')
        emit = [str(compiler), 'compile', str(inputs[0]), str(cu), '--tmexec-config',
                str(inputs[1]), '--runtime-target', str(inputs[2]), '--serving', 'forward',
                '--emit', 'serving', '--frontend', 'dnn', '--batch', str(a.batch),
                '--dump-cg', str(cg)]
        code, limits = bounded(emit, str(output) + '.emit_limits.json',
                               str(output) + '.emit.log', env, root)
        record['emit_command'] = emit
        if not bounded_success(code, limits, emit) or not unchanged(hashes):
            raise RuntimeError('bounded source generation failed or inputs changed')
        emitted_manifest = Path(str(cu) + '.plan.json')
        manifest = json.loads(emitted_manifest.read_text())
        if (manifest['export_sha256'] != hashes[str(inputs[0])] or
            manifest['coordinates_sha256'] != hashes[str(inputs[1])] or
            manifest['plan_sha256'] != sha(cg) or manifest['plan_module'] != str(cg) or
            manifest['runtime_batch'] != a.batch or manifest['runtime_target'] != str(inputs[2])):
            raise RuntimeError('emitted manifest does not bind requested inputs/plan')
        plan = Path(str(output) + '.plan.json'); plan.write_bytes(emitted_manifest.read_bytes())
        command = nvcc_command(nvcc, root, output, manifest)
        command_path = Path(str(output) + '.build_command.txt')
        command_path.write_text(shlex.join(command) + '\n')
        for p in (cu, cg, plan, emitted_manifest, command_path): hashes[str(p)] = sha(p)
        if not unchanged(hashes): raise RuntimeError('inputs changed before nvcc')
        record.update(status='compiling', command=command, input_hashes=hashes)
        write(receipt_path, record)
        log = Path(str(output) + '.ptxas.log'); bounded_path = Path(str(output) + '.build_limits.json')
        code, limits = bounded(command, bounded_path, log, env, root)
        stable = unchanged(hashes)
        success = bounded_success(code, limits, command) and stable and output.is_file()
        result = dict(schema=1, classifier='dnn1_bounded_cpu_lane_v1',
                      status='compiled' if success else 'compiler_error',
                      classification='compilation; not a numerical test', inputs_unchanged=stable,
                      input_hashes=hashes, source_digest=source['source_digest'],
                      bounded_receipt=str(bounded_path), bounded_receipt_sha256=sha(bounded_path),
                      bounded_status=limits.get('status'), bounded_exit_code=limits.get('exit_code'),
                      log=str(log), log_sha256=sha(log), raw_command_sha256=sha(command_path),
                      generated_source_sha256=sha(cu), source_snapshot_sha256=sha(snapshot),
                      export_sha256=hashes[str(inputs[0])], coordinates_sha256=hashes[str(inputs[1])],
                      target_sha256=hashes[str(inputs[2])], output_present=output.is_file(),
                      artifact_identity_generated_before_classification=False, diagnostics=[])
        write(str(output) + '.compile-result.json', result)
        finish_identity(output, root, snapshot, env, success)
        if not unchanged(hashes): raise RuntimeError('inputs changed during identity generation')
        write(str(output) + '.serving.json', dict(forward_mode='L1', forward_loop=1,
              source='native static forward program; bounded DNN CPU lane'))
        record.update(status='compiled', inputs_unchanged=True,
                      identity_sha256=sha(str(output) + '.identity.json'))
    except BaseException as error:
        record.update(status='compiler_error', error=str(error),
                      inputs_unchanged=unchanged(hashes))
        # A later identity error must not leave a loadable certified artifact.
        identity = Path(str(output) + '.identity.json')
        if identity.exists(): identity.rename(str(output) + '.identity.failed.json')
        result_path = Path(str(output) + '.compile-result.json')
        result = json.loads(result_path.read_text()) if result_path.exists() else dict(
            schema=1, classifier='dnn1_bounded_cpu_lane_v1', input_hashes=hashes)
        result.update(status='compiler_error', error=str(error), inputs_unchanged=record['inputs_unchanged'])
        write(result_path, result)
        raise
    finally:
        record['ended'] = time.time(); write(receipt_path, record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('compiler', 'source-root', 'input', 'coordinates', 'target', 'output', 'reject-cuda'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--nvcc', type=Path, default=Path('/usr/local/cuda/bin/nvcc'))
    a = p.parse_args()
    if a.batch not in (1, 2, 4, 8, 16): p.error('DNN batch must be 1, 2, 4, 8, or 16')
    compile_artifact(a)


if __name__ == '__main__': main()
