#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Build, prepare guarded queues, and collect DNN1 production-primitive constants.

`build` and `queue` never initialize CUDA. `run` requires the scheduler's GPU
and NUMA locks. `collect` never modifies sm90_native_constants; --target adds
only the new forward section after all required measurements are complete.
Unimplemented FX variants remain pending rather than receiving synthetic 0s.
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/dnn1_constants.cu'
POLICY = ROOT / 'configs/bench/sm90r2_guard_policy.json'
PYTHON = str(ROOT / 'scripts/dnn1_supervised_python.py')
PCI = {0: '0000:0f:00.0', 5: '0000:87:00.0', 7: '0000:b7:00.0'}
FORWARD_HEADERS = (
    'Backend/Sm90ForwardEpilogue.cuh', 'Backend/Sm90ForwardGemm.cuh',
    'Backend/Sm90ForwardDepthwise.cuh', 'Backend/Sm90ForwardEncoderAttention.cuh',
    'Backend/Sm90ForwardSmall.cuh', 'Backend/Sm90ForwardMark.cuh',
)
OTHER_HEADERS = (
    'Backend/Sm90Gmma.h', 'Backend/Sm90DenseRing.cuh', 'Backend/Sm90DenseRoles.cuh',
    'Backend/Sm90DenseAttentionMath.cuh', 'Backend/Sm90DenseLayout.h',
    'Codegen/executor/Async.cuh', 'Codegen/executor/TensorMapTiled.h',
    'Codegen/DmDescriptors.h',
)
METHOD = {
    'protocol': 'five fresh executable processes per case and GPU; GPUs 0/5/7; process-median median',
    'timer': 'clock64 calibrated in every fresh process against a 20ms globaltimer interval',
    'observer': 'no overhead subtraction or fitting; segment spans include two timer reads and A1 register bookkeeping',
    'resolution': 'one clock cycle; measured globaltimer calibration step recorded per process',
    'segments': 'only within-CTA durations; maximum CTA mean per production call; no cross-SM absolute clocks',
    'direct': 'actual Apply/Finish or uninstrumented Small::Run; input reset/output drainage outside timed interval',
    'rowstat': 'real Finish width=0 excludes local per-column statistics and output store; max CTA interval',
    'small_denominator': {'layout': 'live input elements', 'pool': 'output elements',
        'global_pool': 'input partial elements', 'gemv': 'weight elements',
        'layernorm': 'output elements', 'embedding': 'output elements'},
    'versioning': 'all forward bodies snapshotted and hashed at build; FX paths require separate new build receipt',
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def command_to_log(command, log, *, env=None, timeout_s=3600):
    """Terminate only this command's own process group on timeout."""
    with Path(log).open('w') as output:
        p = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                             cwd=ROOT, env=env, start_new_session=True)
        try:
            code = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise
    return code


def build(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    source_root = (args.source_root or ROOT).resolve()
    fixed = args.implementation == 'fixed'
    extra = ('Backend/Sm90ForwardGemv.cuh', 'Backend/Sm90ForwardResources.h') if fixed else ()
    headers, inputs = {}, {}
    for rel in (*FORWARD_HEADERS, *extra, *OTHER_HEADERS):
        source = source_root / 'include/tilemega' / rel
        dest = out / 'include/tilemega' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        headers[str(source.relative_to(source_root))] = sha(source)
        inputs[str(dest)] = sha(dest)
    source = out / SOURCE.name
    shutil.copy2(SOURCE, source)
    mark = out / 'include/tilemega/Backend/Sm90ForwardMark.cuh'
    original = mark.read_text()
    if original.count('%%globaltimer') != 1 or 'Sm90ForwardTraceNow()' not in original:
        raise ValueError('Mark overlay needs review: expected one production globaltimer read')
    overlay = out / 'overlay/tilemega/Backend/Sm90ForwardMark.cuh'
    overlay.parent.mkdir(parents=True)
    overlay.write_text('// DNN1 probe overlay: all trace span fields are CLOCK CYCLES, not ns.\n' +
                       original.replace('%%globaltimer', '%%clock64'))
    inputs[str(overlay)] = sha(overlay)
    inputs[str(source)] = sha(source)
    receipt = dict(schema=1, status='building', created_unix=time.time(),
        source_root=str(source_root), implementation=args.implementation,
        source=str(source), source_sha256=sha(source), collector_sha256=sha(__file__),
        original_sources=headers, snapshot_sources=inputs,
        clock_overlay=dict(original_sha256=sha(mark), overlay_sha256=sha(overlay),
            only_instruction_replacement='%globaltimer -> %clock64', time_domain='SM clock cycles'),
        method=METHOD, gpu_executed=False, commands=[], binaries={})
    receipt['variant'] = digest(dict(source=sha(source), headers=headers, overlay=sha(overlay),
                                    implementation=args.implementation))
    target = read(ROOT / 'configs/targets/sm_90_h800.json')
    resources = target['resources']
    shared_limit = resources['max_dynamic_smem_per_cta']
    receipt['resource_target'] = dict(path='configs/targets/sm_90_h800.json',
        resources=resources, shared_limit=shared_limit)
    receipt['nvcc_version'] = subprocess.check_output([args.nvcc, '--version'], text=True, env=env)
    write(out / 'build.json', receipt)
    for kind, flag in (('direct', 0), ('segments', 1)):
        binary = out / ('dnn1_constants_' + kind)
        command = [args.nvcc, '-std=c++17', '-O3', '-gencode=arch=compute_90a,code=sm_90a',
            '-lineinfo', '--expt-relaxed-constexpr', f'-DDNN1_SHARED_LIMIT={shared_limit}',
            '-I' + str(out / 'overlay' if flag else out / 'include'),
            '-I' + str(out / 'include'), '-I' + str(source_root / 'include'),
            '-I' + str(source_root / 'third_party/cutlass/include'),
            '-I' + str(source_root / 'third_party/cutlass/tools/util/include'),
            f'-DDNN1_FIXED_FORWARD={int(fixed)}',
            f'-DDNN1_SEGMENT_PROBES={flag}', str(source), '-o', str(binary)]
        log = out / (kind + '_build.log')
        receipt['commands'].append(dict(command=command, shell=shlex.join(command), log=str(log)))
        code = command_to_log(command, log, env=env, timeout_s=args.timeout_s)
        receipt['commands'][-1]['returncode'] = code
        write(out / 'build.json', receipt)
        if code:
            receipt['status'] = 'compile_failed'
            write(out / 'build.json', receipt)
            raise RuntimeError(f'{kind} nvcc failed; original output: {log}')
        receipt['binaries'][kind] = dict(path=str(binary), sha256=sha(binary))
    catalog = []
    for item in receipt['binaries'].values():
        catalog += [json.loads(line) for line in subprocess.check_output(
            [item['path'], '--list'], text=True, env=env).splitlines()]
    if len({x['id'] for x in catalog}) != len(catalog):
        raise ValueError('duplicate case identity')
    reserved = max(0, resources['max_smem_per_sm'] - shared_limit)
    for item in catalog:
        item['resources'] = {}
        for residency in (1, 2):
            budget = max(0, min(shared_limit, resources['max_smem_per_sm']//residency - reserved))
            arena = (budget - 1024)//2048*1024
            item['resources'][str(residency)] = dict(budget=budget, arena=arena,
                fits_arena=item['shared_bytes'] <= arena,
                fits_exclusive_budget=item['shared_bytes'] <= budget)
    write(out / 'catalog.json', catalog)
    receipt.update(status='built_not_gpu_validated', catalog=str(out / 'catalog.json'),
                   catalog_sha256=sha(out / 'catalog.json'), completed_unix=time.time())
    write(out / 'build.json', receipt)
    print(json.dumps(dict(build=str(out / 'build.json'), variant=receipt['variant'],
        ready=sum(x['status'] == 'ready' for x in catalog),
        pending=sum(x['status'] == 'pending' for x in catalog), gpu_executed=False)))


def verify_build(path):
    build = read(path)
    if build['status'] != 'built_not_gpu_validated':
        raise ValueError('successful CPU build receipt required')
    for file, expected in build['snapshot_sources'].items():
        if sha(file) != expected:
            raise ValueError('build snapshot changed: ' + file)
    for item in build['binaries'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('binary changed: ' + item['path'])
    if sha(build['catalog']) != build['catalog_sha256']:
        raise ValueError('case catalog changed')
    return build, read(build['catalog'])


def gpu_env(gpu):
    return dict(CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_DEVICE_INDEX=str(gpu),
        TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock',
        TILEMEGA_NUMA_NODE=str(0 if gpu < 4 else 1),
        TILEMEGA_GPU_PID_MODE='namespace_limited',
        PYTHONPATH=str(ROOT / 'python'), OMP_NUM_THREADS='8')


def queue(args):
    receipt, catalog = verify_build(args.build)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cards = [int(x) for x in args.gpus.split(',')]
    if len(set(cards)) != len(cards) or any(x not in PCI for x in cards):
        raise ValueError('distinct GPU0/5/7 only')
    sys.path.insert(0, str(ROOT / 'python'))
    from tilemega.bench.gpu_guard import require_allowed_gpu
    policy = read(POLICY)
    selected = [x for x in catalog if x['status'] == 'ready' and re.search(args.case_pattern, x['id'])]
    if not selected:
        raise ValueError('case pattern selects no ready primitives')
    affinity = json.loads(args.cpu_affinity) if args.cpu_affinity else {}
    for gpu in cards:
        if affinity and (str(gpu) not in affinity or not affinity[str(gpu)]):
            raise ValueError('explicit CPU affinity required for every parallel card')
    spec = dict(schema=1, build=str(args.build.resolve()), build_sha256=sha(args.build),
        variant=receipt['variant'], gpus=cards, rounds=5, samples=args.samples,
        cases=selected, pending=[x for x in catalog if x['status'] == 'pending'],
        resource_infeasible=[x for x in catalog if x['status'] == 'infeasible'],
        omitted_ready=[x['id'] for x in catalog if x['status'] == 'ready' and x not in selected],
        collector_sha256=sha(__file__), source_recipe=str(Path(__file__).resolve()),
        policy=str(POLICY), policy_sha256=sha(POLICY), method=METHOD,
        uuid_by_gpu=policy['idle_power_mapping_provenance']['uuid_by_gpu'],
        cpu_affinity=affinity, parallel=bool(affinity),
        batch_processes=args.batch_processes,
        process_scope='one fresh process per binary per round; cases share CUDA context' if args.batch_processes else 'one fresh process per case per round',
        guard_python=dict(path=str(Path(args.python).resolve()), sha256=sha(args.python),
            idle_memory_floor_mib=1 if Path(args.python).name=='dnn1_idle_floor_python.py' else 0),
        measurement_status='not_started', created_unix=time.time())
    runner = Path(__file__).resolve()
    if getattr(args, 'strict_processes', False):
        runner = ROOT / 'scripts/dnn1_strict_constants.py'
        spec.update(strict_processes=True, strict_entry=str(runner), strict_entry_sha256=sha(runner))
    write(out / 'queue-spec.json', spec)
    # The scheduler binds its process to one physical GPU lock. Give each
    # card a separate scheduler and run the three launchers serially.
    launchers = []
    for gpu in cards:
        jobs, previous = [], []
        require_allowed_gpu(policy, str(gpu), timing=True)
        node = 0 if gpu < 4 else 1
        card = out / f'gpu{gpu}'
        env = gpu_env(gpu)
        binding = f'--cpunodebind={node}'
        if affinity:
            env['TILEMEGA_NUMA_CONCURRENCY_TRIAL'] = '1'
            binding = '--physcpubind=' + ','.join(map(str, affinity[str(gpu)]))
        for round_ in range(5):
            name = f'dnn1-constants-g{gpu}-r{round_}'
            command = ['numactl', binding, args.python,
                str(ROOT / 'scripts/run_r16_with_device_identity.py'), '--guard-policy', str(POLICY),
                '--out', '{attempt_dir}/job', '--expected-uuid',
                policy['idle_power_mapping_provenance']['uuid_by_gpu'][str(gpu)],
                '--expected-pci', PCI[gpu], '--numa-node', str(node), '--allow-no-measurements',
                '--', args.python, str(runner), 'run', '--spec', str(out / 'queue-spec.json'),
                '--gpu', str(gpu), '--round', str(round_), '--out', '{attempt_dir}/job']
            jobs.append(dict(name=name, gpu=True, timing=True, priority=10,
                timeout_s=args.session_timeout_s, needs_free_mib=4096, cwd=str(ROOT),
                after=previous, env=env, command=command))
            previous = [name]
        write(card / 'queue/queue_dnn1_constants.json', jobs)
        command = ['env', '-u', 'TILEMEGA_DEVICE_GROUP', '-u', 'TILEMEGA_GPU_LOCK_HELD',
            '-u', 'TILEMEGA_RUN_NONCE', '-u', 'TILEMEGA_NUMA_CONCURRENCY_VERIFIED',
            '-u', 'TILEMEGA_NUMA_CONCURRENCY_TRIAL',
            *[f'{k}={v}' for k, v in env.items()], args.python,
            '-m', 'tilemega.bench.scheduler', '--queue-dir', str(card / 'queue'),
            '--out', str(card / 'scheduler'), '--policy', str(POLICY), '--deadline-hours', '96']
        launcher = card / 'launch.sh'
        launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(ROOT)) + '\n' +
                           shlex.join(command) + '\n')
        launcher.chmod(0o755)
        launchers.append(str(launcher))
    serial = out / 'launch_serial.sh'
    serial.write_text('#!/usr/bin/env bash\nset -euo pipefail\n' +
                      ''.join(shlex.join(['bash', path]) + '\n' for path in launchers))
    serial.chmod(0o755)
    launcher = out / 'launch.sh'
    launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(ROOT)) + '\n' +
        shlex.join(['flock', '/tmp/tilemega_dnn1_constants_scheduler.lock', 'bash', str(serial)]) + '\n')
    launcher.chmod(0o755)
    print(json.dumps(dict(queue_root=str(out), sessions=5*len(cards), cases=len(selected),
                         launcher=str(launcher), gpu_executed=False)))


def held(path):
    with open(path, 'a') as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(file, fcntl.LOCK_UN)
    return False


def require_session(gpu, spec=None):
    e = os.environ
    node = 0 if gpu < 4 else 1
    if (e.get('TILEMEGA_GPU_LOCK_HELD') != '1' or e.get('TILEMEGA_TIMING') != '1' or
            e.get('TILEMEGA_DEVICE_INDEX') != str(gpu) or e.get('CUDA_VISIBLE_DEVICES') != str(gpu) or
            e.get('TILEMEGA_DEVICE_GROUP') is not None or
            e.get('TILEMEGA_GPU_LOCK') != f'/tmp/tilemega_gpu{gpu}.lock' or
            not held(f'/tmp/tilemega_gpu{gpu}.lock')):
        raise ValueError('scheduler single-GPU timing guard and actual GPU/NUMA flocks required')
    if spec and spec.get('parallel'):
        affinity = set(spec['cpu_affinity'][str(gpu)])
        if (e.get('TILEMEGA_NUMA_CONCURRENCY_TRIAL') != '1' or
                e.get('TILEMEGA_NUMA_CONCURRENCY_VERIFIED') == '1' or
                not affinity or not set(os.sched_getaffinity(0)) <= affinity):
            raise ValueError('parallel constants need declared affinity and explicit NUMA trial')
    elif not held(f'/tmp/tilemega_numa{node}.lock'):
        raise ValueError('serial constants require actual NUMA flock')


def run(args):
    spec = read(args.spec)
    require_session(args.gpu, spec)
    if sha(__file__) != spec['collector_sha256']:
        raise ValueError('collector or queue recipe changed')
    if args.gpu not in spec['gpus'] or args.round not in range(5):
        raise ValueError('invalid physical GPU or round')
    if sha(spec['build']) != spec['build_sha256'] or sha(POLICY) != spec['policy_sha256']:
        raise ValueError('build receipt or guard policy changed after queue')
    if spec.get('guard_python') and sha(spec['guard_python']['path']) != spec['guard_python']['sha256']:
        raise ValueError('guard Python wrapper changed after queue')
    receipt, _ = verify_build(spec['build'])
    sys.path.insert(0, str(ROOT / 'python'))
    from dnn1_device import gpu
    args.out.mkdir(parents=True, exist_ok=True)
    session = dict(schema=1, gpu=args.gpu, round=args.round, variant=receipt['variant'],
        build_sha256=spec['build_sha256'], pid=os.getpid(), cases=[], started_unix=time.time(),
        method=METHOD, source_spec=str(args.spec.resolve()))
    write(args.out / 'constants_session.json', session)
    batches = {}
    if spec.get('batch_processes'):
        session['process_scope'] = spec['process_scope']
        for kind in sorted({item['binary'] for item in spec['cases']}):
            items = [item for item in spec['cases'] if item['binary'] == kind]
            for item in items:
                (args.out / 'cases' / item['id']).mkdir(parents=True, exist_ok=False)
            binary = receipt['binaries'][kind]
            command = [binary['path'], '--batch-cases', ','.join(item['id'] for item in items),
                '--batch-root', str(args.out / 'cases'), '--samples', str(spec['samples'])]
            before = gpu(args.gpu)
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            after = gpu(args.gpu)
            (args.out / f'{kind}_batch_stdout.log').write_text(stdout)
            (args.out / f'{kind}_batch_stderr.log').write_text(stderr)
            batches[kind] = dict(command=command, pid=process.pid, returncode=process.returncode,
                before=before, after=after, clock_metadata_scope='binary process boundaries; each case has independent clock64/globaltimer calibration')
            write(args.out / 'batch_processes.json', batches)
            if process.returncode:
                raise RuntimeError(f'{kind} batch probe failed: {stderr}')
    for item in spec['cases']:
        path = args.out / 'cases' / item['id']
        binary = receipt['binaries'][item['binary']]
        if batches:
            group = batches[item['binary']]
            before,after,command = group['before'],group['after'],group['command']
            returncode,pid = group['returncode'],group['pid']
            stdout,stderr = (path/'stdout.csv').read_text(),(path/'stderr.log').read_text()
        else:
            path.mkdir(parents=True, exist_ok=False)
            before = gpu(args.gpu)
            command = [binary['path'], '--case', item['id'], '--samples', str(spec['samples']),
                       '--raw', str(path / 'samples.csv')]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            after = gpu(args.gpu)
            returncode,pid = process.returncode,process.pid
            (path / 'stdout.csv').write_text(stdout)
            (path / 'stderr.log').write_text(stderr)
        rows = []
        timing = None
        if returncode == 0:
            rows = list(csv.DictReader(io.StringIO(stdout)))
            if not rows or any(not math.isfinite(float(r['median_ns'])) or float(r['median_ns']) <= 0 for r in rows):
                raise ValueError('no positive finite primitive measurements')
            calibration = re.search(r'calibration cycles=(\d+) global_ns=(\d+) cycles_per_ns=(\S+) '
                r'clock_resolution_ns=(\S+) observed_globaltimer_step_ns=(\d+)', stderr)
            observer = re.search(r'observer_empty_ComputeSync_median_ns=(\S+)', stderr)
            if not calibration or not observer:
                raise ValueError('missing per-process clock calibration/observer evidence')
            timing = dict(zip(('calibration_cycles', 'calibration_global_ns', 'cycles_per_ns',
                'clock_resolution_ns', 'observed_globaltimer_step_ns'), map(float, calibration.groups())))
            timing.update(observer_empty_ComputeSync_median_ns=float(observer.group(1)),
                          observer_subtracted=False)
        row = dict(case=item, command=command, process_pid=pid, fresh_executable_process=True,
            process_scope=spec.get('process_scope','one fresh process per case per round'),
            clock_metadata_scope='binary process boundaries' if batches else 'case process boundaries',
            guard_python=spec.get('guard_python'),
            returncode=returncode, before=before, after=after,
            throttle_invalid=None, power_monitoring=False,
            resource_policy='processes and extra memory only', valid=returncode == 0,
            timing=timing,
            rows=rows, binary_sha256=binary['sha256'], raw=str(path / 'samples.csv'),
            stdout=str(path / 'stdout.csv'), stderr=str(path / 'stderr.log'),
            raw_sha256=sha(path / 'samples.csv') if (path / 'samples.csv').exists() else None,
            stdout_sha256=sha(path / 'stdout.csv'), stderr_sha256=sha(path / 'stderr.log'))
        write(path / 'result.json', row)
        session['cases'].append(str(path / 'result.json'))
        write(args.out / 'constants_session.json', session)
        if returncode:
            raise RuntimeError('primitive failed; original log: ' + str(path / 'stderr.log'))
    session.update(completed_unix=time.time(), status='complete')
    from dnn1_baselines import passive_context_drain
    session['exit_context_drain'] = passive_context_drain(dict(gpu=args.gpu))
    write(args.out / 'constants_session.json', session)


def append_forward_section(target, constants):
    """Append a single root key, leaving every existing byte untouched."""
    target = Path(target)
    original = target.read_text()
    value = json.loads(original)
    if 'sm90_forward_constants' in value:
        raise ValueError('refuse to replace existing measured forward constants; use a new target or explicit review')
    native_before = digest(value.get('sm90_native_constants'))
    close = len(original.rstrip()) - 1
    if close < 0 or original[close] != '}':
        raise ValueError('target must be a JSON object')
    # Compact only the new section to respect the committed-data size limit.
    encoded = json.dumps(constants, sort_keys=True, separators=(',', ':'))
    addition = '  "sm90_forward_constants": ' + encoded.replace('\n', '\n  ')
    prefix = original[:close].rstrip()
    updated = prefix + (',' if value else '') + '\n' + addition + '\n' + original[close:]
    parsed = json.loads(updated)
    if digest(parsed.get('sm90_native_constants')) != native_before:
        raise AssertionError('native constants mutation forbidden')
    for key in value:
        if parsed[key] != value[key]:
            raise AssertionError('existing target key changed: ' + key)
    # Only the whitespace immediately before the final brace may differ.
    if not updated.startswith(prefix):
        raise AssertionError('existing target prefix changed')
    target.write_text(updated)
    return dict(native_sha256=native_before, before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                after_sha256=sha(target))


def collect(args):
    root = args.run_root.resolve()
    spec = read(root / 'queue-spec.json')
    receipt, _ = verify_build(spec['build'])
    if set(spec['gpus']) != {0, 5, 7}:
        raise ValueError('final constants require all GPU0/5/7; diagnostic subset queues cannot freeze')
    groups, evidence = {}, []
    # Only scheduler-accepted attempts may contribute; no log scraping fallback.
    for round_ in range(5):
        for gpu in spec['gpus']:
            state = read(root / f'gpu{gpu}/scheduler/state.json')
            states = state.get('steps', state)
            if isinstance(states, list):
                states = {entry['name']: entry for entry in states}
            name = f'dnn1-constants-g{gpu}-r{round_}'
            entry = states.get(name)
            if not entry or entry.get('status') != 'done':
                raise ValueError('missing accepted scheduler session: ' + name)
            # Read precisely the scheduler's accepted attempt, retaining any
            # rejected/preflight attempts in place without counting them.
            attempt = Path(entry['out'])
            guard = attempt / 'guard_result.json'
            accepted = attempt / 'job/constants_session.json'
            if not guard.is_file() or read(guard).get('code') != 0:
                raise ValueError('missing guard acceptance: ' + name)
            if spec.get('strict_processes'):
                strict = read(accepted.parent / 'strict_acceptance.json')
                if (strict.get('status') != 'sampled_boundaries_and_windows_pass' or
                        strict['strict_entry_sha256'] != spec['strict_entry_sha256'] or
                        sha(spec['strict_entry']) != spec['strict_entry_sha256'] or
                        sha(accepted.parent/'strict_boundaries.json') != strict['boundaries_sha256'] or
                        sha(accepted.parent/'strict_windows.json') != strict['windows_sha256']):
                    raise ValueError('missing or changed strict process acceptance: ' + name)
            identity_file = accepted.parent / 'physical-device-identity.json'
            identity = read(identity_file)
            if (identity['gpu_index'] != gpu or identity['nvml_uuid'] != spec['uuid_by_gpu'][str(gpu)] or
                    identity['guard_policy_sha256'] != spec['policy_sha256'] or
                    identity['numa_node'] != (0 if gpu < 4 else 1)):
                raise ValueError('physical GPU/NUMA provenance mismatch: ' + name)
            session = read(accepted)
            if (session.get('status') != 'complete' or session['variant'] != receipt['variant'] or
                    session['gpu'] != gpu or session['round'] != round_):
                raise ValueError('session identity/status mismatch: ' + str(accepted))
            if len(session['cases']) != len(spec['cases']):
                raise ValueError('session did not execute every selected case exactly once')
            case_by_id = {case['id']: case for case in spec['cases']}
            seen = set()
            for path in session['cases']:
                result = read(path)
                case_id = result['case']['id']
                if (case_id in seen or result['case'] != case_by_id.get(case_id) or
                        result['binary_sha256'] != receipt['binaries'][result['case']['binary']]['sha256']):
                    raise ValueError('case/binary identity mismatch: ' + path)
                seen.add(case_id)
                if not result['valid'] or not result['fresh_executable_process']:
                    raise ValueError('invalid primitive process: ' + path)
                for key in ('raw', 'stdout', 'stderr'):
                    if sha(result[key]) != result[key + '_sha256']:
                        raise ValueError('raw measurement changed: ' + result[key])
                for row in result['rows']:
                    if int(row['samples']) != spec['samples'] - 8:
                        raise ValueError('wrong post-warmup primitive sample count')
                    key = (gpu, row['metric'], row['key'])
                    groups.setdefault(key, []).append(dict(round=round_, median=float(row['median_ns']),
                        process_pid=result['process_pid'],
                        p99=float(row['p99_ns']), samples=int(row['samples']), unit=row['unit'],
                        method=row['method'], clock_start=result['before']['sm_clock_mhz'],
                        clock_end=result['after']['sm_clock_mhz'], result=path))
                evidence.append(dict(session=str(accepted), result=path, result_sha256=sha(path),
                                     physical_device_identity_sha256=sha(identity_file)))
    constants = dict(schema=1, variant=receipt['variant'], method=METHOD,
        process_scope=spec.get('process_scope','one fresh process per case per round'),
        guard_python=spec.get('guard_python'),
        header_sha256=receipt['original_sources'], probe_sha256=receipt['source_sha256'],
        clock_overlay=receipt['clock_overlay'], build_receipt=str(Path(spec['build']).resolve()),
        by_gpu={}, pending=spec['pending'], resource_infeasible=spec.get('resource_infeasible', []),
        unstable=[], completeness='partial_pending_FX' if spec['pending'] or spec.get('omitted_ready') else 'complete',
        omitted_ready=spec.get('omitted_ready', []),
        raw_measurement_root=str(root), raw_evidence='collection_receipt.json; per-session result.json')
    historical = ROOT / 'docs/results/SM90R3_constants_N1.csv'
    with historical.open() as stream:
        old_wgmma = [dict(row) for row in csv.DictReader(stream) if row['name'] == 't_wgmma']
    if not old_wgmma or any(int(row['processes']) != 5 for row in old_wgmma):
        raise ValueError('historical old-model WGMMA reference lacks five-process provenance')
    constants['historical_reference'] = dict(source=str(historical.relative_to(ROOT)),
        sha256=sha(historical), purpose='old-model comparison only; never overwrite native.forward',
        t_wgmma=old_wgmma)
    table = []
    for (gpu, metric, key), rows in sorted(groups.items()):
        if len(rows) != 5 or {r['round'] for r in rows} != set(range(5)):
            raise ValueError('exactly five fresh processes required for: ' + str((gpu, metric, key)))
        if len({r['process_pid'] for r in rows}) != 5:
            raise ValueError('five distinct process IDs required per constant')
        values = [r['median'] for r in rows]
        med = statistics.median(values)
        relative_range = (max(values) - min(values)) / med
        item = dict(median=med, p99=statistics.median(r['p99'] for r in rows),
            process_range=max(values)-min(values), relative_range=relative_range,
            unstable=relative_range > .10, processes=5, unit=rows[0]['unit'])
        constants['by_gpu'].setdefault(str(gpu), {}).setdefault(metric, {})[key] = item
        if item['unstable']:
            constants['unstable'].append(dict(gpu=gpu, metric=metric, key=key, relative_range=relative_range))
        table.append(dict(gpu=gpu, variant=receipt['variant'], metric=metric, key=key,
            median=med, p99=item['p99'], unit=item['unit'], processes=5,
            process_range=item['process_range'], relative_range=relative_range,
            unstable=item['unstable'], start_sm_clocks_mhz=';'.join(str(r['clock_start']) for r in rows),
            end_sm_clocks_mhz=';'.join(str(r['clock_end']) for r in rows),
            source=f'gpu{gpu}/raw/dnn1-constants-g{gpu}-r*/attempt*/job/cases/{Path(rows[0]["result"]).parent.name}/result.json',
            run_root=str(root)))
    ready = {(x['metric'], x['key']) for x in spec['cases']}
    for gpu in spec['gpus']:
        present = {(metric, key) for g, metric, key in groups if g == gpu}
        # Attention contributes segment suffixes rather than one combined row.
        for metric, key in ready:
            if metric == 'attention_ns':
                if not all((metric, key + ';segment=' + s) in present for s in ('qk', 'softmax', 'pv')):
                    raise ValueError('missing full production attention segments')
            elif (metric, key) not in present:
                raise ValueError('missing expected measurement: ' + str((gpu, metric, key)))
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / 'sm90_forward_constants.json', constants)
    write(args.out / 'collection_receipt.json', dict(status='collected', evidence=evidence,
        source_spec_sha256=sha(root / 'queue-spec.json'), target_modified=False))
    if table:
        with (args.out / 'DNN1_constants.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    if args.target:
        if spec['pending'] or spec.get('omitted_ready'):
            raise ValueError('pending FX/constants cannot be published as a complete solver table; collected evidence preserved')
        constants['completeness'] = 'complete'
        mutation = append_forward_section(args.target, constants)
        write(args.out / 'target_mutation.json', mutation)
    print(json.dumps(dict(out=str(args.out), measured_rows=len(table), pending=len(spec['pending']),
                         unstable=len(constants['unstable']), target_modified=bool(args.target))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    b = subs.add_parser('build')
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
    b.add_argument('--source-root', type=Path, help='explicit isolated forward implementation tree')
    b.add_argument('--implementation', choices=('original', 'fixed'), default='original')
    b.add_argument('--timeout-s', type=float, default=3600)
    b.set_defaults(function=build)
    q = subs.add_parser('queue')
    q.add_argument('--build', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--gpus', default='0,5,7')
    q.add_argument('--cpu-affinity', help='JSON mapping physical GPU to disjoint CPU lists; NUMA trial')
    q.add_argument('--batch-processes', action='store_true', help='one fresh process per binary per round, independently calibrated/reset cases')
    q.add_argument('--strict-processes', action='store_true', help='require empty <=1MiB boundaries and sampled stable singleton probe owners')
    q.add_argument('--python', default=PYTHON)
    q.add_argument('--case-pattern', default='.*')
    q.add_argument('--samples', type=int, default=256)
    q.add_argument('--session-timeout-s', type=float, default=21600)
    q.set_defaults(function=queue)
    r = subs.add_parser('run')
    r.add_argument('--spec', type=Path, required=True)
    r.add_argument('--gpu', type=int, choices=tuple(PCI), required=True)
    r.add_argument('--round', type=int, required=True)
    r.add_argument('--out', type=Path, required=True)
    r.set_defaults(function=run)
    c = subs.add_parser('collect')
    c.add_argument('--run-root', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--target', type=Path)
    c.set_defaults(function=collect)
    args = p.parse_args()
    try:
        args.function(args)
    except Exception:
        traceback.print_exc()
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
