#!/usr/bin/env python3
"""Build fixed coordinates and publish complete, single-HEAD DNN2 scans."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('resnet18', 'bert', 'mbv1', 'mbv2', 'nafnet')
CELLS = tuple((model, batch) for model in MODELS for batch in (1, 16))
GPUS = (0, 1, 2, 3, 4, 5, 7)
FIELDS = ['head', 'promoted_change', 'model', 'batch', 'gpu', 'median_ms', 'p90_ms',
          'process_range_ms', 'baseline_ms', 'baseline_system', 'ratio', 'correctness', 'source']


class ResourceRejection(RuntimeError):
    """Quarantine a whole cell when process or memory evidence is ambiguous."""


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def check_grid(cells):
    keys = [(c['model'], int(c['batch'])) for c in cells]
    if len(keys) != 10 or set(keys) != set(CELLS):
        raise ValueError('a scan must contain exactly the ten distinct model/B1/B16 cells')


def baseline_lookup(rows):
    selected = {}
    for key in CELLS:
        choices = [r for r in rows if (r['model'], int(r['batch'])) == key and
                   r['system'] in ('tensorrt', 'torchcompile') and
                   r['precision'] == 'bf16' and r['status'] == 'pass']
        if len(choices) != 4:
            raise ValueError('frozen baseline must contain the four accepted BF16 arms: ' + str(key))
        winner = min(choices, key=lambda r: (float(r['median_ms']), r['arm']))
        value = float(winner['median_ms'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('invalid frozen baseline latency')
        selected[f'{key[0]}_b{key[1]}'] = dict(ms=value, system=winner['arm'])
    return selected


def imports(source):
    sys.path[:0] = [str(Path(source) / 'scripts'), str(Path(source) / 'python'),
                    str(Path(source) / 'python/tilemega/serving')]


def validate_live(scan):
    if git(scan['source_root'], 'rev-parse', 'HEAD') != scan['head']:
        raise ValueError('source HEAD changed during the full scan')
    source = scan['source_snapshot']
    for name, digest in source['source_files'].items():
        if sha(Path(scan['source_root']) / name) != digest:
            raise ValueError('frozen source changed: ' + name)
    for path, digest in scan['files'].items():
        if sha(path) != digest:
            raise ValueError('frozen driver/input/coordinates/compiler changed: ' + path)


def require_freeze(scan, filename):
    freeze = read(filename)
    expected = dict(head=scan['head'], source_digest=scan['source_snapshot']['source_digest'],
                    compiler_sha256=scan['compiler_sha256'])
    if 'out' in scan:
        expected['scan_sha256'] = sha(Path(scan['out']) / 'scan.json')
    if freeze.get('frozen') is not True or any(freeze.get(k) != v for k, v in expected.items()):
        raise ValueError('GPU scan requires the parent source freeze for this HEAD/compiler')
    validate_live(scan)
    return dict(path=str(Path(filename).resolve()), sha256=sha(filename), **expected)


def prepare(args):
    source, compiler, exports = map(lambda p: p.resolve(), (args.source_root, args.compiler, args.exports))
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'runs/dnn2'):
        raise ValueError('raw scans belong under this repository runs/dnn2')
    out.mkdir(parents=True, exist_ok=False)
    head = git(source, 'rev-parse', 'HEAD')
    if subprocess.check_output(['git', '-C', str(source), 'diff', '--binary', 'HEAD', '--',
                               'include', 'lib', 'tools', 'python', 'cmake', 'CMakeLists.txt']):
        raise ValueError('commit the combined source before preparing a scoreboard scan')
    imports(source)
    from tilemega.build.identity import snapshot
    frozen = snapshot(source, out / 'source.json', compiler)
    raw = read(args.fixed_coordinates)
    cells = raw['cells'] if isinstance(raw, dict) else raw
    check_grid(cells)
    files = {str(compiler): sha(compiler), str(args.fixed_coordinates.resolve()): sha(args.fixed_coordinates)}
    for name in ('dnn2_scoreboard.py', 'dnn1_baselines.py', 'dnn1_device.py',
                 'dnn1_idle_floor_python.py', 'dnn1_resource_entry.py', 'dnn1_guard_entry.py',
                 'run_r16_with_device_identity.py'):
        path = source / 'scripts' / name
        files[str(path)] = sha(path)
    resolved = []
    for model, batch in CELLS:
        c = dict(next(c for c in cells if (c['model'], int(c['batch'])) == (model, batch)))
        c.update(batch=batch, key=f'{model}_b{batch}', export=str(exports / model))
        for name in ('coordinates', 'target', 'bridge', 'inputs', 'source_receipt'):
            path = Path(c[name]).resolve()
            digest = sha(path)
            if c.get(name + '_sha256') not in (None, digest):
                raise ValueError('coordinate inventory source hash differs: ' + name)
            c[name] = str(path)
            files[str(path)] = digest
        for name in ('manifest.json', 'exported_program.pt2', 'checkpoint/model.safetensors'):
            path = exports / model / name
            files[str(path)] = sha(path)
        if model == 'nafnet':
            cache = Path(c['reference_cache']).resolve()
            c['reference_cache'] = str(cache)
            for name in ('manifest.json', 'reference.safetensors'):
                files[str(cache / name)] = sha(cache / name)
        c.update(library=str(out / 'build' / c['key'] / 'plan.so'),
                 selection_config=str(out / 'build' / c['key'] / 'selected-config.json'))
        resolved.append(c)
    revision = git(source, 'rev-parse', args.baseline_revision)
    baseline = subprocess.check_output(['git', '-C', str(source), 'show',
                                      revision + ':docs/results/DNN1_baselines.csv'])
    (out / 'frozen_baselines.csv').write_bytes(baseline)
    files[str(out / 'frozen_baselines.csv')] = hashlib.sha256(baseline).hexdigest()
    lookup = baseline_lookup(list(csv.DictReader(io.StringIO(baseline.decode()))))
    scan = dict(schema=1, head=head, promoted_change=args.promoted_change, source_root=str(source),
        compiler=str(compiler), compiler_sha256=sha(compiler), exports=str(exports), out=str(out),
        source_snapshot=frozen, files=files, cells=resolved, baselines=lookup,
        baseline_revision=revision, baseline_csv_sha256=hashlib.sha256(baseline).hexdigest(),
        protocol=dict(warmup=50, iterations=200, correctness_processes=5, timing_processes=3),
        reject_cuda=str(args.reject_cuda.resolve()), created=time.time())
    if not Path(scan['reject_cuda']).is_file():
        raise ValueError('CPU builds require the CUDA-call rejection sentinel')
    scan['files'][scan['reject_cuda']] = sha(scan['reject_cuda'])
    write(out / 'scan.json', scan)
    print(json.dumps(dict(scan=str(out / 'scan.json'), freeze_required=dict(frozen=True,
        head=head, source_digest=frozen['source_digest'], compiler_sha256=sha(compiler),
        scan_sha256=sha(out / 'scan.json')))))


def validate_artifact(scan, cell):
    imports(scan['source_root'])
    from tilemega.build.identity import verify
    identity = verify(cell['library'], expected_source=scan['source_snapshot'])
    if (identity['source']['head'] != scan['head'] or identity['trace'] or
            identity['plan']['runtime_batch'] != cell['batch'] or
            identity['plan']['export_sha256'] != sha(cell['bridge'])):
        raise ValueError('compiled artifact HEAD/trace/batch/bridge differs from this scan')
    return identity


def build(args):
    scan = read(args.scan)
    validate_live(scan)
    records = []
    for cell in scan['cells']:
        library = Path(cell['library'])
        library.parent.mkdir(parents=True, exist_ok=True)
        if not library.exists():
            command = ['/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python',
                str(Path(scan['source_root']) / 'scripts/dnn1_cpu_compile.py'),
                '--compiler', scan['compiler'], '--source-root', scan['source_root'],
                '--input', cell['bridge'], '--coordinates', cell['coordinates'],
                '--target', cell['target'], '--output', str(library),
                '--reject-cuda', scan['reject_cuda'], '--batch', str(cell['batch'])]
            write(library.parent / 'command.json', command)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', LD_PRELOAD=scan['reject_cuda'],
                       PYTHONPATH=str(Path(scan['source_root']) / 'python'))
            with (library.parent / 'build.log').open('w') as log:
                subprocess.run(command, cwd=scan['source_root'], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        identity = validate_artifact(scan, cell)
        selection = dict(schema=1, artifact_id=identity['artifact_id'], binary_sha256=sha(library),
            identity_sha256=sha(str(library) + '.identity.json'), source_head=scan['head'],
            selection=dict(protocol='DNN2 fixed accepted coordinates', coordinates=cell['coordinates'],
                           coordinates_sha256=sha(cell['coordinates'])))
        write(cell['selection_config'], selection)
        records.append(dict(key=cell['key'], artifact_id=identity['artifact_id'],
                            binary_sha256=sha(library), identity_sha256=sha(str(library) + '.identity.json')))
        validate_live(scan)
    write(Path(scan['out']) / 'built.json', dict(head=scan['head'], scan_sha256=sha(args.scan), cells=records))


def empty_boundary(row):
    return not row.get('owners') and 0 <= row.get('used_mib', float('inf')) <= 1


def idle_boundary(cell, timeout_s=10., interval_s=.25, *, observe=None, clock=None, sleep=None):
    """Keep two consecutive process-and-memory observations, with no later re-read.

    The inherited drain counts process owners only. Its return followed by a
    separate memory sample is not a single boundary observation. Keep every
    sample, reset on either resource, and use only the final observed sample.
    """
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    if observe is None:
        imports(ROOT)
        from dnn1_baselines import exit_context_snapshot
        observe = lambda value: exit_context_snapshot(value, max(.001, timeout_s - (clock() - start)))
    start = clock()
    record = dict(started=time.time(), timeout_s=timeout_s, interval_s=interval_s,
                  required_empty_samples=2, memory_floor_mib=1, samples=[],
                  attribution='unknown; observations only', power_monitoring=False)
    streak = 0
    while True:
        try:
            row = observe(cell)
            if not isinstance(row.get('owners'), dict) or 'used_mib' not in row:
                raise ValueError('boundary telemetry omitted owners or memory')
        except Exception as error:
            record.update(status='telemetry_unavailable', error=str(error))
            break
        record['samples'].append(dict(elapsed_s=clock()-start, telemetry=row))
        streak = streak + 1 if empty_boundary(row) else 0
        if streak >= 2:
            record['status'] = 'empty_observed'
            break
        remaining = timeout_s - (clock()-start)
        if remaining <= 0:
            record['status'] = 'timeout'
            break
        sleep(min(interval_s, remaining))
    record.update(ended=time.time(), elapsed_s=clock()-start)
    return record


def final_boundary(drain):
    if drain.get('status') != 'empty_observed' or len(drain.get('samples', [])) < 2:
        raise ResourceRejection('two process-and-memory idle samples were not observed')
    if any(not empty_boundary(s['telemetry']) for s in drain['samples'][-2:]):
        raise ResourceRejection('final resource observations were not idle')
    return drain['samples'][-1]['telemetry']


def cell_run(args):
    scan, cell = read(args.scan), read(args.cell)
    require_freeze(scan, args.source_freeze)
    imports(scan['source_root'])
    import dnn1_baselines as protocol
    protocol.require_session(cell)
    identity = validate_artifact(scan, cell)
    args.out.mkdir(parents=True, exist_ok=False)
    record = dict(schema=1, head=scan['head'], scan_sha256=sha(args.scan), cell=cell,
        artifact_id=identity['artifact_id'], library_sha256=sha(cell['library']),
        status='running', correctness=[], timing=[], started=time.time())
    python = str(Path(scan['source_root']) / 'scripts/dnn1_idle_floor_python.py')
    env = dict(os.environ, TILEMEGA_FORWARD_CONSTANTS_FILE=cell['target'])
    try:
        for role, count in (('correctness', 5), ('timing', 3)):
            for index in range(count):
                validate_live(scan)
                before_drain = idle_boundary(cell)
                record.setdefault('boundaries', []).append(dict(role=role, index=index,
                    before_drain=before_drain))
                write(args.out / 'result.json', record)
                before = final_boundary(before_drain)
                folder = args.out / f'{role}-{index}'
                common = ['--library', cell['library'], '--export', cell['export'],
                          '--bridge', cell['bridge'], '--inputs', cell['inputs'],
                          '--batch', str(cell['batch'])]
                if role == 'correctness':
                    destination = folder / 'gate.json'
                    command = [python, '-m', 'tilemega.dnn.check', '--model', cell['model'],
                               *common, '--random-inputs', '--out', str(destination)]
                    if cell['model'] == 'nafnet':
                        command += ['--reference-cache', cell['reference_cache']]
                else:
                    destination = folder / 'payload/measurements/measurements.json'
                    command = [python, '-m', 'tilemega.dnn.benchmark_timing_only', *common,
                        '--protocol', 'P-DNN', '--selection-config', cell['selection_config'],
                        '--round-index', str(index), '--warmup', '50', '--iterations', '200',
                        '--out', str(folder / 'payload')]
                process = protocol.execute(command, folder, env, 3600)
                drain = idle_boundary(cell)
                after = drain['samples'][-1]['telemetry'] if drain['samples'] else {}
                process.update(index=index, receipt=str(destination),
                    receipt_sha256=sha(destination) if destination.exists() else None,
                    before=before, after=after, drain=drain, before_drain=before_drain)
                record[role].append(process)
                write(args.out / 'result.json', record)
                final_boundary(drain)
                if process['returncode']:
                    raise ValueError('fresh process failed; retain its original error')
                if role == 'correctness' and read(destination).get('passed') is not True:
                    raise ValueError('P5 numerical gate failed')
                if role == 'timing':
                    try:
                        timing_window(read(destination), [])
                    except ValueError as error:
                        raise ResourceRejection(str(error)) from error
        validate_live(scan)
        record.update(status='complete_pending_outer_audit', ended=time.time())
    except ResourceRejection as error:
        record.update(status='quarantined', error=str(error), ended=time.time())
        raise SystemExit(75)
    except Exception as error:
        record.update(status='rejected', error=str(error), ended=time.time())
        raise
    finally:
        write(args.out / 'result.json', record)


def queue_cell(scan, cell, gpu, affinity, freeze, attempt):
    imports(scan['source_root'])
    import dnn1_baselines as protocol
    folder = Path(scan['out']) / 'sessions' / cell['key'] / f'attempt-{attempt}'
    folder.mkdir(parents=True, exist_ok=False)
    value = dict(cell, gpu=gpu, cpu_affinity=affinity, parallel=True)
    write(folder / 'cell.json', value)
    python = str(Path(scan['source_root']) / 'scripts/dnn1_idle_floor_python.py')
    env = protocol.gpu_env(gpu, dict(TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
        OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', OMP_NUM_THREADS='4'))
    uuid, pci = [s.strip() for s in subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader,nounits'], text=True).strip().split(',')]
    domain, bus, slot = pci.split(':')
    pci = f'{int(domain, 16):04x}:{bus}:{slot}'
    command = ['numactl', '--all', '--physcpubind=' + ','.join(map(str, affinity)), python,
        str(Path(scan['source_root']) / 'scripts/run_r16_with_device_identity.py'),
        '--guard-policy', str(protocol.POLICY), '--out', '{attempt_dir}/job',
        '--expected-uuid', uuid, '--expected-pci', pci, '--numa-node', str(0 if gpu < 4 else 1),
        '--allow-no-measurements', '--', python, str(Path(__file__).resolve()), 'cell',
        '--scan', str(Path(scan['out']) / 'scan.json'), '--cell', str(folder / 'cell.json'),
        '--source-freeze', str(freeze), '--out', '{attempt_dir}/job/cell']
    write(folder / 'queue/queue_scoreboard.json', [dict(name=cell['key'], gpu=True, timing=True,
        priority=10, timeout_s=28800, needs_free_mib=8192, cwd=scan['source_root'],
        after=[], env=env, command=command)])
    launcher = ['flock', f'/tmp/tilemega_dnn2_scoreboard_gpu{gpu}.lock', 'env',
        '-u', 'TILEMEGA_DEVICE_GROUP', '-u', 'TILEMEGA_GPU_LOCK_HELD', '-u', 'TILEMEGA_RUN_NONCE',
        *[f'{key}={value}' for key, value in env.items()], python, '-m', 'tilemega.bench.scheduler',
        '--queue-dir', str(folder / 'queue'), '--out', str(folder / 'scheduler'),
        '--policy', str(protocol.POLICY), '--deadline-hours', '96']
    log = (folder / 'scheduler.log').open('w')
    process = subprocess.Popen(launcher, cwd=scan['source_root'], stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(folder / 'launch.json', dict(pid=process.pid, command=launcher, time=time.time()))
    return process, log, folder


def run(args):
    scan = read(args.scan)
    require_freeze(scan, args.source_freeze)
    built = read(Path(scan['out']) / 'built.json')
    if built['head'] != scan['head'] or built['scan_sha256'] != sha(args.scan) or len(built['cells']) != 10:
        raise ValueError('all ten HEAD-bound artifacts must be built first')
    imports(scan['source_root'])
    import dnn1_baselines as protocol
    import dnn1_device as device
    from dnn1_dispatch import affinities
    gpus = [int(x) for x in args.gpus.split(',')]
    if not gpus or len(set(gpus)) != len(gpus) or not set(gpus) <= set(GPUS) or not 1 <= args.max_parallel <= 3:
        raise ValueError('distinct healthy cards excluding GPU6; at most three simultaneous cells')
    binding = affinities(gpus, 8)
    lock = (Path(scan['out']) / 'dispatcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending, active, done, failed = [], {}, [], []
    attempts = {}
    for cell in scan['cells']:
        parent = Path(scan['out']) / 'sessions' / cell['key']
        attempts[cell['key']] = len(list(parent.glob('attempt-*')))
        accepted = parent / 'accepted.json'
        if accepted.exists():
            receipt = read(accepted)
            if receipt['head'] != scan['head'] or receipt['scan_sha256'] != sha(args.scan):
                raise ValueError('cannot resume a cell from another scan or HEAD')
            done.append(cell['key'])
        else:
            if attempts[cell['key']] >= 2:
                raise ValueError('two complete attempts retained; inspect the failure before resuming')
            pending.append(cell)
    cancelled = False
    def cancel(*_):
        nonlocal cancelled
        cancelled = True
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        while pending or active:
            validate_live(scan)
            for gpu, (process, log, cell, folder) in list(active.items()):
                if process.poll() is None:
                    continue
                log.close()
                state = read(folder / 'scheduler/state.json').get(cell['key'], {})
                if state.get('status') == 'done':
                    write(folder.parent / 'accepted.json', dict(head=scan['head'],
                        scan_sha256=sha(args.scan), folder=str(folder)))
                    done.append(cell['key'])
                else:
                    raw = Path(state['out']) if state.get('out') else None
                    guard = read(raw / 'guard_result.json') if raw and (raw/'guard_result.json').exists() else {}
                    if guard.get('code') == 75 and attempts[cell['key']] < 2:
                        pending.insert(0, cell)
                    else:
                        failed.append(cell['key'])
                del active[gpu]
            if not cancelled and not failed:
                # Prefer a free physical lock, but let the ordinary guard
                # queue behind a holder when an owner-free card is between
                # payloads. Requiring an unheld lock here starves under a
                # continuous stream of other authorized jobs.
                for gpu in sorted(gpus, key=lambda g: protocol.held(f'/tmp/tilemega_gpu{g}.lock')):
                    if not pending or len(active) >= args.max_parallel:
                        break
                    if gpu in active:
                        continue
                    sample = device.gpu(gpu)
                    if not empty_boundary(sample) or sample['free_mib'] < 8192:
                        continue
                    cell = pending.pop(0)
                    attempts[cell['key']] += 1
                    process, log, folder = queue_cell(scan, cell, gpu, binding[gpu],
                        args.source_freeze.resolve(), attempts[cell['key']])
                    active[gpu] = (process, log, cell, folder)
            write(Path(scan['out']) / 'dispatch.json', dict(head=scan['head'],
                status='cancelled' if cancelled else ('rejected' if failed else 'collecting'),
                pending=[c['key'] for c in pending], active={str(g): dict(pid=p.pid, key=c['key'])
                    for g, (p, _, c, _) in active.items()}, done=done, failed=failed, time=time.time()))
            if cancelled or failed:
                if cancelled or not active:
                    break
            time.sleep(5)
    finally:
        from tilemega.bench.gpu_guard import stop
        for process, log, _, _ in active.values():
            stop(process)
            log.close()
    if cancelled or failed or pending:
        raise RuntimeError('incomplete scan retained; no scoreboard rows published')


def timing_window(receipt, outer):
    samples = [json.loads(line) for line in Path(receipt['guard_log']).read_text().splitlines()]
    initialization = [s for s in samples if s['label'] == 'before-initialization']
    if (len(initialization) != 1 or initialization[0]['pids'] or
            float(initialization[0]['telemetry']['memory.used']) > 1):
        raise ValueError('timing initialization was not empty')
    before = [s for s in samples if s['label'] == 'before-measurement']
    after = [s for s in samples if s['label'] == 'after-measurement']
    if len(before) != 1 or len(after) != 1:
        raise ValueError('missing or duplicate timing boundaries')
    a, b = before[0], after[0]
    pids = set(map(str, a['pids']))
    if (len(pids) != 1 or pids != set(map(str, b['pids'])) or a['time'] > b['time'] or
            initialization[0]['time'] > a['time']):
        raise ValueError('timing boundaries require one stable singleton PID')
    for s in samples:
        if a['time'] <= s['time'] <= b['time'] and (
                set(map(str, s['pids'])) != pids or s.get('foreign_pids') or not s.get('accepted', True)):
            raise ValueError('dirty inner timing window')
    observed = [s for s in outer if a['time'] <= s.get('time', 0) <= b['time']]
    if any(set(map(str, s.get('owners', {}))) != pids for s in observed):
        raise ValueError('extra or missing owner in outer timing window')
    return dict(singleton_pid=next(iter(pids)), outer_samples=len(observed),
                outer_observation='sampled' if observed else 'no outer sample in this short window',
                scope='inner boundary proof; sampled outer evidence only, container policy retained')


def collect_cell(scan, cell, session_path, guard_path):
    session, guard = read(session_path), read(guard_path)
    if guard.get('code') != 0 or session.get('status') != 'complete_pending_outer_audit':
        raise ValueError('cell requires completed session and accepted outer guard')
    if session['head'] != scan['head'] or session['scan_sha256'] != scan['_sha256']:
        raise ValueError('cell belongs to another HEAD or full-scan manifest')
    if (session['cell']['model'], session['cell']['batch']) != (cell['model'], cell['batch']):
        raise ValueError('cell result was relabelled')
    if session['cell']['gpu'] not in GPUS:
        raise ValueError('cell ran on an excluded device')
    checks, times = session['correctness'], session['timing']
    if len(checks) != 5 or len(times) != 3 or len({p['pid'] for p in checks + times}) != 8:
        raise ValueError('P5 and P-DNN require eight distinct fresh processes')
    outer = [json.loads(line) for line in guard_path.with_name('guard.jsonl').read_text().splitlines()]
    medians, p90s, windows = [], [], []
    for role, processes in (('correctness', checks), ('timing', times)):
        for index, process in enumerate(processes):
            if (process['returncode'] or process['receipt_sha256'] != sha(process['receipt']) or
                    not empty_boundary(process['before']) or not empty_boundary(process['after']) or
                    process['drain']['status'] != 'empty_observed'):
                raise ValueError('process failed, evidence changed, or process boundary was dirty')
            if (final_boundary(process['before_drain']) != process['before'] or
                    final_boundary(process['drain']) != process['after']):
                raise ValueError('boundary metadata is not its final consecutive-idle observation')
            value = read(process['receipt'])
            if value['artifact_id'] != session['artifact_id'] or value['pid'] != process['pid']:
                raise ValueError('processes used different artifacts')
            if role == 'correctness':
                if (value.get('passed') is not True or value.get('batch') != cell['batch'] or
                        value.get('model') != cell['model'] or
                        value.get('library_sha256') != session['library_sha256'] or
                        value.get('input_tensors_sha256') != sha(cell['inputs'])):
                    raise ValueError('numerical gate/input mismatch')
            else:
                if (value['selected_source_head'] != scan['head'] or value['round_index'] != index or
                        value['identity']['source']['head'] != scan['head'] or
                        value['identity']['binary_sha256'] != session['library_sha256'] or
                        value['gpu_index'] != session['cell']['gpu'] or value['batch'] != cell['batch'] or
                        value['input_sha256'] != sha(cell['inputs']) or
                        value['measurement']['warmup'] != 50 or value['measurement']['iterations'] != 200 or
                        len(value['samples_ms']) != 200 or value['protocol'] != 'P-DNN'):
                    raise ValueError('timing HEAD/library/input/protocol mismatch')
                numbers = [value['median_ms'], value['p90_ms'], *value['samples_ms']]
                if any(not math.isfinite(x) or x <= 0 for x in numbers):
                    raise ValueError('invalid timing sample')
                samples = sorted(value['samples_ms'])
                if (not math.isclose(value['median_ms'], statistics.median(samples), rel_tol=1e-12) or
                        not math.isclose(value['p90_ms'], samples[179], rel_tol=1e-12)):
                    raise ValueError('declared timing statistics differ from the 200 samples')
                medians.append(value['median_ms'])
                p90s.append(value['p90_ms'])
                windows.append(timing_window(value, outer))
    metrics = [read(p['receipt'])['metrics'] for p in checks]
    if any(value != metrics[0] for value in metrics[1:]):
        raise ValueError('fresh-process correctness metrics differ')
    return dict(head=scan['head'], promoted_change=scan['promoted_change'], model=cell['model'],
        batch=cell['batch'], gpu=session['cell']['gpu'], median_ms=statistics.median(medians),
        p90_ms=statistics.median(p90s), process_range_ms=max(medians) - min(medians),
        correctness='pass:5/5', source=str(session_path), windows=windows)


def publish_rows(scan, records, destination):
    check_grid(records)
    if any(r['head'] != scan['head'] or r['promoted_change'] != scan['promoted_change'] for r in records):
        raise ValueError('refuse mixed-HEAD or mixed-promotion scoreboard')
    rows = []
    for key in CELLS:
        record = dict(next(r for r in records if (r['model'], r['batch']) == key))
        baseline = scan['baselines'][f'{key[0]}_b{key[1]}']
        record.update(baseline_ms=baseline['ms'], baseline_system=baseline['system'],
                      ratio=record['median_ms'] / baseline['ms'])
        rows.append({field: record[field] for field in FIELDS})
    previous = []
    if destination.exists():
        with destination.open() as stream:
            previous = list(csv.DictReader(stream))
    groups = {}
    for row in previous:
        groups.setdefault((row['head'], row['promoted_change']), []).append(row)
    for group in groups.values():
        check_grid(group)
    if (scan['head'], scan['promoted_change']) in groups:
        raise ValueError('that complete checkpoint is already published; preserve its original scan')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(previous + rows)
    temporary.replace(destination)
    return rows


def collect(args):
    scan = read(args.scan)
    validate_live(scan)
    scan['_sha256'] = sha(args.scan)
    records = []
    for cell in scan['cells']:
        identity = validate_artifact(scan, cell)
        parent = Path(scan['out']) / 'sessions' / cell['key']
        selected = read(parent / 'accepted.json')
        if selected['head'] != scan['head'] or selected['scan_sha256'] != scan['_sha256']:
            raise ValueError('accepted cell locator belongs to another scan')
        folder = Path(selected['folder'])
        state = read(folder / 'scheduler/state.json').get(cell['key'], {})
        if state.get('status') != 'done':
            raise ValueError('no partial scoreboard: cell is incomplete ' + cell['key'])
        attempt = Path(state['out'])
        result = read(attempt / 'job/cell/result.json')
        if result['artifact_id'] != identity['artifact_id'] or result['library_sha256'] != sha(cell['library']):
            raise ValueError('completed cell differs from the HEAD-bound artifact')
        records.append(collect_cell(scan, cell, attempt / 'job/cell/result.json', attempt / 'guard_result.json'))
    rows = publish_rows(scan, records, args.scoreboard.resolve())
    write(Path(scan['out']) / 'accepted.json', dict(head=scan['head'], rows=rows,
        source_scan_sha256=scan['_sha256'], baseline_revision=scan['baseline_revision'],
        baseline_csv_sha256=scan['baseline_csv_sha256'], records=records))
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    for name in ('source-root', 'compiler', 'exports', 'fixed-coordinates', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--promoted-change', required=True)
    p.add_argument('--baseline-revision', default='9945dcb02')
    p.add_argument('--reject-cuda', type=Path, required=True)
    p = sub.add_parser('build'); p.add_argument('--scan', type=Path, required=True)
    p = sub.add_parser('run'); p.add_argument('--scan', type=Path, required=True)
    p.add_argument('--source-freeze', type=Path, required=True)
    p.add_argument('--gpus', default='0,1,2,3,4,5,7'); p.add_argument('--max-parallel', type=int, default=3)
    p = sub.add_parser('cell')
    for name in ('scan', 'cell', 'source-freeze', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p = sub.add_parser('collect'); p.add_argument('--scan', type=Path, required=True)
    p.add_argument('--scoreboard', type=Path, required=True)
    args = parser.parse_args()
    return {'prepare': prepare, 'build': build, 'run': run, 'cell': cell_run, 'collect': collect}[args.action](args)


if __name__ == '__main__':
    main()
