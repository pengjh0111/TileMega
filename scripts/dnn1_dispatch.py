#!/usr/bin/env python3
"""Monitor healthy GPUs and dispatch baseline cells through existing guards."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import dnn1_baselines as baseline
import dnn1_device as device

ROOT = baseline.ROOT


def fresh_recovery(config, cell_key, worker_folder):
    """Retain prior incomplete caches when a migrated cell needs another root."""
    recovery = dict(config.get('tvm_infrastructure_recovery', {}))
    if cell_key in recovery:
        value = dict(recovery[cell_key])
        previous = Path(value['work_dir'])
        if previous.exists():
            fresh = previous.with_name(previous.name + '__' + worker_folder.name)
            if fresh.exists():
                raise FileExistsError('recovery allocation must never overwrite ' + str(fresh))
            value.update(work_dir=str(fresh), previous_recovery_work_dir=str(previous),
                         recovery_reason='new queued attempt retains all previous cache evidence')
            recovery[cell_key] = value
    return recovery


def cpu_list(text):
    result = set()
    for item in text.strip().split(','):
        bounds = list(map(int, item.split('-')))
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def affinities(gpus, count, allowed=None, explicit=None):
    allowed = set(os.sched_getaffinity(0) if allowed is None else allowed)
    result = {}
    if explicit is not None:
        if set(map(int, explicit)) != set(gpus):
            raise ValueError('explicit CPU affinity must cover exactly the selected GPUs')
        occupied = set()
        for gpu in gpus:
            cpus = explicit[str(gpu)] if str(gpu) in explicit else explicit[gpu]
            node = 0 if gpu < 4 else 1
            local = cpu_list(Path(f'/sys/devices/system/node/node{node}/cpulist').read_text())
            if (not cpus or any(type(cpu) is not int for cpu in cpus)
                    or len(set(cpus)) != len(cpus) or not set(cpus) <= allowed & local
                    or occupied & set(cpus)):
                raise ValueError('explicit CPU affinity must be allowed, NUMA-local and disjoint')
            result[gpu] = sorted(cpus)
            occupied.update(cpus)
        return result
    for node in (0, 1):
        devices = [g for g in gpus if (0 if g < 4 else 1) == node]
        cpus = sorted(cpu_list(Path(f'/sys/devices/system/node/node{node}/cpulist').read_text()) & allowed)
        if len(cpus) < count * len(devices):
            raise ValueError('not enough disjoint CPUs on NUMA node ' + str(node))
        for index, gpu in enumerate(devices):
            result[gpu] = cpus[index*count:(index+1)*count]
    return result


def accepted(root):
    if not (root / 'scheduler/state.json').is_file():
        return []
    spec = baseline.read(root / 'queue-spec.json')
    state = baseline.read(root / 'scheduler/state.json')
    result = []
    for cell_path in spec['cells']:
        cell = baseline.read(cell_path)
        key = f"{cell['model']}_b{cell['batch']}"
        record = state.get('dnn1-cell-' + key, {})
        if record.get('status') != 'done':
            continue
        attempt = Path(record['out'])
        guard = baseline.read(attempt / 'guard_result.json')
        session = baseline.read(attempt / 'job/cell-result.json')
        if guard.get('code') == 0 and session.get('status') == 'complete':
            result.append(dict(key=key, cell=cell_path, state=record, run_root=str(root)))
    return result


def available(row, policy, needs_mib, locked):
    if locked:
        return False, 'physical GPU lock held'
    if not device.idle(row, policy, needs_mib):
        return False, 'process or extra memory occupancy; or insufficient free memory'
    return True, 'idle scheduling hint; child guard still makes the final decision'


def halt_worker(process):
    from tilemega.bench.gpu_guard import descendants, stop
    for pid in descendants(process.pid):
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes()
            if str(ROOT / 'scripts/dnn1_guard_entry.py').encode() in command:
                os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    # The DNN guard handles its separately sessioned children before releasing
    # physical locks. This only targets descendants of this worker launcher.
    time.sleep(2.2)
    stop(process)


class AdoptedWorker:
    """Observe a surviving launcher without restarting its GPU descendants."""
    def __init__(self, pid, root):
        from tilemega.bench.gpu_guard import process_start
        self.pid, self.root = pid, Path(root)
        self.started = process_start(pid)
        self.returncode = None
        stat = Path(f'/proc/{pid}/stat')
        live = self.started is not None and stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
        if live:
            command = Path(f'/proc/{pid}/cmdline').read_bytes()
            if str(self.root / 'launch.sh').encode() not in command:
                raise ValueError('adoption PID does not identify the recorded launcher')

    def poll(self):
        from tilemega.bench.gpu_guard import process_start
        stat = Path(f'/proc/{self.pid}/stat')
        try:
            alive = (self.started is not None and process_start(self.pid) == self.started
                     and stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z')
        except OSError:
            alive = False
        if alive:
            return None
        self.returncode = 0 if len(accepted(self.root)) == 1 else 1
        return self.returncode

    def wait(self, timeout=None):
        started = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - started >= timeout:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(.05)
        return self.returncode


def completed_rejections(root, job):
    """Verify historical payloads are rejected and all recorded processes exited.

    This permits cached remeasurement to leave a busy card without accepting
    any part of the rejected session. Missing or live evidence remains a veto.
    """
    from tilemega.bench.gpu_guard import process_start, session_members
    payloads = list((root / 'raw').glob('dnn1-cell-*/attempt*/job'))
    if not payloads:
        return []
    if job.get('exit_code') != 75 or not job.get('ended'):
        return None
    receipts = []
    for payload in payloads:
        guard_path, session_path = payload.parent/'guard_result.json', payload/'cell-result.json'
        if not guard_path.is_file() or not session_path.is_file():
            return None
        guard, session = baseline.read(guard_path), baseline.read(session_path)
        if (guard.get('code') != 75 or session.get('status') not in ('hard_stop', 'complete')
                or not session.get('ended') or session['ended'] > job['ended']):
            return None
        attempts = list(payload.rglob('logical-attempt.json'))
        if not attempts:
            return None
        exited = []
        for path in attempts:
            attempt = baseline.read(path)
            pid = attempt.get('pid')
            if (not pid or 'returncode' not in attempt or not attempt.get('ended')
                    or attempt['ended'] > session['ended']
                    or process_start(pid) is not None or session_members(pid)):
                return None
            exited.append(dict(path=str(path), sha256=baseline.sha(path), pid=pid))
        receipts.append(dict(session=str(session_path), session_sha256=baseline.sha(session_path),
            guard=str(guard_path), guard_sha256=baseline.sha(guard_path), exited_processes=exited))
    return receipts


def initial_wait_probe(root):
    from tilemega.bench.gpu_guard import process_start
    try:
        pid = int((root / 'scheduler/scheduler.pid').read_text())
        process = Path(f'/proc/{pid}')
        wait = (process / 'wchan').read_text().strip()
        descriptors = {p.name: os.readlink(p) for p in (process / 'fd').iterdir()}
        return dict(pid=pid, start=process_start(pid), wchan=wait, descriptors=descriptors)
    except (OSError, ValueError):
        return None


def initial_cpu_waiter(root, key, commands, probe):
    """Prove a scheduler has not passed its first CPU-export lock acquisition."""
    if (root / 'scheduler/state.json').exists():
        return None
    if any(path.is_file() for path in (root / 'raw').rglob('*')):
        return None
    queue_paths = list((root / 'queue').glob('queue_*.json'))
    if len(queue_paths) != 1:
        return None
    steps = baseline.read(queue_paths[0])
    if len(steps) != 2:
        return None
    export, cell = steps
    if (export.get('name') != 'dnn1-export-' + key or export.get('gpu') is not False
            or export.get('after') or cell.get('name') != 'dnn1-cell-' + key
            or cell.get('gpu') is not True or cell.get('after') != [export['name']]):
        return None
    command = export.get('command', [])
    if (len(command) < 3 or command[0] != 'flock'
            or not command[1].startswith('/tmp/tilemega_gpu')
            or not probe or probe.get('wchan') != 'locks_lock_inode_wait'
            or command[1] not in probe.get('descriptors', {}).values()):
        return None
    scheduler_path = root / 'scheduler/scheduler.pid'
    if not scheduler_path.is_file():
        return None
    scheduler_pid = scheduler_path.read_text().strip()
    if int(scheduler_pid) != probe.get('pid') or probe.get('start') is None:
        return None
    scheduler_command = commands.get(scheduler_pid, '')
    if ('-m tilemega.bench.scheduler ' not in scheduler_command
            or '--queue-dir ' + str(root / 'queue') not in scheduler_command
            or '--out ' + str(root / 'scheduler') not in scheduler_command):
        return None
    # A child payload may have escaped the launcher's process group. Refuse any
    # descendant other than the known launcher/flock/scheduler chain.
    for pid, command in commands.items():
        if pid == scheduler_pid:
            continue
        if command.strip() == 'bash ' + str(root / 'launch.sh'):
            continue
        if (command.startswith('flock /tmp/tilemega_dnn1_baseline_scheduler_gpu')
                and '-m tilemega.bench.scheduler ' in command
                and '--queue-dir ' + str(root / 'queue') in command):
            continue
        return None
    return dict(stage='before_first_state_persist', scheduler_pid=int(scheduler_pid),
                queue=str(queue_paths[0]), queue_sha256=baseline.sha(queue_paths[0]),
                raw_files=[], commands=commands, lock_wait_probe=probe)


def migrate_waiter(worker):
    """Cancel only owned CPU waiters with no live or unfinished GPU payload."""
    from tilemega.bench.gpu_guard import descendants, process_start
    process, root = worker['process'], worker['root']
    state_path = root / 'scheduler/state.json'
    jobs = baseline.read(state_path) if state_path.is_file() else None
    if jobs is not None and jobs.get('dnn1-cell-' + worker['key'], {}).get('status') != 'pending':
        return None
    probe = initial_wait_probe(root) if jobs is None else None
    if jobs is None and not probe:
        return None
    try:
        if process.poll() is not None or os.getpgid(process.pid) != process.pid:
            return None
    except ProcessLookupError:
        return None
    os.killpg(process.pid, signal.SIGSTOP)
    cancelled = False
    try:
        ids = descendants(process.pid)
        commands = {str(pid): Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
                    for pid in ids if Path(f'/proc/{pid}/cmdline').exists()}
        initial = None
        if not state_path.is_file():
            initial = initial_cpu_waiter(root, worker['key'], commands, probe)
            if initial is None:
                return None
            if process_start(probe['pid']) != probe['start']:
                return None
            jobs, rejected = {}, []
        else:
            jobs = baseline.read(state_path)
            job = jobs.get('dnn1-cell-' + worker['key'], {})
            if job.get('status') != 'pending':
                return None
            rejected = completed_rejections(root, job)
            if rejected is None:
                return None
        if any('dnn1_guard_entry.py' in command or '--tune-request' in command
               for command in commands.values()):
            return None
        if rejected and any('tilemega.bench.scheduler' not in command
                            for pid, command in commands.items() if int(pid) != process.pid):
            return None
        starts = {str(pid): process_start(pid) for pid in ids}
        receipt = dict(time=time.time(), key=worker['key'], launcher_pid=process.pid,
            scheduler_state=jobs, descendants=commands, process_start=starts,
            completed_rejected_payloads=rejected,
            initial_cpu_waiter=initial,
            reason='no live GPU payload; preserve rejected history and requeue through a fresh guard')
        baseline.write(root / 'migration.json', receipt)
        os.killpg(process.pid, signal.SIGTERM)
        os.killpg(process.pid, signal.SIGCONT)
        cancelled = True
        process.wait(timeout=10)
        return receipt
    finally:
        if not cancelled:
            try:
                os.killpg(process.pid, signal.SIGCONT)
            except ProcessLookupError:
                pass


def blocking_failures(failures):
    """A completed resource rejection must not starve unrelated queued cells."""
    blocking = []
    for failure in failures:
        root = failure.get('root')
        sessions = list(Path(root).glob('raw/dnn1-cell-*/attempt*/job/cell-result.json')) if root else []
        results = [baseline.read(path) for path in sessions]
        latest = max(results, key=lambda row: row.get('ended') or 0) if results else {}
        error = latest.get('error', '')
        resource = ('whole cell invalidated by timing guard/throttle' in error or
                    'TVM infrastructure rejection; no FP16 alternative' in error)
        if not (latest.get('status') == 'hard_stop' and latest.get('ended') and resource):
            blocking.append(failure)
    return blocking


def main():
    global baseline
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-spec', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6')
    p.add_argument('--max-parallel', type=int, default=7)
    p.add_argument('--cpus-per-worker', type=int, default=8)
    p.add_argument('--poll-seconds', type=float, default=15)
    p.add_argument('--reuse-root', type=Path, action='append', default=[])
    p.add_argument('--once', action='store_true', help='CPU-only monitoring snapshot; dispatch nothing')
    p.add_argument('--resume', action='store_true', help='adopt recorded launchers; never restart their payloads')
    a = p.parse_args()
    sys.path.insert(0, str(ROOT / 'python'))
    from tilemega.bench.gpu_guard import require_allowed_gpu
    gpu=device.gpu
    gpus = list(map(int, a.gpus.split(',')))
    if (not gpus or len(gpus) != len(set(gpus)) or any(g not in baseline.PCI for g in gpus)
            or any(g not in device.allowed_gpu_indices() for g in gpus)
            or a.max_parallel < 1 or a.cpus_per_worker < 1 or a.poll_seconds < 1):
        p.error('distinct currently mapped GPUs and positive limits are required')
    policy = baseline.read(baseline.POLICY)
    for g in gpus:
        require_allowed_gpu(policy, str(g), timing=True)
    config = baseline.read(a.base_spec)
    driver_name = config.get('baseline_driver', 'dnn1_baselines')
    if driver_name not in ('dnn1_baselines', 'dnn1_tvm_infra_driver', 'dnn1_tvm_reuse_driver'):
        raise ValueError('unsupported versioned CPU baseline driver')
    baseline = importlib.import_module(driver_name)
    binding = affinities(gpus, a.cpus_per_worker, explicit=config.get('cpu_affinity'))
    python = Path(config.get('python', ROOT/'scripts/dnn1_supervised_python.py')).absolute()
    if not python.is_file():
        raise FileNotFoundError(python)
    systems = config.get('systems', ['tensorrt', 'torchcompile'])
    if not systems or not set(systems) <= {'tensorrt', 'torchcompile', 'tvm'}:
        raise ValueError('systems must select supported baseline systems')
    if python == ROOT/'scripts/dnn1_idle_floor_python.py':
        from dnn1_idle_floor_python import idle
        device.idle = idle
    a.out.mkdir(parents=True, exist_ok=True)
    singleton = (a.out / 'dispatcher.lock').open('a')
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = baseline.read(a.out / 'state.json') if a.resume else None
    if (a.out / 'state.json').exists() and not a.resume:
        raise FileExistsError('retain dispatcher evidence; choose a new output directory')
    sources = [Path(__file__), ROOT / 'scripts/dnn1_baselines.py',
               ROOT / 'python/tilemega/dnn/baselines.py', ROOT / 'python/tilemega/dnn/tvm_baseline.py',
               ROOT/'scripts/dnn1_device.py',ROOT/'scripts/dnn1_resource_entry.py',
               ROOT/'scripts/dnn1_supervised_python.py',ROOT/'scripts/dnn1_guard_entry.py',python]
    if driver_name != 'dnn1_baselines':
        sources.append(ROOT/'scripts'/f'{driver_name}.py')
    if driver_name == 'dnn1_tvm_reuse_driver':
        # Existing adopted payloads still use v1; keep pinning its unchanged file.
        sources.append(ROOT/'scripts/dnn1_tvm_infra_driver.py')
    frozen = {str(path.resolve()): baseline.sha(path) for path in sources}
    if previous:
        changes = {path: dict(before=digest, after=frozen.get(path))
                   for path, digest in previous['source_sha256'].items()
                   if frozen.get(path) != digest}
        if set(changes) - {str(Path(__file__).resolve())}:
            raise ValueError('resume may change only the CPU dispatcher source')
        if (previous['gpus'] != gpus or previous['cpu_affinity'] != {str(g): cpus for g, cpus in binding.items()}
                or previous['systems'] != systems or previous['python'] != str(python)):
            raise ValueError('resume must preserve the recorded resource and framework configuration')
    prior = [record for root in a.reuse_root for record in accepted(root.resolve())]
    if len({r['key'] for r in prior}) != len(prior):
        raise ValueError('duplicate accepted prior cells; select one complete session explicitly')
    completed = {r['key'] for r in prior}
    pairs = config.get('cells') or [dict(model=m, batch=b) for m in baseline.MODELS for b in baseline.BATCHES]
    pending = [pair for pair in pairs if f"{pair['model']}_b{pair['batch']}" not in completed]
    if len({(r['model'],r['batch']) for r in pending}) != len(pending):
        raise ValueError('duplicate pending baseline cells')
    state = dict(schema=1, status='monitoring', pid=os.getpid(), started=time.time(),
        resource_policy=device.POLICY,power_monitoring=False,
        gpus=gpus, excluded_gpus=[], max_parallel=a.max_parallel, cpu_affinity=binding,
        pending=pending, workers={}, accepted=prior, failures=[], source_sha256=frozen,
        scheduling_amendment='user 2026-11-06: seven remaining GPUs authorized after removal; parallel cells, no same-card/session constraint',
        systems=systems, python=str(python),
        tvm_status='scheduled' if 'tvm' in systems else 'not requested',
        numa_concurrency='user-authorized trial with disjoint CPU sets; not claimed verified')
    (a.out / 'dispatcher.pid').write_text(str(os.getpid()) + '\n')
    workers = {}
    if previous:
        state = previous
        state.setdefault('resume_history', []).append(dict(time=time.time(), previous_pid=state['pid'],
            pid=os.getpid(), source_changes=changes))
        state.update(pid=os.getpid(), source_sha256=frozen, status='resuming')
        pending = state['pending']
        for g, record in state['workers'].items():
            root = Path(record['run_root'])
            workers[int(g)] = dict(process=AdoptedWorker(record['pid'], root), root=root,
                key=record['key'], log=(root.parent / 'worker.log').open('a'))
    state['baseline_driver'] = driver_name
    stopping = False
    def event(kind, **values):
        row = dict(time=time.time(), event=kind, **values)
        with (a.out / 'events.jsonl').open('a') as output:
            output.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    def stop_requested(number, _frame):
        nonlocal stopping
        stopping = True
        event('stop_requested', signal=number)
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, stop_requested)
    def persist():
        state['updated'] = time.time()
        baseline.write(a.out / 'state.json', state)
    def collect_view():
        view = a.out / 'collection-view'
        cells = [r['cell'] for r in state['accepted']]
        baseline.write(view / 'queue-spec.json', dict(cells=cells, dispatcher=str(a.out.resolve())))
        baseline.write(view / 'scheduler/state.json',
            {'dnn1-cell-' + r['key']:r['state'] for r in state['accepted']})
        baseline.collect(argparse.Namespace(run_root=view, out=a.out / 'collection'))
    try:
        while True:
            for g, worker in list(workers.items()):
                process = worker['process']
                if process.poll() is None:
                    q = worker['root'] / 'scheduler/state.json'
                    if q.exists():
                        jobs = baseline.read(q)
                        job = jobs.get('dnn1-cell-' + worker['key'], {})
                        state['workers'][str(g)]['scheduler_status'] = job.get('status','preparing')
                        if job.get('out'):
                            report = Path(job['out']) / 'job/cell-result.json'
                            if report.is_file():
                                current = baseline.read(report)
                                state['workers'][str(g)].update(logical_jobs_finished=len(current.get('logical_jobs',[])),
                                    latest_position=current.get('order',[])[-1:] or None)
                    continue
                worker['log'].close()
                valid = accepted(worker['root'])
                if process.returncode == 0 and len(valid)==1:
                    state['accepted'].extend(valid)
                    event('cell_accepted', gpu=g, key=worker['key'], root=str(worker['root']))
                    collect_view()
                else:
                    failure = dict(gpu=g, key=worker['key'], root=str(worker['root']), exit_code=process.returncode)
                    state['failures'].append(failure)
                    event('worker_failed', **failure)
                del workers[g]
                del state['workers'][str(g)]
            if stopping or (a.out / 'STOP').exists():
                for g, worker in workers.items():
                    halt_worker(worker['process'])
                    worker['log'].close()
                    event('worker_cancelled', gpu=g, key=worker['key'])
                state['status']='stopped';persist();return 2
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                futures = {g:pool.submit(gpu,g) for g in gpus}
                snapshots = {}
                for g, future in futures.items():
                    try:
                        row = future.result()
                        lock = baseline.held(f'/tmp/tilemega_gpu{g}.lock')
                        eligible, reason = available(row,policy,config.get('needs_free_mib',8192),lock)
                        snapshots[str(g)]=dict(row,physical_lock_held=lock,eligible=eligible,reason=reason)
                    except Exception as error:
                        snapshots[str(g)]=dict(eligible=False,error=str(error))
            state['devices']=snapshots
            if a.once:
                state['status']='monitor_only';persist();print(json.dumps(snapshots),flush=True);return 0
            if any(baseline.sha(path)!=digest for path,digest in frozen.items()):
                state['failures'].append(dict(reason='baseline recipe changed during dispatcher lifetime'))
            for g, worker in list(workers.items()):
                alternatives = [card for card in gpus if card not in workers
                                and snapshots[str(card)]['eligible']]
                if (snapshots[str(g)]['eligible']
                        or time.time() - state['workers'][str(g)]['assigned'] < 2*a.poll_seconds):
                    continue
                try:
                    receipt = migrate_waiter(worker)
                except Exception as error:
                    event('migration_error', gpu=g, key=worker['key'], error=repr(error))
                    continue
                if receipt is None:
                    continue
                cell = baseline.read(baseline.read(worker['root'] / 'queue-spec.json')['cells'][0])
                pending.insert(0, dict(model=cell['model'], batch=cell['batch']))
                state.setdefault('migrations', []).append(dict(gpu=g, root=str(worker['root']), **receipt))
                worker['log'].close()
                event('pending_migrated', gpu=g, key=worker['key'], alternatives=alternatives,
                      root=str(worker['root']))
                del workers[g]
                del state['workers'][str(g)]
            for g in gpus:
                if (not pending or len(workers)>=a.max_parallel or blocking_failures(state['failures'])):
                    break
                if g in workers or not snapshots[str(g)]['eligible']:
                    continue
                pair=pending.pop(0);key=f"{pair['model']}_b{pair['batch']}"
                folder=a.out/'workers'/key
                version=1
                while folder.exists():
                    version+=1
                    folder=a.out/'workers'/f'{key}__dispatch{version:03d}'
                folder.mkdir(parents=True,exist_ok=False)
                child=dict(config,gpu=g,cells=[pair],parallel=True,cpu_affinity=binding[g],
                    systems=state['systems'],scheduling_amendment=state['scheduling_amendment'],
                    python=str(python))
                if config.get('tvm_infrastructure_recovery'):
                    child['tvm_infrastructure_recovery'] = fresh_recovery(config,key,folder)
                child['native_prefixes']={}
                baseline.write(folder/'input.json',child)
                baseline.queue(argparse.Namespace(spec=folder/'input.json',out=folder/'run',refresh_prepared=False))
                log=(folder/'worker.log').open('w')
                process=subprocess.Popen(['bash',str(folder/'run/launch.sh')],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                workers[g]=dict(process=process,log=log,root=folder/'run',key=key)
                state['workers'][str(g)]=dict(key=key,pid=process.pid,run_root=str(folder/'run'),
                    assigned=time.time(),cpu_affinity=binding[g])
                event('cell_assigned',gpu=g,key=key,pid=process.pid,cpu_affinity=binding[g])
            state['pending']=pending
            if not workers and (not pending or blocking_failures(state['failures'])):
                state['status']='primary_complete' if not state['failures'] else 'needs_diagnosis'
                collect_view();persist();return 0 if not state['failures'] else 2
            state['status']='running' if workers else 'waiting_for_idle_cards'
            persist()
            time.sleep(a.poll_seconds)
    except BaseException:
        for worker in workers.values():
            halt_worker(worker['process'])
            worker['log'].close()
        state['status']='dispatcher_error';persist();raise


if __name__ == '__main__':
    raise SystemExit(main())
