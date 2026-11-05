#!/usr/bin/env python3
"""Release AIT only after source-bound TVM25, then dispatch guarded P5 cells."""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

import dnn1_baselines as base
import dnn1_device as device
import dnn1_aitemplate as runner
import dnn1_aitemplate_cell as cell_runner
from dnn1_background_cards import main_reservations
from dnn1_owned_controller import OwnedController

ROOT = runner.ROOT
CELLS = {(m, b) for m in base.MODELS for b in (1, 2, 4, 8, 16)}


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def tvm_state(path):
    try:
        return base.read(path)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return dict(status='state_initializing', workers={}, error=repr(error))


def campaign_paths(config):
    return [Path(p) for p in config.get('tvm_epochs', [config['tvm_epoch']])]


def gpu_pool(config):
    values = config.get('gpus', [5, 7, 0, 1])
    if (not values or any(type(value) is not int for value in values) or
            len(values) != len(set(values)) or not set(values) <= {0,1,2,3,4,5,7}):
        raise ValueError('collection requires distinct cards0–5/7, neverGPU6')
    return values


def collection_cells(config):
    values = config.get('cells', [[m, b] for m in base.MODELS for b in (1, 2, 4, 8, 16)])
    cells = [tuple(value) for value in values]
    if not cells or len(cells) != len(set(cells)) or not set(cells) <= CELLS:
        raise ValueError('AIT collection requires distinct supported cells')
    return cells


def disk_snapshot(epoch, phase, model, batch):
    usage = shutil.disk_usage(ROOT)
    owned = {}
    for directory in (runner.AIT_ROOT / 'engines', epoch / 'jobs'):
        if not directory.exists():
            owned[str(directory)] = dict(allocated_bytes=0)
            continue
        try:
            result = subprocess.run(['du', '--summarize', '--block-size=1', '--', str(directory)],
                text=True, capture_output=True, timeout=30)
            owned[str(directory)] = (dict(allocated_bytes=int(result.stdout.split()[0]))
                if result.returncode == 0 else dict(error=result.stderr.strip()))
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as error:
            owned[str(directory)] = dict(error=repr(error))
    record = dict(time=time.time(), phase=phase, model=model, batch=batch,
        filesystem_total_bytes=usage.total, filesystem_free_bytes=usage.free,
        owned_caches=owned, deleted_files=False)
    with (epoch / 'disk_usage.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    return record


def resource_snapshot(epoch, gpu):
    try:
        return device.gpu(gpu)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        with (epoch / 'resource_errors.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), gpu=gpu, error=repr(error),
                disposition='skip this card; no cached-good telemetry substitution')) + '\n')
        return None


def release(config, destination):
    tvm = Path(config['tvm_epoch'])
    campaigns = campaign_paths(config)
    allowed = ('primary_complete', 'needs_diagnosis') if len(campaigns) > 1 else ('primary_complete',)
    for directory in campaigns:
        state = base.read(directory / 'state.json')
        if (state.get('status') not in allowed or state.get('workers') or state.get('active') or state.get('pending')):
            raise RuntimeError('TVM repairs remain incomplete or active')
    proof = base.read(config['cpu_prepare'])
    if (len(proof['rows']) != 25 or {(r['model'], r['batch']) for r in proof['rows']} != CELLS or
            any(r['status'] != 'ir_created' for r in proof['rows']) or
            proof['identity'] != runner.source_identity()):
        raise ValueError('CPU25 proof does not match the current uniform graph recipe/source')
    frozen = tvm / 'frozen20.csv'
    if base.sha(frozen) != config['frozen20_sha256']:
        raise ValueError('the frozen20 snapshot changed')
    if any(base.sha(directory / 'frozen20.csv') != config['frozen20_sha256'] for directory in campaigns):
        raise ValueError('TVM epochs do not share the identical frozen20 snapshot')
    primary = Path(config['primary_csv'])
    if base.sha(primary) != config['primary_sha256']:
        raise ValueError('primary125 snapshot changed')
    primary_rows = [r for r in rows(primary) if r['system'] in ('tensorrt', 'torchcompile')]
    if (len(primary_rows) != 125 or len({(r['model'], r['batch'], r['arm']) for r in primary_rows}) != 125 or
            sum(r['status'] == 'pass' for r in primary_rows) != 120 or
            sum(r['status'] == 'N/A' for r in primary_rows) != 5):
        raise ValueError('primary snapshot is not the frozen completed125')
    selected = rows(frozen) + [row for directory in campaigns for row in rows(directory / 'accepted_delta.csv')]
    if len(selected) != 25 or {(r['model'], int(r['batch'])) for r in selected} != CELLS:
        raise ValueError('exactly25 distinct accepted TVM cells are required')
    evidence = destination / 'release_evidence'
    evidence.mkdir(parents=True, exist_ok=False)
    records = []
    for row in selected:
        if row['status'] != 'pass' or row['correctness_passes'] != '5/5':
            raise ValueError('unvalidated TVM outcome cannot release AIT')
        files = [Path(row['cell_session']), *map(Path, json.loads(row['source']))]
        if len(files) != 6 or len(set(files)) != 6:
            raise ValueError('TVM release requires its session and all five receipts')
        hashes = {str(p): base.sha(p) for p in files}
        first = base.read(files[1])
        source = first.get('configuration', {}).get('tvm_diag', {}).get('cache_identity')
        source = source or first['versions']
        filename = evidence / f"{row['model']}_b{row['batch']}.json"
        base.write(filename, dict(row=row, files=hashes, source_version=source))
        records.append(dict(model=row['model'], batch=int(row['batch']), status='validated',
            path=str(filename), sha256=base.sha(filename), source_version=source))
    result = dict(schema=1, primary_cells_complete=125,
        primary_evidence=dict(path=str(primary), sha256=base.sha(primary)), cells=records,
        dispatchers=[str(p / 'state.json') for p in campaigns] + config.get('additional_dispatchers', []), graph_mode=True,
        cpu_prepare=dict(path=config['cpu_prepare'], sha256=base.sha(config['cpu_prepare'])))
    path = ROOT / 'runs/dnn1/resume_tvm/aitemplate_release.json'
    if path.exists():
        if config.get('reuse_release_sha256') != base.sha(path):
            raise FileExistsError('retain the earlier AIT release ledger; explicit hash-bound reuse required')
        publication = runner.require_tvm_complete()
        previous = base.read(path)
        def signature(records):
            return {(record['model'], record['batch']):
                dict(status=record['status'], source_version=record['source_version'],
                     files=base.read(record['path'])['files']) for record in records}
        if (previous['primary_evidence'] != result['primary_evidence'] or
                signature(previous['cells']) != signature(result['cells'])):
            raise ValueError('reused release must authorize the identical25 TVM artifacts')
        binding = destination / 'source_release_binding.json'
        base.write(binding, dict(tvm_release=publication, current_cpu_prepare=result['cpu_prepare'],
            current_AIT_identity=proof['identity'], graph_mode=True,
            retained_release_unchanged=True, verified_TVM25_evidence=records))
        return dict(publication, current_source_binding=str(binding),
                    current_source_binding_sha256=base.sha(binding))
    base.write(path, result)
    return runner.require_tvm_complete()


def enqueue(epoch, model, batch, gpu, provider='original', admission=1):
    suffix = '' if admission == 1 else f'_admission{admission}'
    folder = epoch / 'jobs' / f'{model}_b{batch}{suffix}'
    cell_runner.queue(argparse.Namespace(out=folder, model=model, batch=batch,
                                        gpu=gpu, graph=True, provider=provider))
    log = (folder / 'scheduler.log').open('w')
    process = subprocess.Popen(['bash', str(folder / 'launch.sh')], cwd=ROOT,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    base.write(folder / 'launch.json', dict(pid=process.pid, started=time.time()))
    return process, log, folder, model, batch


def available_card(epoch, gpu, *, fifo=False):
    if main_reservations(gpu) or (not fifo and base.held(f'/tmp/tilemega_gpu{gpu}.lock')):
        return False
    snapshot = resource_snapshot(epoch, gpu)
    return bool(snapshot is not None and not snapshot['owners'] and
                0 <= snapshot['used_mib'] <= 1 and snapshot['free_mib'] >= 8192)


def _dispatch(args, lifetime):
    epoch = args.epoch.resolve()
    config = base.read(epoch / 'spec.json')
    gpus = gpu_pool(config)
    lock = (epoch / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (epoch / 'started.json').exists():
        raise FileExistsError('audit the preserved controller state before resuming')
    base.write(epoch / 'started.json', dict(time=time.time(), driver_sha256=base.sha(__file__)))
    campaigns = campaign_paths(config)
    while True:
        states = [tvm_state(p / 'state.json') for p in campaigns]
        drained = all(s.get('status') in ('primary_complete', 'needs_diagnosis') and
                      not s.get('workers') and not s.get('active') and not s.get('pending') for s in states)
        if drained:
            if sum(s.get('accepted', 0) for s in states) != 5:
                base.write(epoch / 'state.json', dict(status='blocked_tvm_repair', tvm_states=states))
                return 1
            break
        base.write(epoch / 'state.json', dict(status='waiting_tvm25', time=time.time(),
            tvm_accepted=sum(s.get('accepted', 0) for s in states), tvm_states=states, gpu_workers={}))
        time.sleep(30)
    publication = release(config, epoch)
    pending = collection_cells(config)
    requested_cells = len(pending)
    active, accepted, failures, admissions = {}, [], [], {}
    reassign_after = max(45, config.get('cpu_waiter_reassign_after_s', 90))
    max_admissions = min(8, max(1, config.get('max_admissions_per_cell', 8)))
    fifo = config.get('guard_provider') == 'fifo-settle'
    while pending or active:
        limit_path = epoch / 'dispatch_limit.json'
        limit = base.read(limit_path)['max_background_cards'] if limit_path.exists() else 2
        if type(limit) is not int or limit not in (1, 2):
            raise ValueError('AIT dispatcher limit must be one or two physical cards')
        for gpu, (process, log, folder, model, batch) in list(active.items()):
            if process.poll() is None:
                continue
            log.close()
            try:
                entry = base.read(folder / 'scheduler/state.json')[f'ait-{model}-b{batch}']
                if entry['status'] != 'done':
                    raise ValueError('scheduler did not accept the whole AIT cell')
                attempt = Path(entry['out'])
                if base.read(attempt / 'guard_result.json')['code'] != 0:
                    raise ValueError('outer guard rejected the AIT cell')
                output = folder / 'collected.json'
                status = cell_runner.collect(argparse.Namespace(
                    session=attempt / 'job/ait/cell-result.json', guard=attempt / 'guard.jsonl', out=output))
                if status:
                    raise ValueError('strict AIT collector quarantined the cell')
                accepted.append(base.read(output))
                base.write(epoch / 'accepted.json', accepted)
            except Exception as error:
                failures.append(dict(model=model, batch=batch, folder=str(folder), error=repr(error),
                    policy='preserve original graph failure; no silent uncaptured fallback'))
            disk_snapshot(epoch, 'after-cell', model, batch)
            del active[gpu]
        for gpu, (process, log, folder, model, batch) in list(active.items()):
            key = (model, batch)
            if (fifo or admissions[key] >= max_admissions or
                    time.time() - base.read(folder / 'launch.json')['started'] < reassign_after):
                continue
            alternate = next((g for g in gpus if g != gpu and g not in active and
                              available_card(epoch, g)), None)
            if alternate is None or not lifetime.cancel_cpu_waiter(process, folder):
                continue
            log.close()
            del active[gpu]
            admissions[key] += 1
            active[alternate] = enqueue(epoch, model, batch, alternate,
                config.get('guard_provider', 'original'), admissions[key])
            lifetime.register(active[alternate][0])
            with (epoch / 'cpu_reassignments.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(time=time.time(), model=model, batch=batch,
                    old_gpu=gpu, new_gpu=alternate, previous_folder=str(folder),
                    next_folder=str(active[alternate][2]), admission=admissions[key],
                    scope='CPU-only; never interrupt a model payload')) + '\n')
        for gpu in gpus:
            if not pending or len(active) >= limit:
                break
            if gpu in active:
                continue
            if not available_card(epoch, gpu, fifo=fifo):
                continue
            disk = disk_snapshot(epoch, 'before-cell', *pending[0])
            if disk['filesystem_free_bytes'] < config.get('minimum_free_gib', 16) * (1 << 30):
                continue
            model, batch = pending.pop(0)
            admissions[(model, batch)] = 1
            active[gpu] = enqueue(epoch, model, batch, gpu, config.get('guard_provider', 'original'))
            lifetime.register(active[gpu][0])
        base.write(epoch / 'state.json', dict(status='collecting' if pending or active else
            ('complete' if not failures else 'needs_diagnosis'), accepted=len(accepted),
            pending=[f'{m}_b{b}' for m, b in pending],
            gpu_workers={str(g): dict(pid=p.pid, model=m, batch=b, folder=str(folder),
                admission=admissions[(m,b)]) for g,(p,_,folder,m,b) in active.items()},
            failures=failures, release=publication, graph_mode=True, priority=200,
            max_background_cards=limit, dispatch_limit_sha256=(base.sha(limit_path)
                if limit_path.exists() else None), gpus=gpus, requested_cells=requested_cells,
            cpu_waiter_reassign_after_s=reassign_after, max_admissions_per_cell=max_admissions,
            power_monitoring=False, time=time.time()))
        if pending or active:
            time.sleep(15)
    return int(bool(failures))


def dispatch(args):
    with OwnedController(args.epoch) as lifetime:
        return _dispatch(args, lifetime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=Path, required=True)
    raise SystemExit(dispatch(parser.parse_args()))
