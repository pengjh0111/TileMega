#!/usr/bin/env python3
"""Release AIT only after source-bound TVM25, then dispatch guarded P5 cells."""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import time

import dnn1_baselines as base
import dnn1_device as device
import dnn1_aitemplate as runner
import dnn1_aitemplate_cell as cell_runner
from dnn1_background_cards import main_reservations

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
        raise FileExistsError('retain the earlier AIT release ledger')
    base.write(path, result)
    return runner.require_tvm_complete()


def enqueue(epoch, model, batch, gpu):
    folder = epoch / 'jobs' / f'{model}_b{batch}'
    cell_runner.queue(argparse.Namespace(out=folder, model=model, batch=batch, gpu=gpu, graph=True))
    log = (folder / 'scheduler.log').open('w')
    process = subprocess.Popen(['bash', str(folder / 'launch.sh')], cwd=ROOT,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    base.write(folder / 'launch.json', dict(pid=process.pid, started=time.time()))
    return process, log, folder, model, batch


def dispatch(args):
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
    pending = [(m, b) for m in base.MODELS for b in (1, 2, 4, 8, 16)]
    active, accepted, failures = {}, [], []
    while pending or active:
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
            del active[gpu]
        for gpu in gpus:
            if not pending or len(active) >= 2:
                break
            if gpu in active:
                continue
            if main_reservations(gpu):
                continue
            snapshot = device.gpu(gpu)
            if (snapshot['owners'] or snapshot['used_mib'] > 1 or snapshot['free_mib'] < 8192 or
                    base.held(f'/tmp/tilemega_gpu{gpu}.lock')):
                continue
            model, batch = pending.pop(0)
            active[gpu] = enqueue(epoch, model, batch, gpu)
        base.write(epoch / 'state.json', dict(status='collecting' if pending or active else
            ('complete' if not failures else 'needs_diagnosis'), accepted=len(accepted),
            pending=[f'{m}_b{b}' for m, b in pending],
            gpu_workers={str(g): dict(pid=p.pid, model=m, batch=b) for g,(p,_,_,m,b) in active.items()},
            failures=failures, release=publication, graph_mode=True, priority=200,
            max_background_cards=2, gpus=gpus, power_monitoring=False, time=time.time()))
        if pending or active:
            time.sleep(15)
    return int(bool(failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=Path, required=True)
    raise SystemExit(dispatch(parser.parse_args()))
