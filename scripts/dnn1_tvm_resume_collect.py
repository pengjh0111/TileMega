#!/usr/bin/env python3
"""Low-priority guarded TVM resumption; each cell keeps independent provenance."""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import dnn1_baselines as base
import dnn1_device as device
import dnn1_tvm_publish as publisher
import dnn1_tvm_audit as auditor
from dnn1_aitemplate_cell import AFFINITY, audit_windows
from dnn1_dispatch import affinities
from dnn1_background_cards import main_reservations
from dnn1_baseline_boundaries import idle_boundary, final_boundary
from dnn1_tvm_resume_sample import prior_identity

ROOT = Path(__file__).resolve().parents[1]


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=base.FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run_cell(args):
    cell = base.read(args.cell)
    base.require_session(cell)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'cell-result.json').exists():
        raise FileExistsError('retain the original cell attempt')
    session = dict(schema=1, status='running', model=cell['model'], batch=cell['batch'],
        gpu=cell['gpu'], logical_jobs=[], started=time.time(), cell_spec=str(args.cell),
        cell_spec_sha256=base.sha(args.cell), p_cell_complete=False)
    for index in range(5):
        for filename, digest in cell.get('protocol_sources', {}).items():
            if base.sha(filename) != digest:
                raise ValueError('prepared background protocol source changed: ' + filename)
        role = 'measurement' if index < 3 else 'correctness'
        position = index if index < 3 else index - 3
        folder = args.out / 'measurements' / f'{role}-{position}' / 'tvm-bf16/attempt1'
        payload = folder / 'payload'
        before_drain = idle_boundary(cell)
        if before_drain['status'] != 'empty_observed':
            session.update(status='infrastructure_rejection', error='nonempty pre-process GPU',
                           boundary=before_drain, ended=time.time())
            base.write(args.out / 'cell-result.json', session)
            return 75
        before = final_boundary(before_drain)
        command = [cell['python'], str(ROOT / 'scripts/dnn1_tvm_resume_sample.py'),
            '--cell', str(args.cell), '--out', str(payload), '--round-index', str(index)]
        if role == 'correctness':
            command += ['--correctness-only']
        result = base.execute(command, folder, os.environ.copy(), 5400)
        drain = idle_boundary(cell)
        after = (drain['samples'][-1]['telemetry'] if drain['samples'] else
                 dict(owners={}, used_mib=float('inf')))
        receipt = payload / 'measurements.json'
        result.update(folder=str(payload), process_before=before, process_after=after,
            before_drain=before_drain, drain=drain, measurement=str(receipt) if receipt.exists() else None,
            receipt_sha256=base.sha(receipt) if receipt.exists() else None,
            status='pass' if result['returncode'] == 0 else 'fail')
        if receipt.exists():
            value = base.read(receipt)
            snapshots = [json.loads(s) for s in (payload / 'guard.jsonl').read_text().splitlines()]
            for label, key in [('before-measurement', 'clock_start'), ('after-measurement', 'clock_end')]:
                sample = next(s for s in snapshots if s['label'] == label)
                result[key] = dict(sm_clock_mhz=float(sample['telemetry']['clocks.sm']), sample=sample)
            inputs = value['configuration']['inputs_sha256']
            if session.get('inputs_sha256', inputs) != inputs:
                raise ValueError('fixed input identity changed between fresh processes')
            session['inputs_sha256'] = inputs
        elif (payload / 'failure.json').exists():
            result['failure'] = base.read(payload / 'failure.json')
        job = dict(arm=dict(arm='tvm-bf16', system='tvm', precision='bf16'), role=role,
            index=position, status=result['status'], attempts=[result], final=result)
        session['logical_jobs'].append(job)
        base.write(args.out / 'cell-result.json', session)
        if (result['returncode'] or after['owners'] or after['used_mib'] > 1 or
                drain['status'] != 'empty_observed'):
            session.update(status='failed', ended=time.time())
            base.write(args.out / 'cell-result.json', session)
            return result['returncode'] or 75
    session.update(status='complete', ended=time.time())
    base.write(args.out / 'cell-result.json', session)
    return 0


def collect_cell(cell, attempt, destination):
    session_path = attempt / 'job/cell-result.json'
    session = base.read(session_path)
    guard = base.read(attempt / 'guard_result.json')
    if session['status'] != 'complete' or guard.get('code') != 0:
        raise ValueError('whole cell requires complete P5 and accepted outer guard')
    outer = [json.loads(s) for s in (attempt / 'guard.jsonl').read_text().splitlines()]
    jobs = session['logical_jobs']
    if len(jobs) != 5 or len({j['final']['pid'] for j in jobs}) != 5:
        raise ValueError('exactly five distinct fresh processes required')
    if ([j['role'] for j in jobs].count('measurement') != 3 or
            [j['role'] for j in jobs].count('correctness') != 2):
        raise ValueError('P-DNN needs three timing and two extra correctness processes')
    identities, libraries, windows = [], [], []
    for job in jobs:
        process = job['final']
        receipt = base.read(process['measurement'])
        if (any(p['owners'] or not 0 <= p['used_mib'] <= 1 for p in
                (process['process_before'], process['process_after'])) or
                process['drain']['status'] != 'empty_observed'):
            raise ValueError('dirty process/memory boundary or undrained context')
        if (process['process_before'] != final_boundary(process['before_drain']) or
                process['process_after'] != final_boundary(process['drain'])):
            raise ValueError('process boundary differs from the final two idle observations')
        if (receipt['system'] != 'tvm' or receipt['model'] != cell['model'] or
                receipt['batch'] != cell['batch'] or receipt['gpu_index'] != cell['gpu'] or
                receipt['precision'] != 'bf16' or receipt['pid'] != process['pid']):
            raise ValueError('row/process/cell/GPU identity differs')
        if receipt['process_role'] != job['role']:
            raise ValueError('fresh-process role differs from its recorded job')
        mapped = dict(process, receipt=process['measurement'],
                      before=process['process_before'], after=process['process_after'])
        audit = audit_windows(mapped, receipt, outer)
        if audit['issues']:
            raise ValueError('dirty PID windows: ' + repr(audit['issues']))
        windows.append(audit)
        diag = receipt['configuration']['tvm_diag']
        identities.append(base.compact(diag['cache_identity']))
        libraries.append(diag['model_sha256'])
        if receipt['process_role'] == 'measurement' and (
                len(receipt['samples_ms']) != 200 or receipt['measurement']['warmup'] != 50):
            raise ValueError('P-DNN requires50 warmup/200 event samples')
    if len(set(identities)) != 1 or len(set(libraries)) != 1:
        raise ValueError('fresh processes used different source/VM/database identities')
    session['source'] = str(session_path)
    row = base.aggregate_arm(cell, session, 'tvm-bf16', jobs, True)
    if row['status'] != 'pass':
        raise ValueError(row['reason'])
    csv_write(destination / 'raw.csv', [row])
    publisher.publish(destination / 'raw.csv', destination / 'enriched.csv')
    audit = auditor.audit(destination / 'enriched.csv', destination / 'PID_windows.json')
    if not all(c['sampled_windows_clean'] for c in audit['cells']):
        raise ValueError('independent PID auditor rejected the row')
    with (destination / 'enriched.csv').open() as stream:
        final = next(csv.DictReader(stream))
    base.write(destination / 'accepted.json', dict(status='pass', row=final, windows=windows,
        source_session=str(session_path), session_sha256=base.sha(session_path)))
    return final


def enqueue(cell, epoch, gpu, cpus):
    key = f"{cell['model']}_b{cell['batch']}"
    folder = epoch / 'jobs' / key
    folder.mkdir(parents=True, exist_ok=False)
    cell = dict(cell, gpu=gpu, cpu_affinity=cpus, parallel=True,
        python=str(ROOT / 'scripts/dnn1_background_python.py'),
        correctness_policy='five-independent-gates-v2', prefix=dict(available=False))
    cell['boundary_protocol'] = 'paired-process-memory-v2'
    cell['protocol_sources'] = {str(ROOT / 'scripts' / name): base.sha(ROOT / 'scripts' / name)
        for name in ('dnn1_tvm_resume_collect.py', 'dnn1_baseline_boundaries.py',
                     'dnn1_background_python.py', 'dnn1_background_cards.py',
                     'dnn1_idle_floor_python.py', 'dnn1_baselines.py')}
    cell['prior_files'] = prior_identity(cell)['files']
    base.write(folder / 'cell.json', cell)
    env = base.gpu_env(gpu, dict(TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
        TILEMEGA_DNN1_TVM_ROOT=cell['tvm_root'], OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4'))
    uuid, pci = [s.strip() for s in subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader,nounits'], text=True).strip().split(',')]
    domain, bus, slot = pci.split(':')
    pci = f'{int(domain,16):04x}:{bus}:{slot}'
    python = cell['python']
    command = ['numactl', '--physcpubind=' + ','.join(map(str, cpus)), python,
        str(ROOT / 'scripts/run_r16_with_device_identity.py'), '--guard-policy', str(base.POLICY),
        '--out', '{attempt_dir}/job', '--expected-uuid', uuid, '--expected-pci', pci,
        '--numa-node', str(0 if gpu < 4 else 1), '--allow-no-measurements', '--',
        python, str(Path(__file__).resolve()), 'cell', '--cell', str(folder / 'cell.json'),
        '--out', '{attempt_dir}/job']
    base.write(folder / 'queue/queue_tvm_resume.json', [dict(name=key, gpu=True, timing=True,
        priority=200, timeout_s=28800, needs_free_mib=8192, cwd=str(ROOT), after=[], env=env, command=command)])
    launcher = ['flock', f'/tmp/tilemega_dnn1_baseline_scheduler_gpu{gpu}.lock',
        'env', '-u', 'TILEMEGA_DEVICE_GROUP', '-u', 'TILEMEGA_GPU_LOCK_HELD', '-u', 'TILEMEGA_RUN_NONCE',
        *[f'{k}={v}' for k,v in env.items()], python, '-m', 'tilemega.bench.scheduler',
        '--queue-dir', str(folder / 'queue'), '--out', str(folder / 'scheduler'),
        '--policy', str(base.POLICY), '--deadline-hours', '96']
    log = (folder / 'scheduler.log').open('w')
    process = subprocess.Popen(launcher, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    base.write(folder / 'launch.json', dict(pid=process.pid, command=launcher, started=time.time()))
    return process, log, folder, cell


def dispatch(args):
    epoch = args.epoch.resolve()
    config = base.read(epoch / 'spec.json')
    gpus = [int(s) for s in args.gpus.split(',')]
    if not set(gpus) <= {0,1,5,7} or not 1 <= args.max_parallel <= 2:
        raise ValueError('background resumption uses at most2 healthy cards, neverGPU6')
    binding = affinities(gpus, 8, explicit={str(g): AFFINITY[g] for g in gpus})
    lock = (epoch / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending, active, accepted, failures = list(config['cells']), {}, [], []
    while pending or active:
        for gpu, (process, log, folder, cell) in list(active.items()):
            if process.poll() is None:
                continue
            log.close()
            key = f"{cell['model']}_b{cell['batch']}"
            try:
                state = base.read(folder / 'scheduler/state.json')[key]
                if state['status'] != 'done':
                    raise ValueError('scheduler did not accept the whole cell')
                row = collect_cell(cell, Path(state['out']), folder / 'collection')
                accepted.append(row)
                csv_write(epoch / 'accepted_delta.csv', accepted)
            except Exception as error:
                failures.append(dict(key=key, folder=str(folder), error=str(error)))
            del active[gpu]
        for gpu in gpus:
            if not pending or len(active) >= args.max_parallel:
                break
            if gpu in active:
                continue
            if main_reservations(gpu):
                continue
            row = device.gpu(gpu)
            if (row['owners'] or row['used_mib'] > 1 or row['free_mib'] < 8192 or
                    base.held(f'/tmp/tilemega_gpu{gpu}.lock')):
                continue
            cell = pending[0]
            if cell.get('cpu_ready') and not Path(cell['cpu_ready']).exists():
                continue
            active[gpu] = enqueue(pending.pop(0), epoch, gpu, binding[gpu])
        status = ('primary_complete' if len(accepted) == len(config['cells']) and not failures else
                  'needs_diagnosis') if not pending and not active else 'collecting'
        workers = {str(g):dict(pid=p.pid,key=f"{c['model']}_b{c['batch']}")
                   for g,(p,_,_,c) in active.items()}
        base.write(epoch / 'state.json', dict(status=status,
            pending=[f"{c['model']}_b{c['batch']}" for c in pending],
            active=workers, workers=workers,
            accepted=len(accepted), failures=failures, max_background_cards=args.max_parallel,
            gpus=gpus, borrowed_main_cards=[g for g in active if g in (0,1)],
            priority=200, power_monitoring=False, time=time.time()))
        if pending or active:
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('dispatch'); p.add_argument('--epoch', type=Path, required=True)
    p.add_argument('--gpus', default='5,7'); p.add_argument('--max-parallel', type=int, default=2)
    p = sub.add_parser('cell'); p.add_argument('--cell', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    return dispatch(args) if args.command == 'dispatch' else run_cell(args)


if __name__ == '__main__':
    raise SystemExit(main())
