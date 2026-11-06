#!/usr/bin/env python3
"""Independently finalize retained P5 cells after exact device-annotation proof."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

import dnn1_baselines as base
import dnn1_tvm_resume_collect as original
import dnn1_receipt_integrity as integrity

ROOT = original.ROOT


def collect(cell, attempt, destination):
    audit = original.audit_windows
    def verified_window(process, receipt, outer):
        proof = integrity.verify(process, attempt / 'job/physical-device-identity.json',
            ROOT / 'scripts/run_r16_with_device_identity.py', cell['gpu'])
        mapped = dict(process, receipt_sha256=proof['current_sha256'])
        result = audit(mapped, receipt, outer)
        result['receipt_integrity'] = proof
        return result
    original.audit_windows = verified_window
    try:
        row = original.collect_cell(cell, attempt, destination)
    finally:
        original.audit_windows = audit
    base.write(destination / 'finalizer.json', dict(
        protocol='unchanged original collector/guard/P5/numerical/VM/source gates; only exact official annotation proof added',
        finalizer_sha256=base.sha(__file__), integrity_sha256=base.sha(integrity.__file__),
        original_collector_sha256=base.sha(original.__file__),
        original_session=str(attempt / 'job/cell-result.json'),
        original_session_sha256=base.sha(attempt / 'job/cell-result.json')))
    return row


def dispatch(args):
    epoch, upstream = args.epoch.resolve(), args.upstream.resolve()
    epoch.mkdir(parents=True, exist_ok=True)
    lock = (epoch / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (epoch / 'started.json').exists():
        raise FileExistsError('preserve and audit the original finalizer before resuming')
    base.write(epoch / 'started.json', dict(time=time.time(), upstream=str(upstream),
        finalizer_sha256=base.sha(__file__), integrity_sha256=base.sha(integrity.__file__)))
    for filename in ('frozen20.csv', 'frozen20.sha256'):
        shutil.copy2(upstream / filename, epoch / filename)
    spec = base.read(upstream / 'spec.json')
    accepted, failures = {}, {}
    while True:
        try:
            snapshot = (upstream / 'state.json').read_bytes()
            state = json.loads(snapshot)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(.25)
            continue
        for cell_spec in spec['cells']:
            key = f"{cell_spec['model']}_b{cell_spec['batch']}"
            if key in accepted or key in failures:
                continue
            folder = upstream / 'jobs' / key
            state_path = folder / 'scheduler/state.json'
            if not state_path.exists():
                continue
            try:
                step = base.read(state_path).get(key, {})
            except json.JSONDecodeError:
                continue
            if step.get('status') != 'done':
                continue
            try:
                cell = base.read(folder / 'cell.json')
                accepted[key] = collect(cell, Path(step['out']), epoch / 'jobs' / key / 'collection')
                original.csv_write(epoch / 'accepted_delta.csv', list(accepted.values()))
            except Exception as error:
                failures[key] = dict(error=repr(error), folder=str(folder))
        drained = not state.get('workers') and not state.get('active') and not state.get('pending')
        if drained:
            status = 'primary_complete' if len(accepted) == len(spec['cells']) and not failures else 'needs_diagnosis'
        else:
            status = 'collecting'
        base.write(epoch / 'state.json', dict(status=status, accepted=len(accepted),
            workers=state.get('workers', {}), active=state.get('active', {}), pending=state.get('pending', []),
            failures=failures, upstream_dispatcher=str(upstream / 'state.json'),
            upstream_dispatcher_sha256=hashlib.sha256(snapshot).hexdigest(),
            upstream_collector_rejections=state.get('failures', []),
            scope='independent finalization; retained measurements/session/collector/provider never rewritten',
            finalizer_sha256=base.sha(__file__), integrity_sha256=base.sha(integrity.__file__), time=time.time()))
        if drained:
            return int(status != 'primary_complete')
        time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--epoch', type=Path, required=True)
    raise SystemExit(dispatch(parser.parse_args()))
