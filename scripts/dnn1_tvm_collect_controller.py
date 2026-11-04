#!/usr/bin/env python3
"""Publish audited TVM cells and remeasure quarantined cells after tuning drains."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import dnn1_baselines as baseline
import dnn1_tvm_audit as audit
import dnn1_tvm_publish as publisher

TERMINAL = {'primary_complete', 'needs_diagnosis', 'stopped', 'dispatcher_error'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def key(row):
    return tuple(row[name] for name in ('model', 'batch', 'system', 'arm', 'precision'))


def reconcile(epoch, roots, cache):
    selected, clean, bad = {}, set(), set()
    fields, evidence = [], []
    for root in roots:
        source = root / 'dispatch/collection/DNN1_baselines.csv'
        if not source.exists():
            continue
        digest = sha(source)
        output = root / 'audited'
        if cache.get(str(source)) != digest:
            publisher.publish(source, output / 'enriched.csv')
            audit.audit(output / 'enriched.csv', output / 'PID_windows.json')
            cache[str(source)] = digest
        report = read(output / 'PID_windows.json')
        verdict = {(cell['session'], cell['precision']): cell for cell in report['cells']}
        with (output / 'enriched.csv').open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields += [name for name in reader.fieldnames if name not in fields]
            rows = list(reader)
        for row in rows:
            identity = key(row)
            if row['system'] == 'tvm' and row['status'] == 'pass':
                cell = verdict[(row['cell_session'], row['precision'])]
                evidence.append(cell)
                if cell['sampled_windows_clean']:
                    clean.add(identity)
                    selected[identity] = row
                else:
                    bad.add(identity)
                    if identity not in clean:
                        masked = dict(row, status='quarantined_interference',
                            reason='sampled measurement PID window failed audit; queued for cached remeasurement')
                        for name in ('median_ms', 'p90_ms', 'process_range_ms', 'process_range_pct'):
                            masked[name] = ''
                        selected[identity] = masked
            elif identity not in clean:
                selected[identity] = row
    bf16_cells = {(identity[0], identity[1]) for identity in clean if identity[4] == 'bf16'}
    obsolete_fp16 = {identity for identity in selected if identity[4] == 'fp16'
                     and (identity[0], identity[1]) in bf16_cells}
    for identity in obsolete_fp16:
        selected.pop(identity)
    dirty = bad - clean - obsolete_fp16
    pending = sorted({(identity[0], int(identity[1])) for identity in dirty})
    if fields:
        rows = sorted(selected.values(), key=key)
        write_csv(epoch / 'published/DNN1_baselines.csv', fields, rows)
        write_csv(epoch / 'published/DNN1_performance_verified.csv', fields,
                  [row for row in rows if row['status'] != 'quarantined_interference'])
        write_csv(epoch / 'published/DNN1_quarantined.csv', fields,
                  [row for row in rows if row['status'] == 'quarantined_interference'])
    baseline.write(epoch / 'published/PID_windows.json', dict(evidence='verified', cells=evidence,
        scope='sampled PID windows only; continuous exclusivity and host/container identity remain unproven'))
    return selected, clean, pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    epoch = args.epoch.resolve()
    lock = (epoch / 'collect_controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = read(epoch / 'freeze.json')
    if frozen['recipe']['schema'] != 6:
        raise ValueError('this controller requires the uniformly frozen recipe6 epoch')
    cache = {}
    while True:
        try:
            roots = [epoch] + sorted(epoch.glob('remeasure[0-9][0-9][0-9]'))
            rows, clean, pending = reconcile(epoch, roots, cache)
            active = roots[-1]
            state_path = active / 'dispatch/state.json'
            state = read(state_path) if state_path.exists() else {'status': 'starting', 'accepted': [], 'workers': {}}
            failed_rows = [dict(model=row['model'], batch=int(row['batch']), arm=row['arm'],
                               status=row['status'], reason=row['reason'])
                           for row in rows.values() if row['status'] not in ('pass', 'quarantined_interference')]
            report = dict(time=time.time(), controller_sha256=sha(__file__),
                epochs=[str(root) for root in roots], dispatcher_status=state['status'],
                active_epoch=str(active), accepted_by_dispatchers=sum(
                    len(read(root/'dispatch/state.json').get('accepted', [])) for root in roots
                    if (root/'dispatch/state.json').exists()),
                clean_performance_cells=len({(k[0], k[1]) for k in clean}),
                pending_remeasure=[dict(model=model, batch=batch) for model, batch in pending],
                unresolved_rows=failed_rows, status='collecting',
                scope='original collector retained; quarantined rows require fresh cached sessions; at most one max3 dispatcher active')
            if state['status'] == 'primary_complete' and pending:
                report['status'] = 'remeasure_queued'
                if not args.once:
                    hashes = {str(baseline.ROOT / path): digest
                              for path, digest in frozen['source_sha256'].items()}
                    hashes.update(frozen['tvm_source_identity']['modified_sources'])
                    hashes.update(frozen['tvm_source_identity']['patches'])
                    hashes.update({item['path']: item['sha256'] for item in frozen['libraries']})
                    changed = [path for path, digest in hashes.items() if sha(path) != digest]
                    if changed:
                        raise RuntimeError('frozen recipe changed before cached remeasurement: ' + repr(changed))
                    retry = epoch / f'remeasure{len(roots):03d}'
                    retry.mkdir(exist_ok=False)
                    config = read(epoch / 'base-spec.json')
                    config['cells'] = report['pending_remeasure']
                    baseline.write(retry / 'base-spec.json', config)
                    baseline.write(retry / 'trigger.json', report)
                    command = [sys.executable, str(baseline.ROOT/'scripts/dnn1_dispatch.py'),
                        '--base-spec', str(retry/'base-spec.json'), '--out', str(retry/'dispatch'),
                        '--gpus', '0,1,2,3,4,5,7', '--max-parallel', '3', '--poll-seconds', '15']
                    with (retry/'dispatcher.log').open('w') as log:
                        process = subprocess.Popen(command, cwd=baseline.ROOT, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    baseline.write(retry/'launch.json', dict(command=command, pid=process.pid,
                        started=time.time(), cache_policy='same recipe6 identity; completed models reused, no retuning'))
            elif state['status'] == 'primary_complete':
                expected = {(model, str(batch)) for model in baseline.MODELS for batch in baseline.BATCHES}
                covered = {(identity[0], identity[1]) for identity in clean}
                report['status'] = 'complete' if expected <= covered and not failed_rows else 'needs_diagnosis'
                report['missing_cells'] = sorted(expected - covered)
            elif state['status'] in TERMINAL:
                report['status'] = 'needs_diagnosis'
            baseline.write(epoch/'publication_state.json', report)
            if args.once or report['status'] == 'complete':
                return
        except Exception:
            baseline.write(epoch/'publication_error.json', dict(time=time.time(), traceback=traceback.format_exc()))
            if args.once:
                raise
        time.sleep(30)


if __name__ == '__main__':
    main()
