#!/usr/bin/env python3
"""Merge audited TVM cells while retaining prior CSV records byte for byte."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('DNN1_baselines.csv', 'DNN1_tvm_diag.csv', 'DNN1_correctness.csv')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    data = path.read_bytes()
    lines = data.decode().splitlines(keepends=True)
    reader = csv.DictReader(io.StringIO(data.decode()))
    fields = reader.fieldnames
    previous = reader.line_num
    header = ''.join(lines[:previous])
    records = []
    for row in reader:
        end = reader.line_num
        records.append((row, ''.join(lines[previous:end])))
        previous = end
    return dict(path=path, data=data, fields=fields, header=header,
                records=records, tail=''.join(lines[previous:]))


def line(row, fields):
    stream = io.StringIO()
    csv.DictWriter(stream, fieldnames=fields, lineterminator='\n').writerow(
        {name: row.get(name, '') for name in fields})
    return stream.getvalue()


def same_cell(row, incoming):
    return (row['model'], row['batch'], row.get('system', 'tvm')) == (
        incoming['model'], incoming['batch'], 'tvm')


def matching(row, incoming):
    return same_cell(row, incoming) and row.get('precision') == incoming['precision']


def merge_records(old, incoming, remove_reference=False):
    if set(incoming) - set(old['fields']):
        raise ValueError('new evidence columns require an explicit schema update')
    output, found = [old['header']], False
    for row, raw in old['records']:
        if matching(row, incoming):
            if found:
                raise ValueError('duplicate destination TVM key')
            found = True
            if row['status'] == 'pass':
                # An already accepted sample is immutable; never pick a later
                # faster timing or silently substitute another build/session.
                for name in ('source', 'median_ms', 'p90_ms', 'process_range_ms',
                             'process_range_pct', 'versions', 'library_sha256',
                             'engine_or_library_sha256', 'cell_session'):
                    if name in incoming and row[name] != incoming[name]:
                        raise ValueError('accepted row differs from incoming evidence: ' + name)
                output.append(raw)
            else:
                output.append(line(incoming, old['fields']))
        elif (remove_reference and same_cell(row, incoming)
              and row.get('precision') == 'fp16' and row['status'] != 'pass'):
            continue
        else:
            output.append(raw)
    if not found:
        output.append(line(incoming, old['fields']))
    output.append(old['tail'])
    return ''.join(output).encode()


def correctness_row(incoming, diag, measurements, session, session_sha256):
    all_metrics = [metric for receipt in measurements for metric in receipt['correctness']]
    def minimum(name):
        values = [float(metric[name]) for metric in all_metrics if name in metric]
        return repr(min(values)) if values else ''
    return dict(model=incoming['model'], batch=incoming['batch'], system='tvm',
        check='baseline equivalence recipe6', protocol='P-CORR-v2: five independent original-threshold gates',
        gpu=incoming['gpu'], processes='5', modes='fixed-random-inputs', status='pass',
        cosine_min_vs_fp32=minimum('cosine_min_vs_fp32'),
        top1_agreement=minimum('top1_agreement_vs_fp32'),
        psnr_min_vs_fp32=minimum('psnr_min_vs_fp32_db'), full_p_corr_completed='True',
        library_sha256=diag['library_sha256'],
        checker_sha256=sha((ROOT/'python/tilemega/dnn/baselines.py').read_bytes()),
        receipt_sha256=session_sha256, source=str(session.relative_to(ROOT)),
        arm=incoming['arm'], precision=incoming['precision'],
        metrics_identical=incoming['correctness_metrics_identical'], correctness_passes='5',
        metrics=json.dumps(measurements[0]['correctness'], separators=(',', ':')),
        correctness_policy=incoming['correctness_policy'], strict_v1_status=incoming['strict_v1_status'])


def verified_evidence(incoming, audit):
    session = Path(incoming['cell_session'])
    identity = (incoming['model'], int(incoming['batch']), incoming['precision'], int(incoming['gpu']))
    if (audit['model'], audit['batch'], audit['precision'], audit['gpu']) != identity:
        raise ValueError('audit and strict publication identify different cells')
    if Path(audit['session']).resolve() != session.resolve():
        raise ValueError('audit identifies another cell session')
    hashes = {}
    def read_verified(path, expected):
        data = path.read_bytes()
        digest = sha(data)
        if digest != expected:
            raise ValueError('raw evidence SHA256 mismatch: ' + str(path))
        hashes[str(path)] = digest
        return data
    session_data = json.loads(read_verified(session, audit['session_sha256']))
    if (session_data['model'], session_data['batch'], session_data['gpu'], session_data['status']) != (
            identity[0], identity[1], identity[3], 'complete'):
        raise ValueError('source session is not the completed published cell')
    read_verified(session.parent.parent/'guard.jsonl', audit['outer_guard_sha256'])
    guard = session.parent.parent/'guard_result.json'
    guard_data = guard.read_bytes()
    if json.loads(guard_data)['code'] != 0:
        raise ValueError('retain original guard rejection')
    hashes[str(guard)] = sha(guard_data)
    paths = [Path(path) for path in json.loads(incoming['source'])]
    processes = {Path(process['receipt']).resolve(): process for process in audit['processes']}
    if len(paths) != 5 or len(processes) != 5 or set(processes) != {path.resolve() for path in paths}:
        raise ValueError('audit does not cover exactly the five published receipts')
    measurements = []
    for path in paths:
        process = processes[path.resolve()]
        if process['issues']:
            raise ValueError('audited process contains an interference finding')
        receipt = json.loads(read_verified(path, process['receipt_sha256']))
        read_verified(path.parent/'guard.jsonl', process['snapshots_sha256'])
        if (receipt['system'], receipt['model'], receipt['batch'], receipt['precision'],
                receipt['gpu_index'], receipt['correctness_passed']) != (
                'tvm', *identity, True):
            raise ValueError('measurement does not establish the published numerical gate')
        measurements.append(receipt)
    return session, guard, paths, measurements, hashes


def merge_one(epoch, incoming, diag, audit, source_sha256):
    if not audit['sampled_windows_clean'] or len(audit['processes']) != 5:
        raise ValueError('only five clean sampled process windows may be published')
    if incoming['status'] != 'pass' or incoming['correctness_passes'] != '5/5':
        raise ValueError('only complete five-process correctness passes may be published')
    if json.loads(incoming['versions'])['tvm_recipe_schema'] != 6:
        raise ValueError('do not mix recipe epochs')
    session, guard, paths, measurements, raw_hashes = verified_evidence(incoming, audit)
    old = {name: load(ROOT/'docs/results'/name) for name in NAMES}
    new_correctness = correctness_row(incoming, diag, measurements, session, raw_hashes[str(session)])
    prepared = {
        NAMES[0]: merge_records(old[NAMES[0]], incoming, incoming['precision'] == 'bf16'),
        NAMES[1]: merge_records(old[NAMES[1]], diag, incoming['precision'] == 'bf16'),
        NAMES[2]: merge_records(old[NAMES[2]], new_correctness),
    }
    if all(prepared[name] == value['data'] for name, value in old.items()):
        return None
    out = epoch/'publication_updates'/(
        f"{incoming['model']}_b{incoming['batch']}_{incoming['precision']}_{time.time_ns()}")
    out.mkdir(parents=True, exist_ok=False)
    (out/'merger.py').write_bytes(Path(__file__).read_bytes())
    proofs = {}
    for name, value in old.items():
        (out/(name+'.before')).write_bytes(value['data'])
        after_path = out/(name+'.after')
        after_path.write_bytes(prepared[name])
        after = load(after_path)
        protected = [(row, raw) for row, raw in value['records']
                     if row.get('system', 'tvm') != 'tvm' or row['status'] == 'pass']
        after_by_key = {}
        def record_key(row):
            return tuple(row.get(k, '') for k in ('model', 'batch', 'system', 'arm', 'precision', 'check'))
        for row, raw in after['records']:
            identity = record_key(row)
            after_by_key.setdefault(identity, []).append(raw)
        for row, raw in protected:
            identity = record_key(row)
            if raw not in after_by_key.get(identity, []):
                raise ValueError('an existing primary or accepted row changed')
        protected_keys = {record_key(row) for row, _ in protected}
        if ([raw for _, raw in protected]
                != [raw for row, raw in after['records'] if record_key(row) in protected_keys]):
            raise ValueError('existing protected records changed order or multiplicity')
        proofs[name] = dict(before_sha256=sha(value['data']), after_sha256=sha(prepared[name]),
            before_rows=len(value['records']), after_rows=len(after['records']),
            protected_rows_byte_identical=len(protected),
            protected_records_sha256=sha(''.join(raw for _, raw in protected).encode()))
        if name == NAMES[0]:
            primary = [raw for row, raw in value['records'] if row['system'] != 'tvm']
            prior = [raw for row, raw in value['records'] if row['system'] == 'tvm' and row['status'] == 'pass']
            if len(primary) != 125:
                raise ValueError('coordinate before changing the 125-row primary table')
            proofs[name].update(primary_rows_byte_identical=125, prior_tvm_rows_byte_identical=len(prior),
                primary_sha256=sha(''.join(primary).encode()), prior_tvm_sha256=sha(''.join(prior).encode()))
    if any(value['path'].read_bytes() != value['data'] for value in old.values()):
        raise RuntimeError('concurrent CSV edit; retain snapshots and retry from current data')
    if any(sha(Path(path).read_bytes()) != digest for path, digest in raw_hashes.items()):
        raise RuntimeError('raw evidence changed during publication; no main CSV was written')
    for name, value in old.items():
        temporary = value['path'].with_suffix('.tvm-new')
        temporary.write_bytes(prepared[name])
        temporary.replace(value['path'])
    receipt = dict(evidence='verified', time=time.time(), model=incoming['model'],
        batch=int(incoming['batch']), precision=incoming['precision'],
        merger_sha256=sha(Path(__file__).read_bytes()), strict_publication_sha256=source_sha256,
        original_guard_sha256=raw_hashes[str(guard)], session_sha256=raw_hashes[str(session)],
        measurement_sha256={str(path): raw_hashes[str(path)] for path in paths},
        verified_raw_sha256=raw_hashes,
        sampled_window_audit=audit, csv=proofs, median_ms=incoming['median_ms'],
        p90_ms=incoming['p90_ms'], process_range_pct=incoming['process_range_pct'],
        scope='incremental accepted cell only; existing primary/accepted records unchanged; no report edit or git commit')
    receipt_path = out/'receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    return str(receipt_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    epoch = args.epoch.resolve()
    lock = (epoch/'incremental_publish.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            source = epoch/'published/DNN1_performance_verified.csv'
            if source.exists():
                loaded = load(source)
                audits = json.loads((epoch/'published/PID_windows.json').read_text())['cells']
                for incoming, _ in loaded['records']:
                    if incoming['system'] != 'tvm' or incoming['status'] != 'pass':
                        continue
                    matching_audits = [cell for cell in audits if cell['session'] == incoming['cell_session']
                                       and cell['precision'] == incoming['precision']]
                    if not matching_audits or not matching_audits[-1]['sampled_windows_clean']:
                        raise ValueError('strict publication and audit disagree')
                    # A cached remeasurement may live under remeasureNNN;
                    # its row's session identifies the matching collector.
                    session = Path(incoming['cell_session'])
                    dispatch = next(parent for parent in session.parents if parent.name == 'dispatch')
                    diag = next(row for row, _ in load(dispatch/'collection/DNN1_tvm_diag.csv')['records']
                                if matching(row, incoming))
                    result = merge_one(epoch, incoming, diag, matching_audits[-1], sha(loaded['data']))
                    if result:
                        print(json.dumps(dict(published=result)), flush=True)
            state = json.loads((epoch/'publication_state.json').read_text())
            if not args.watch or state['status'] == 'complete':
                return
        except Exception:
            (epoch/'incremental_publish_error.json').write_text(json.dumps(
                dict(time=time.time(), traceback=traceback.format_exc()), indent=2)+'\n')
            if not args.watch:
                raise
        time.sleep(30)


if __name__ == '__main__':
    main()
