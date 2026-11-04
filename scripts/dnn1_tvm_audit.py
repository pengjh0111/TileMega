#!/usr/bin/env python3
"""Audit recorded TVM measurement PID windows without changing acceptance."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(source, destination):
    cells = []
    for row in csv.DictReader(source.open(newline='')):
        if row['system'] != 'tvm' or row['status'] != 'pass':
            continue
        session_path = Path(row['cell_session'])
        session = json.loads(session_path.read_text())
        outer_path = session_path.parent.parent / 'guard.jsonl'
        outer = [json.loads(line) for line in outer_path.read_text().splitlines()]
        receipts = {str(Path(path).resolve()) for path in json.loads(row['source'])}
        processes = []
        for job in session['logical_jobs']:
            for attempt in job['attempts']:
                measurement = attempt.get('measurement')
                if not measurement or str(Path(measurement).resolve()) not in receipts:
                    continue
                snapshots_path = Path(attempt['folder']) / 'guard.jsonl'
                snapshots = [json.loads(line) for line in snapshots_path.read_text().splitlines()]
                starts = [item for item in snapshots if item['label'] == 'before-measurement']
                ends = [item for item in snapshots if item['label'] == 'after-measurement']
                issues = []
                window = []
                if len(starts) != 1 or len(ends) != 1:
                    issues.append('missing or duplicate measurement boundary')
                else:
                    start, end = starts[0], ends[0]
                    owners = set(map(str, start['pids']))
                    if len(owners) != 1 or owners != set(map(str, end['pids'])):
                        issues.append('measurement boundary PID set is not one stable PID')
                    window = [item for item in outer
                              if start['time'] <= item['time'] <= end['time']]
                    if any(set(item.get('owners', {})) - owners for item in window):
                        issues.append('outer sample contains another PID during measurement')
                for boundary in ('process_before', 'process_after'):
                    if attempt[boundary]['owners']:
                        issues.append(boundary + ' contains GPU owner')
                processes.append(dict(role=job['role'], index=job['index'],
                    pid_namespace='container and host PIDs are not treated as interchangeable',
                    process_pid=attempt['pid'], started=attempt['started'], ended=attempt['ended'],
                    process_before=attempt['process_before'], process_after=attempt['process_after'],
                    before_measurement=starts, after_measurement=ends,
                    outer_measurement_samples=window, issues=issues,
                    receipt=measurement, receipt_sha256=sha(measurement),
                    snapshots_sha256=sha(snapshots_path)))
        if len(processes) != 5:
            raise ValueError('accepted TVM row does not map to five recorded process windows')
        cells.append(dict(model=row['model'], batch=int(row['batch']), precision=row['precision'],
            gpu=int(row['gpu']), session=str(session_path), session_sha256=sha(session_path),
            outer_guard_sha256=sha(outer_path), processes=processes,
            sampled_windows_clean=all(not item['issues'] for item in processes)))
    report = dict(evidence='verified', source=str(source.resolve()), source_sha256=sha(source),
        auditor_sha256=sha(__file__), cells=cells,
        scope='read-only sampled PID-window audit; does not change collector acceptance or prove continuous exclusivity or host/container PID identity')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verified-out', type=Path)
    parser.add_argument('--quarantine-out', type=Path)
    args = parser.parse_args()
    report = audit(args.input, args.out)
    if bool(args.verified_out) != bool(args.quarantine_out):
        parser.error('provide both verified and quarantine outputs')
    if args.verified_out:
        verdict = {(cell['session'], cell['precision']): cell['sampled_windows_clean']
                   for cell in report['cells']}
        with args.input.open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        clean, quarantine = [], []
        for row in rows:
            selected = quarantine if (row['system'] == 'tvm' and row['status'] == 'pass'
                and not verdict[(row['cell_session'], row['precision'])]) else clean
            selected.append(row)
        for path, selected in ((args.verified_out, clean), (args.quarantine_out, quarantine)):
            if path.resolve() == args.input.resolve():
                raise ValueError('retain the original collector or publisher input')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
                writer.writeheader()
                writer.writerows(selected)
    print(json.dumps(dict(cells=len(report['cells']),
                          sampled_windows_clean=sum(c['sampled_windows_clean'] for c in report['cells']))))


if __name__ == '__main__':
    main()
