#!/usr/bin/env python3
"""Join a campaign's staged test plans, Slurm accounting and raw result paths."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess


FIELDS = ('model', 'batch', 'test_type', 'label', 'source_commit', 'artifact_sha256',
          'plan_sha256', 'job_id', 'status', 'elapsed', 'exit_code', 'result_path', 'updated_at')


def plans(path):
    if not path or not path.is_file():
        return [], ''
    payload = path.read_bytes()
    text = payload.decode().lstrip()
    decoder = json.JSONDecoder()
    records = []
    while text:
        value, end = decoder.raw_decode(text)
        records.append(value)
        text = text[end:].lstrip()
    return records, hashlib.sha256(payload).hexdigest()


def accounting(ids):
    if not ids:
        return {}
    report = subprocess.run(
        ['sacct', '-XnP', '-j', ','.join(sorted(ids)), '-o', 'JobID,State,Elapsed,ExitCode'],
        check=True, capture_output=True, text=True, timeout=20)
    result = {}
    for line in report.stdout.splitlines():
        job, status, elapsed, code = line.split('|')
        owner = job.split('_', 1)[0]
        if owner in ids:
            result.setdefault(owner, []).append((job, status, elapsed, code))
    return result


def collect(entries, workspace):
    ids = set()
    expanded = []
    for entry in entries:
        ticket = workspace/entry['ticket']
        job = ticket.read_text().strip() if ticket.is_file() else ''
        if job and not re.fullmatch(r'[0-9]+', job):
            raise ValueError(f'invalid Slurm job in {ticket}')
        plan_path = workspace/entry['plan'] if entry.get('plan') else None
        records, plan_sha = plans(plan_path)
        if any(record.get('model', entry['model']) != entry['model'] or
               record.get('batch', entry['batch']) != entry['batch'] for record in records):
            raise ValueError(f'mismatched model/batch at {plan_path}')
        sources = {r.get('source_sha') for r in records} - {None}
        artifacts = {r.get(field) for r in records
                     for field in ('decode_sha256', 'binary_sha256', 'prepared_shortlist_sha256')}
        artifacts.discard(None)
        if not sources and entry.get('repo') and (workspace/entry['repo']/'.git').exists():
            sources.add(subprocess.check_output(
                ['git', '-C', str(workspace/entry['repo']), 'rev-parse', 'HEAD'],
                text=True, timeout=5).strip())
        expanded.append((entry, job, plan_sha, sources, artifacts))
        if job:
            ids.add(job)

    states = accounting(ids)
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    for entry, job, plan_sha, sources, artifacts in expanded:
        status = 'submitted' if job else entry.get('before_submission', 'planned')
        if entry.get('hold') and (workspace/entry['hold']).is_file() and not job:
            status = 'retained_control'
        allocations = states.get(job, [(job, status, '', '')])
        for allocation, state, elapsed, code in allocations:
            yield dict(model=entry['model'], batch=entry['batch'], test_type=entry['test_type'],
                       label=entry['label'], source_commit=' '.join(sorted(sources)),
                       artifact_sha256=' '.join(sorted(artifacts)), plan_sha256=plan_sha,
                       job_id=allocation, status=state, elapsed=elapsed, exit_code=code,
                       result_path=entry['result_path'].format(job=allocation.split('_', 1)[0]),
                       updated_at=now)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())
    rows = list(collect(entries, args.workspace))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.incoming')
    with temporary.open('w', newline='') as file:
        output = csv.DictWriter(file, fieldnames=FIELDS, lineterminator='\n')
        output.writeheader()
        output.writerows(rows)
    temporary.replace(args.out)


if __name__ == '__main__':
    main()
