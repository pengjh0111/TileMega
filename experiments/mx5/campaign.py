#!/usr/bin/env python3
"""Prepare immutable stage batches and run three fresh processes under a guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def prepare(args):
    rows = [json.loads(line) for line in args.checked.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row['kind'] in args.kinds and row['key']['threads'] in args.threads]
    blocked = [row for row in selected if not row.get('fits') or row.get('selected_resource_step') is None
               or row.get('query_status') != 0 or not row.get('query_only')]
    if blocked:
        raise ValueError(f'{len(blocked)} requested keys lack a passing derived resource query')
    if not selected:
        raise ValueError('no requested stage keys')
    if args.out.exists():
        raise ValueError('output directory already exists; preserve the earlier campaign')
    seen = set()
    for row in selected:
        index = row['kind'], json.dumps(row['key'], sort_keys=True)
        if index in seen:
            raise ValueError('duplicate checked stage key')
        seen.add(index)
        metadata = json.loads(Path(row['library']).with_suffix('.json').read_text())
        binary = hashlib.sha256(Path(row['library']).read_bytes()).hexdigest()
        if binary != row['provenance']['binary_sha256'] or metadata['binary_sha256'] != binary:
            raise ValueError('checked binary identity changed')
        for field in ('source_sha256', 'binary_sha256', 'base_head'):
            if metadata[field] != row['provenance'][field]:
                raise ValueError('checked build identity changed: ' + field)
        baseline = row['r_next_reference']['provenance']
        if baseline['source_sha256'] != metadata['source_sha256']:
            raise ValueError('R_next and measured kernel have different body sources')
    args.out.mkdir(parents=True)
    jobs = []
    for kind in args.kinds:
        group = [row for row in selected if row['kind'] == kind]
        group.sort(key=lambda row: tuple(row['key'].get(field, 0) for field in
                                       ('n', 'k', 'dim', 'q', 'hkv', 'b', 'length', 'b_min',
                                        'tn', 'wn', 'depth', 'op', 'norm', 'part', 'threads')))
        for begin in range(0, len(group), args.batch_size):
            batch = group[begin:begin + args.batch_size]
            directory = args.out / f'{kind}_{begin // args.batch_size:03d}'
            directory.mkdir()
            keys = []
            for row in batch:
                key = dict(row['key'], library=row['library'])
                if kind != 'attention':
                    key['pw'] = key.pop('prefetch')
                keys.append(key)
            key_file = directory / 'keys.json'
            reference_file = directory / 'references.json'
            key_file.write_text(json.dumps(keys, indent=2) + '\n')
            reference_file.write_text(json.dumps(dict(references=[row['r_next_reference'] for row in batch]),
                                                 indent=2) + '\n')
            (directory / 'checked.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in batch))
            jobs.append(dict(kind=kind, keys=str(key_file.resolve()), directory=str(directory.resolve()),
                             keys_sha256=hashlib.sha256(key_file.read_bytes()).hexdigest(), rows=len(keys),
                             suggested_card=7 if len(jobs) % 2 == 0 else 6))
    manifest = dict(schema_version=1, full_table=False, checked=str(args.checked.resolve()),
                    checked_sha256=hashlib.sha256(args.checked.read_bytes()).hexdigest(),
                    kinds=args.kinds, threads=args.threads, rows=len(selected), batch_size=args.batch_size,
                    processes_per_key=3, jobs=jobs)
    (args.out / 'campaign.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(rows=len(selected), jobs=len(jobs), manifest=str(args.out / 'campaign.json'))))


def run(args):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('run batches through scripts/mx_gpu_job.sh')
    document = json.loads(args.manifest.read_text())
    job = document['jobs'][args.job]
    key_file = Path(job['keys'])
    if hashlib.sha256(key_file.read_bytes()).hexdigest() != job['keys_sha256']:
        raise ValueError('campaign keys changed')
    directory = Path(job['directory'])
    outputs = [directory / f'round{round_number}.jsonl' for round_number in (1, 2, 3)]
    if any(path.exists() or path.with_suffix('.log').exists() for path in outputs):
        raise ValueError('batch has previous output; use a new campaign for retries')
    script = 'stage_bench.py' if job['kind'] == 'gemm' else job['kind'] + '_stage_bench.py'
    for round_number, output in enumerate(outputs, 1):
        command = [sys.executable, str(ROOT / 'experiments/mx5' / script), '--keys', str(key_file),
                   '--out', str(output), '--round', str(round_number), '--iterations', str(args.iterations)]
        with output.with_suffix('.log').open('x') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        (directory / f'process{round_number}.json').write_text(json.dumps(
            dict(round=round_number, command=command, returncode=result.returncode), indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'batch failed: {output.with_suffix(".log")}')
        print(json.dumps(dict(job=args.job, round=round_number, output=str(output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('prepare')
    create.add_argument('--checked', type=Path, required=True)
    create.add_argument('--out', type=Path, required=True)
    create.add_argument('--threads', type=int, nargs='+', choices=(512, 576), default=[512])
    create.add_argument('--kinds', nargs='+', choices=('gemm', 'head', 'attention'), required=True)
    create.add_argument('--batch-size', type=int, default=12)
    execute = commands.add_parser('run')
    execute.add_argument('--manifest', type=Path, required=True)
    execute.add_argument('--job', type=int, required=True)
    execute.add_argument('--iterations', type=int, default=200)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.batch_size < 1:
            parser.error('batch size must be positive')
        prepare(args)
    else:
        if args.job < 0 or args.iterations < 3:
            parser.error('invalid job/iteration count')
        run(args)


if __name__ == '__main__': main()
