"""Append completed C0 cohorts with their binary/source-bound raw evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


FIELDS = ('record_type', 'evidence', 'gpu_indices', 'source_rank', 'destination_rank',
          'width', 'allocation', 'store', 'load', 'packets', 'requested_epochs',
          'all_to_all', 'fresh_processes', 'passed_processes', 'torn', 'timeouts',
          'roundtrip_ns', 'single_way_ns', 'bytes_per_second', 'ctas', 'peers',
          'status', 'binary_sha256', 'source_sha256', 'raw_evidence')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_evidence(value):
    path, separator, pair = value.partition(':')
    return str(Path(path).resolve()) + separator + pair


def collect(root):
    result = []
    p2p = root / 'c0_p2p_job.log'
    identity = json.loads((root / 'sync/p2p_probe_identity.json').read_text())
    for line in p2p.read_text().splitlines():
        if line.startswith('{'):
            record = json.loads(line)
            result.append(dict(record_type='p2p', evidence='verified', gpu_indices='4,5,6,7',
                               source_rank=record['source'], destination_rank=record['destination'],
                               status='pass' if record['pass'] else 'fail', fresh_processes=1,
                               passed_processes=int(record['pass']), binary_sha256=identity['binary_sha256'],
                               source_sha256=identity['source_sha256'],
                               raw_evidence=str(p2p) + f":{record['source']}->{record['destination']}"))
    for log in sorted(root.glob('resume*/c0_healthy_p2p.log')):
        guard = json.loads((log.parent / 'c0/healthy_p2p_guard/guard_result.json').read_text())
        if guard['code'] or sha(Path(identity['binary'])) != identity['binary_sha256']:
            raise ValueError('resumed P2P scope is not source-bound and qualified')
        records = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
        if len(records) != 12:
            raise ValueError('resumed four-card P2P matrix is incomplete')
        for record in records:
            result.append(dict(record_type='p2p', evidence='verified',
                gpu_indices=','.join(guard['gpu_indices']), source_rank=record['source'],
                destination_rank=record['destination'], fresh_processes=1,
                passed_processes=int(record['pass']), status='pass' if record['pass'] else 'fail',
                binary_sha256=identity['binary_sha256'], source_sha256=identity['source_sha256'],
                raw_evidence=str(log) + f":{record['source']}->{record['destination']}"))
    summaries = set(root.glob('c0/**/summary.json')) | set(root.glob('resume*/c0/**/summary.json'))
    summaries |= set(root.glob('resume*/alternate_*/**/summary.json'))
    for summary in sorted(summaries):
        cohort = summary.parent
        processes = sorted(cohort.glob('process*/result.json'))
        if len(processes) != 5:
            raise ValueError('C0 summary is not backed by five fresh processes')
        inputs = [json.loads(path.read_text()) for path in processes]
        first = inputs[0]
        if all('PID' in row for row in inputs) and len({row['PID'] for row in inputs}) != 5:
            raise ValueError('C0 requires five distinct fresh process IDs')
        if 'alternate_' in str(summary) and not all('PID' in row for row in inputs):
            raise ValueError('new hardware qualification requires captured process IDs')
        if len({(row['binary_sha256'], row['source_sha256'], tuple(row['command'])) for row in inputs}) != 1:
            raise ValueError('probe source/binary/command differs within cohort')
        values = [row['records'][-1] for row in inputs if row['records']]
        passed = sum(row['passed'] for row in inputs)
        if Path(first['binary']).name == 'p2p_probe':
            if any(len(row['records']) != 12 for row in inputs):
                raise ValueError('four-card P2P matrix is incomplete')
            for index, entry in enumerate(first['records']):
                pair = (entry['source'], entry['destination'])
                if any((row['records'][index]['source'], row['records'][index]['destination']) != pair for row in inputs):
                    raise ValueError('P2P pair order differs between fresh processes')
                pair_passes = sum(row['passed'] and row['records'][index]['pass'] for row in inputs)
                result.append(dict(record_type='p2p', evidence='verified',
                    gpu_indices=','.join(map(str, first['gpu_indices'])), source_rank=pair[0],
                    destination_rank=pair[1], fresh_processes=5, passed_processes=pair_passes,
                    status='pass' if pair_passes == 5 else 'fail', binary_sha256=first['binary_sha256'],
                    source_sha256=first['source_sha256'], raw_evidence=str(summary) + f':{pair[0]}->{pair[1]}'))
            continue
        row = dict(record_type='ll' if Path(first['binary']).name == 'll_litmus' else 'p2p_timing', evidence='verified',
                   gpu_indices=','.join(map(str, first['gpu_indices'])), fresh_processes=5,
                   passed_processes=passed, status='pass' if passed == 5 else 'fail',
                   binary_sha256=first['binary_sha256'], source_sha256=first['source_sha256'],
                   raw_evidence=str(summary), torn=sum(value.get('torn', 0) for value in values),
                   timeouts=sum(value.get('timeouts', 0) for value in values))
        if values:
            value = values[0]
            for key in ('width', 'allocation', 'store', 'load', 'packets', 'all_to_all', 'ctas', 'peers'):
                if key in value:
                    row[key] = value[key]
            if 'epochs' in value:
                row['requested_epochs'] = value['epochs']
            for key in ('roundtrip_ns', 'single_way_ns', 'bytes_per_second'):
                if key in value:
                    row[key] = statistics.median(value[key] for value in values)
        else:
            command = first['command']
            if row['record_type'] == 'p2p_timing':
                row.update(width=8, allocation=int(command[-5]), store=int(command[-4]),
                           load=int(command[-3]), ctas=int(command[-2]), peers=int(command[-1]))
            else:
                # Unsupported allocation/API outcomes keep all five stderr records.
                row.update(width=int(command[-7]), allocation=int(command[-4]), store=int(command[-3]),
                           load=int(command[-2]), packets=int(command[-6]), requested_epochs=int(command[-5]),
                           all_to_all=int(command[-1]))
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/mx7'))
    parser.add_argument('--out', type=Path, default=Path('docs/results/MX7_tp_probe_c550.csv'))
    args = parser.parse_args()
    rows = collect(args.root)
    old = list(csv.DictReader(args.out.open())) if args.out.exists() else []
    seen = {canonical_evidence(row['raw_evidence']) for row in old}
    fresh = []
    for row in rows:
        key = canonical_evidence(row['raw_evidence'])
        if key not in seen:
            fresh.append(row)
            seen.add(key)
    if fresh:
        with args.out.open('a', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator='\n')
            if not old:
                writer.writeheader()
            writer.writerows(fresh)
    print('C0 completed cohorts appended', len(fresh), 'total', len(old) + len(fresh))


if __name__ == '__main__':
    main()
