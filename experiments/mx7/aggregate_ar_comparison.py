"""Append source-bound resident-grid AR diagnostics; never launch an engine."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from tilemega.tp.process_cohorts import checked, validate_run


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def collect(proof):
    record = json.loads(Path(proof).read_text())
    result = []
    for entry in record['records']:
        if entry['kind'] != 'cost':
            continue
        validate_run(entry['run']['manifest'])
        summary_path = checked(entry['result'])
        summaries = json.loads(summary_path.read_text())
        processes = [json.loads(p.read_text()) for p in sorted(summary_path.parent.glob('process*/result.json'))]
        assert len(processes) == len({p['PID'] for p in processes}) == 3
        assert all(p['returncode'] == 0 for p in processes)
        assert all(p['gpu_indices'] == entry['group'] for p in processes)
        builds = [p['build'] for p in processes]
        assert all(b == builds[0] for b in builds)
        checked(dict(path=builds[0]['binary'], sha256=builds[0]['binary_sha256']))
        variant = entry.get('variant', 'system_relaxed')
        for index, item in enumerate(summaries):
            assert item['processes'] == 3 and item['key']['tp'] == entry['tp']
            assert all(p['records'][index]['pass'] for p in processes)
            assert item['rounds'] == [p['records'][index]['median_ns'] for p in processes]
            result.append(dict(record_type='resident_grid_communication_delta', evidence='verified',
                variant=variant, tp=item['key']['tp'], batch=item['key']['b'],
                stage=item['key']['stage'], tn=item['key']['tn'], gpu_indices=entry['group'],
                median_ns=item['median_ns'], p90_ns=item['p90_ns'], spread=item['spread'],
                fresh_processes=3, process_ids_json=json.dumps([p['PID'] for p in processes]),
                source_digest=item['source_digest'], artifact_sha256=builds[0]['binary_sha256'],
                source_proof_json=json.dumps(ref(proof)), measurements_json=json.dumps(ref(summary_path))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, default=Path('docs/results/MX7_tp_ar_diagnostics_c550.csv'))
    args = parser.parse_args()
    rows = [row for proof in args.proof for row in collect(proof)]
    old = list(csv.DictReader(args.out.open())) if args.out.exists() else []
    key = lambda r: (r['variant'], str(r['tp']), str(r['batch']), r['stage'], str(r['tn']), r['measurements_json'])
    seen = {key(r) for r in old}
    fresh = [r for r in rows if key(r) not in seen]
    if fresh:
        with args.out.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fresh[0]), lineterminator='\n')
            if not old:
                writer.writeheader()
            writer.writerows(fresh)
    print('communication diagnostic rows appended', len(fresh))


if __name__ == '__main__':
    main()
