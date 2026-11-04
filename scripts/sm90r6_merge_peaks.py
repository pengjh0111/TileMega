#!/usr/bin/env python3
"""Validate all H1 medians against five guarded raw processes before publication."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from sm90r6_hw_bound import peaks

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    summary = root / 'summary.csv'
    with summary.open() as stream:
        rows = list(csv.DictReader(stream))
    bw, compute = peaks(summary)
    if {r['binary_sha256'] for r in rows} != {sha(root / 'hw_peaks')} or \
            {r['source_sha256'] for r in rows} != {sha(ROOT / 'scripts/sm90r6_hw_peaks.cu')}:
        raise ValueError('H1 summary does not match the retained binary and source')
    records = []
    for gpu in [0, 1, 2, 3, 4, 5, 7]:
        for item in ['hbm_read', 'hbm_read_bulk', 'bf16_wgmma', 'bf16_mma_sync']:
            # Collectors retry whole items. Only the last complete p0..p4
            # series can supersede an earlier attempt; no best-sample selection.
            samples = sorted((root / f'gpu{gpu}' / item).glob('*/samples.json'),
                             key=lambda p: p.stat().st_mtime)[-5:]
            if len(samples) != 5 or {p.parent.name.split('-')[0] for p in samples} != \
                    {'p0', 'p1', 'p2', 'p3', 'p4'}:
                raise ValueError(f'incomplete fresh-process series: GPU {gpu}, {item}')
            data, evidence = {}, []
            for path in samples:
                guard = json.loads((path.parent / 'guard_result.json').read_text())
                if guard['code'] != 0:
                    raise ValueError(f'rejected H1 raw process: {path.parent}')
                evidence.append(dict(directory=str(path.parent), samples_sha256=sha(path), guard=guard))
                for sample in json.loads(path.read_text())['samples']:
                    data.setdefault(sample['config'], []).append(sample['value'])
            selected = [r for r in rows if int(r['gpu']) == gpu and r['item'] == item]
            if {r['config'] for r in selected} != set(data):
                raise ValueError(f'raw configuration coverage differs: GPU {gpu}, {item}')
            for row in selected:
                values = data[row['config']]
                for name, value in [('median', statistics.median(values)),
                                    ('minimum', min(values)), ('maximum', max(values))]:
                    if len(values) != 5 or abs(value - float(row[name])) >= 0.00051:
                        raise ValueError(f'raw {name} differs: GPU {gpu}, {item}, {row["config"]}')
            records.append(dict(gpu=gpu, item=item, fresh_processes=5, evidence=evidence))
    manifest = dict(evidence='verified', normalization='maximum of five-process configuration medians',
        bandwidth_gbps=bw, wgmma_tflops=compute, records=records, summary_sha256=sha(summary),
        numa_concurrency='user-authorized trial; DT-P not certified',
        ownership='guard acceptance provisional_container_pid_namespace',
        unsupported='8 x 32 KiB exceeds CTA shared-memory limit; user approved omission')
    (root / 'validated_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows.sort(key=lambda r: (int(r['gpu']), r['item'], r['config']))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(rows=len(rows), bandwidth_gbps=bw, wgmma_tflops=compute)))


if __name__ == '__main__':
    raise SystemExit(main())
