#!/usr/bin/env python3
"""Aggregate accepted fresh-process native calibration receipts without pooling quantiles."""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics

from sm90d1_common import read, sha, write

NUMBERS = {'grid', 'slot_bytes', 'slots', 'samples', 'median', 'p99', 'batch', 'tile_n',
           'input_k', 'tile_k', 'vector_sums', 'head_dim', 'queries', 'qk_norm',
           'chunks', 'block_keys', 'past', 'warps_n', 'warps_k', 'register_stages'}


def records(log, attempt):
    text = log.read_text().split(f'\nATTEMPT {attempt} ', 1)[-1]
    headers = [line for line in text.splitlines() if line.startswith('metric,')]
    if len(headers) != 1:
        raise ValueError(f'{log}: expected one calibration header in accepted attempt')
    header = headers[0]
    lines = text.split(header + '\n', 1)[1].splitlines()
    width = len(next(csv.reader([header])))
    data = [header] + [line for line in lines if len(next(csv.reader([line]))) == width]
    result = list(csv.DictReader(io.StringIO('\n'.join(data))))
    if not result:
        raise ValueError('accepted calibration produced no rows')
    for row in result:
        for key in NUMBERS & row.keys():
            row[key] = float(row[key]) if key in ('median', 'p99') else int(row[key])
        if row['samples'] < 1 or row['median'] <= 0 or row['p99'] < row['median']:
            raise ValueError(f'invalid calibration sample: {row}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--processes', type=int, default=5)
    p.add_argument('--resource-audit', type=Path)
    a = p.parse_args()
    state = read(a.state)
    provenance, observations = [], {}
    for name in a.names:
        for fresh in range(a.processes):
            job = f'{name}-f{fresh}'
            row = state[job]
            if row['status'] != 'done':
                raise ValueError(f'{job} is incomplete')
            directory = Path(row['out'])
            guard = read(directory / 'guard_result.json')
            if guard['code']:
                raise ValueError('calibration guard did not accept entire process')
            identity = directory / 'unit/physical-device-identity.json'
            command = directory / 'unit/physical-device-command.json'
            if read(command)['actual_returncode']:
                raise ValueError('calibration child failed')
            artifact = Path(read(command)['command'][0])
            provenance.append(dict(job=job, attempt=row['attempts'],
                guard_sha256=sha(directory / 'guard_result.json'),
                physical_device_identity_sha256=sha(identity),
                physical_uuid=read(identity)['nvml_uuid'], binary_sha256=sha(artifact),
                raw_log=str(a.state.parent / (job + '.log')),
                ownership_verified=guard['ownership_verified']))
            for value in records(a.state.parent / (job + '.log'), row['attempts']):
                key = tuple(sorted((k, v) for k, v in value.items()
                                   if k not in ('samples', 'median', 'p99')))
                observations.setdefault(key, []).append(value)
    result = []
    for key, values in sorted(observations.items()):
        if len(values) != a.processes:
            raise ValueError('component does not have the requested number of fresh processes')
        row = dict(key)
        median = statistics.median(v['median'] for v in values)
        row.update(processes=len(values), samples=sum(v['samples'] for v in values),
                   median=median, p99=statistics.median(v['p99'] for v in values),
                   process_range_pct=(max(v['median'] for v in values) -
                                      min(v['median'] for v in values)) / median * 100)
        result.append(row)
    if a.resource_audit:
        audit = read(a.resource_audit)
        if {p['binary_sha256'] for p in provenance} != {audit['binary_sha256']}:
            raise ValueError('resource audit belongs to a different calibration binary')
        dimensions = ('tile_n', 'tile_k', 'warps_n', 'warps_k', 'register_stages')
        kernels = {tuple(r[k] for k in dimensions): r for r in audit['kernels']}
        for row in result:
            if row['metric'] != 'native_ldmatrix_mma_k16':
                raise ValueError('mainloop resource audit supplied for another suite')
            facts = kernels[tuple(row[k] for k in dimensions)]
            if not facts['production_compatible']:
                raise ValueError('calibration surrogate contains extra local accesses or spills')
            row.update(production_compatible=True, resource_audit_sha256=sha(a.resource_audit))
    fields = sorted(set().union(*(v.keys() for v in result)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(result)
    write(a.out.with_suffix('.json'), dict(schema=1, observations=result, provenance=provenance,
        evidence='verified measurements; namespace-limited ownership is provisional',
        quantiles='median of process medians and median of process p99; no pooled p99',
        bulk_first_byte='first16 is a split-transfer proxy, not the first byte of an unsplit transfer'))
    print(json.dumps(dict(components=len(result), processes=len(provenance))))


if __name__ == '__main__':
    main()
