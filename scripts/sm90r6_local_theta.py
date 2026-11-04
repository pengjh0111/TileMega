#!/usr/bin/env python3
"""Select AB1 family partitions from the native domain and measured ledgers."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics


def geometry_key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def traffic_choice(options, grid):
    if not options:
        raise ValueError('native family domain is empty at this residency')
    distance = lambda row: max(grid / 2 - row['tasks'], row['tasks'] - grid * 2, 0)
    closest = min(map(distance, options))
    # User decision 2026-11-04: only an empty task window admits its nearest shell.
    eligible = [row for row in options if distance(row) == closest]
    chosen = min(eligible, key=lambda row: (row['traffic_bytes'], abs(row['tasks'] - grid)))
    return chosen, dict(task_window=[grid / 2, grid * 2], window_distance=closest,
                       exception='nearest_task_window_shell' if closest else None,
                       tie_break='distance to G; remaining ties retain native enumeration order')


def measured_choice(options, records, layers):
    by_geometry = {}
    for row in records:
        if row['guard_code'] != 0 or not row['fresh_process']:
            raise ValueError('unaccepted probe process')
        key = geometry_key(row['geometry'])
        processes = by_geometry.setdefault(key, {})
        process = row['process_id']
        if process in processes:
            raise ValueError('duplicate probe process for one geometry')
        samples = row['stage_ns']
        if len(samples) != layers or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError('probe must time exactly L target stage instances')
        processes[process] = statistics.median(samples)
    expected = {geometry_key(row['geometry']) for row in options}
    if set(by_geometry) != expected or any(len(p) != 3 for p in by_geometry.values()):
        raise ValueError('measured AB1 needs every legal geometry in three fresh processes')
    duration = {key: statistics.median(p.values()) for key, p in by_geometry.items()}
    chosen = min(options, key=lambda row: duration[geometry_key(row['geometry'])])
    return chosen, dict(isolated_ns=duration[geometry_key(chosen['geometry'])],
                       processes=3, instances_per_process=layers,
                       tie_break='equal medians retain native enumeration order')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--criterion', choices=('traffic', 'measured'), required=True)
    p.add_argument('--domains', type=Path, required=True)
    p.add_argument('--cell', required=True)
    p.add_argument('--a0', type=Path, required=True, help='A0 selected coordinate JSON')
    p.add_argument('--measurements', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(line) for line in a.domains.read_text().splitlines() if line.strip()]
    cells = [row for row in rows if row['input']['id'] == a.cell]
    if len(cells) != 1:
        raise ValueError('domain catalog must contain exactly one matching cell')
    cell = cells[0]
    coordinates = json.loads(a.a0.read_text())
    result = copy.deepcopy(coordinates)
    records = [json.loads(line) for line in a.measurements.read_text().splitlines()] if a.measurements else []
    choices = []
    for domain in cell['domains']:
        family = domain['family']
        if family == 'attention' or domain['residency'] != coordinates['residency']:
            continue
        if a.criterion == 'traffic':
            chosen, evidence = traffic_choice(domain['geometries'], domain['grid'])
        else:
            chosen, evidence = measured_choice(domain['geometries'],
                [r for r in records if r['family'] == family], cell['layers'])
        result['gemms'][family] = chosen['geometry']
        choices.append(dict(family=family, grid=domain['grid'], **chosen, **evidence))
    if not choices:
        raise ValueError('no matching family domain')
    if a.out.exists() or Path(str(a.out)+'.selection.json').exists():
        raise ValueError('retain previous theta and selection evidence; choose a new path')
    receipt = dict(criterion=a.criterion, cell=a.cell, choices=choices,
        domains=str(a.domains.resolve()), domains_sha256=hashlib.sha256(a.domains.read_bytes()).hexdigest(),
        a0=str(a.a0.resolve()), a0_sha256=hashlib.sha256(a.a0.read_bytes()).hexdigest(),
        approximation=cell['approximation'],
        scope='pinned GEMM coordinates; remaining coordinates are re-solved by Algorithm 1')
    if a.measurements:
        receipt.update(measurements=str(a.measurements.resolve()),
            measurements_sha256=hashlib.sha256(a.measurements.read_bytes()).hexdigest())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    Path(str(a.out)+'.selection.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(cell=a.cell, families=len(choices), exceptions=sum(bool(c.get('exception')) for c in choices))))


if __name__ == '__main__':
    main()
