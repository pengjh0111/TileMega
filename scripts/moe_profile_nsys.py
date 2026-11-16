"""Export actual kernel activity without inferring stage identity from names."""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import sqlite3

from moe_profile_metrics import interval_union


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sqlite', type=Path, required=True)
    parser.add_argument('--mapping', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(args.mapping.read_text()) if args.mapping else []
    if any(not row.get('source_evidence') or not row.get('confirmation') for row in mapping):
        raise ValueError('stage mapping requires source/shape/NVTX confirmation evidence')
    connection = sqlite3.connect(f'file:{args.sqlite}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    strings = dict(connection.execute('SELECT id,value FROM StringIds'))
    kernels = []
    for source in connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'):
        row = dict(source)
        row['name'] = strings[row['demangledName']]
        matches = [item for item in mapping if re.search(item['kernel_regex'], row['name'])]
        if len(matches) > 1:
            raise ValueError('ambiguous stage mapping;fusion must remain one measured region')
        row['stage'] = matches[0]['stage'] if matches else 'unresolved'
        row['source_evidence'] = matches[0]['source_evidence'] if matches else ''
        kernels.append(row)
    if not kernels:
        raise ValueError('no captured GPU kernels;not a zero-time model result')
    groups = collections.defaultdict(list)
    stages = collections.defaultdict(list)
    for row in kernels:
        resource = tuple(row[key] for key in ('gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ',
                                             'registersPerThread', 'staticSharedMemory', 'dynamicSharedMemory'))
        groups[(row['deviceId'], row['name'], resource)].append(row)
        stages[(row['deviceId'], row['stage'])].append(row)
    rows = []
    for (device, name, resource), group in groups.items():
        rows.append(dict(device=device, kernel=name, calls=len(group),
                         kernel_sum_ns=sum(row['end']-row['start'] for row in group),
                         kernel_busy_union_ns=interval_union([(row['start'], row['end']) for row in group]),
                         grid=str(resource[:3]), block=str(resource[3:6]),
                         registers_per_thread=resource[6], static_shared_bytes=resource[7],
                         dynamic_shared_bytes=resource[8], stage=group[0]['stage'],
                         source_evidence=group[0]['source_evidence'],
                         source_tool='Nsight Systems CUPTI kernel activity',
                         counter_status='unavailable:NCU ERR_NVGPUCTRPERM',
                         actual_m='', actual_n='', actual_k='', shape_evidence='',
                         sm_throughput_pct='', tensor_throughput_pct='', achieved_occupancy='',
                         dram_read_bytes='', dram_write_bytes='', dram_gbps='', l2_hit_rate='',
                         effective_flops='', arithmetic_intensity='',
                         scope='Captured ROI only;launch metadata is not hardware throughput/occupancy.'))
    rows.sort(key=lambda row: row['kernel_sum_ns'], reverse=True)
    write_csv(args.out / 'kernel_metrics.csv', rows)
    stage_rows = []
    for (device, stage), group in sorted(stages.items()):
        stage_rows.append(dict(device=device, stage=stage, calls=len(group),
                               kernel_sum_ns=sum(row['end']-row['start'] for row in group),
                               stage_busy_union_ns=interval_union([(row['start'], row['end']) for row in group]),
                               scope='Stage kernel sum may overlap other stages;do not sum stage unions as wall time.'))
    write_csv(args.out / 'stage_breakdown.csv', stage_rows)
    devices = {}
    for device in sorted({row['deviceId'] for row in kernels}):
        group = [row for row in kernels if row['deviceId'] == device]
        devices[device] = dict(kernel_sum_ns=sum(row['end']-row['start'] for row in group),
                               kernel_busy_union_ns=interval_union([(row['start'], row['end']) for row in group]),
                               first_to_last_kernel_span_ns=max(row['end'] for row in group)-min(row['start'] for row in group),
                               calls=len(group))
    (args.out / 'gpu_activity.json').write_text(json.dumps(dict(
        sqlite=str(args.sqlite), devices=devices,
        wall_clock_scope='First-to-last-kernel span excludes host work outside this span;not request E2E. '
                         'Busy union excludes copies;capture all streams before attributing idle gaps.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
