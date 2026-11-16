"""Export actual kernel activity without inferring stage identity from names."""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics

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
    parser.add_argument('--nvtx-proof', type=Path, help='Worker engine.json with audited module ranges')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(args.mapping.read_text()) if args.mapping else []
    if any(not row.get('source_evidence') or not row.get('confirmation') for row in mapping):
        raise ValueError('stage mapping requires source/shape/NVTX confirmation evidence')
    connection = sqlite3.connect(f'file:{args.sqlite}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    strings = dict(connection.execute('SELECT id,value FROM StringIds'))
    proofs = {}
    if args.nvtx_proof:
        engine = json.loads(args.nvtx_proof.read_text())
        for rank in engine['nvtx_attribution']:
            for row in rank['modules']:
                proofs[row['label']] = row
    ranges = collections.defaultdict(list)
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if proofs and 'NVTX_EVENTS' in tables:
        for source in connection.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
            row = dict(source)
            text = row.get('text') or strings.get(row.get('textId')) or ''
            label = text.split('|inputs=', 1)[0]
            if label in proofs:
                row['proof'] = dict(proofs[label], observed_nvtx_text=text)
                ranges[row['globalTid']].append(row)
    runtime_stage = {}
    if ranges:
        for source in connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
            row = dict(source)
            containing = [item for item in ranges[row['globalTid']]
                          if item['start'] <= row['start'] and row['end'] <= item['end']]
            if containing:
                inner = min(containing, key=lambda item: item['end']-item['start'])
                runtime_stage[(row['globalTid'] & ~0xffffff, row['correlationId'])] = inner['proof']
    kernels = []
    for source in connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'):
        row = dict(source)
        row['name'] = strings[row['demangledName']]
        matches = [item for item in mapping if re.search(item['kernel_regex'], row['name'])]
        if len(matches) > 1:
            raise ValueError('ambiguous stage mapping;fusion must remain one measured region')
        row['stage'] = matches[0]['stage'] if matches else 'unresolved'
        row['source_evidence'] = matches[0]['source_evidence'] if matches else ''
        proof = runtime_stage.get((row.get('globalPid'), row['correlationId']))
        if proof:
            row['stage'] = proof['stage']
            row['source_evidence'] = json.dumps(dict(
                confirmation='Same-process CUDA runtime correlation inside source-audited NVTX module call',
                **proof), separators=(',', ':'))
        kernels.append(row)
    if not kernels:
        raise ValueError('no captured GPU kernels;not a zero-time model result')
    groups = collections.defaultdict(list)
    stages = collections.defaultdict(list)
    for row in kernels:
        resource = tuple(row[key] for key in ('gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ',
                                             'registersPerThread', 'staticSharedMemory', 'dynamicSharedMemory'))
        groups[(row['deviceId'], row['name'], resource, row['stage'])].append(row)
        stages[(row['deviceId'], row['stage'])].append(row)
    rows = []
    for (device, name, resource, stage), group in groups.items():
        rows.append(dict(device=device, kernel=name, calls=len(group),
                         kernel_sum_ns=sum(row['end']-row['start'] for row in group),
                         duration_median_ns=statistics.median(row['end']-row['start'] for row in group),
                         kernel_busy_union_ns=interval_union([(row['start'], row['end']) for row in group]),
                         grid=str(resource[:3]), block=str(resource[3:6]),
                         registers_per_thread=resource[6], static_shared_bytes=resource[7],
                         dynamic_shared_bytes=resource[8], stage=stage,
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
        intervals = [(row['start'], row['end']) for row in group]
        copies = []
        for table in ('CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
            if table in tables:
                copies.extend(dict(row) for row in connection.execute(
                    f'SELECT * FROM {table} WHERE deviceId=?', (device,)))
        all_intervals = intervals + [(row['start'], row['end']) for row in copies]
        graph_intervals = []
        graph_scope = 'none'
        if 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE' in tables:
            graph_intervals = [(row['start'], row['end']) for row in connection.execute(
                'SELECT * FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE deviceId=? ORDER BY start', (device,))]
            graph_scope = 'Whole graph execution spans,not per-node SM busy'
        else:
            launch_ids = {(row['globalTid'] & ~0xffffff, row['correlationId'])
                          for row in connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')
                          if 'GraphLaunch' in strings[row['nameId']]}
            graph_groups = collections.defaultdict(list)
            for row in group:
                identity = (row['globalPid'], row['correlationId'])
                if identity in launch_ids:
                    graph_groups[identity].append((row['start'], row['end']))
            graph_intervals = sorted((min(start for start, end in values), max(end for start, end in values))
                                     for values in graph_groups.values())
            if graph_intervals:
                graph_scope = 'Per-launch span of correlated graph-node kernels;includes internal gaps/overlap'
        devices[device] = dict(kernel_sum_ns=sum(row['end']-row['start'] for row in group),
                               kernel_busy_union_ns=interval_union(intervals),
                               captured_kernel_copy_busy_union_ns=interval_union(all_intervals),
                               captured_copy_memset_sum_ns=sum(row['end']-row['start'] for row in copies),
                               graph_launches=len(graph_intervals), graph_scope=graph_scope,
                               graph_span_median_ns=statistics.median(end-start for start, end in graph_intervals)
                               if graph_intervals else None,
                               graph_launch_cadence_median_ns=statistics.median(graph_intervals[i+1][0]-graph_intervals[i][0]
                                   for i in range(len(graph_intervals)-1)) if len(graph_intervals)>1 else None,
                               graph_span_sum_ns=sum(end-start for start, end in graph_intervals),
                               activity_coverage_including_graph_spans_ns=interval_union(all_intervals+graph_intervals),
                               first_to_last_kernel_span_ns=max(row['end'] for row in group)-min(row['start'] for row in group),
                               calls=len(group))
    (args.out / 'gpu_activity.json').write_text(json.dumps(dict(
        sqlite=str(args.sqlite), devices=devices,
        wall_clock_scope='First-to-last-kernel span excludes host work outside this span;not request E2E. '
                         'Kernel busy union excludes copies;separate all-activity union includes captured '
                         'copies/memsets. Capture all streams before attributing idle gaps.'), indent=2) + '\n')
    connection.close()


if __name__ == '__main__':
    main()
