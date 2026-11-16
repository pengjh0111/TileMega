#!/usr/bin/env python3
"""Validate boundary windows from three fresh children of one GPU reservation."""
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from tilemega.build.identity import sha, verify


def read(path):
    return json.loads(Path(path).read_text())


def qualify(nontrace_path, cohort_path, library, reference, gpu):
    measured, cohort = read(nontrace_path), read(cohort_path)
    identity, control = verify(library), verify(reference)
    if (measured['protocol'] != 'P-SMOKE' or measured['gpu'] != gpu
            or len(measured['measurements']) != 1 or control['trace']
            or identity['defines'].get('TILEMEGA_TRACE_STAGE') != '3'
            or identity['plan']['native_forward'] != control['plan']['native_forward']):
        raise ValueError('boundary cohort differs from its nontrace reference')
    row = measured['measurements'][0]
    timing = read(row['measurement'])
    if (row['guard']['code'] or timing['gpu_index'] != gpu
            or timing['artifact_id'] != control['artifact_id']
            or timing['measurement']['warmup'] != 50
            or timing['measurement']['iterations'] != 200
            or len(timing['samples_ms']) != 200
            or timing['median_ms'] != measured['candidate_ms']):
        raise ValueError('boundary cohort lacks a complete nontrace timing process')
    if (cohort['status'] != 'passed' or len(cohort['processes']) != 3
            or len(cohort['traces']) != 3 or read(cohort['guard'])['code']
            or any(p['exit_code'] for p in cohort['processes'])):
        raise ValueError('boundary cohort lacks three successful guarded processes')
    stage_count = len(identity['plan']['native_forward']['stages'])
    windows, processes, clocks = [], [], []
    for process, path in zip(cohort['processes'], cohort['traces']):
        folder = Path(path)
        result, samples = read(folder/'result.json'), read(folder/'samples.json')
        if (folder.resolve() != (Path(process['out'])/'trace').resolve()
                or result['pid'] != process['pid'] or result['gpu'] != gpu
                or result['artifact_id'] != identity['artifact_id']
                or result['reference_sha256'] != sha(reference)
                or result['input_sha256'] != measured['input_sha256']
                or result['trace_level'] != 3 or result['warmup'] != 20
                or result['samples'] != 50):
            raise ValueError('boundary child differs from the checked artifact or card')
        if (len(samples) != 50
                or any([r['stage'] for r in sample] != list(range(stage_count))
                       or any(not math.isfinite(r['stage_ns']) or r['stage_ns'] <= 0
                              for r in sample) for sample in samples)):
            raise ValueError('boundary child has incomplete or invalid stage windows')
        windows.append(statistics.median(sum(r['stage_ns'] for r in sample) for sample in samples))
        processes.append(dict(pid=result['pid'], out=str(folder)))
        clocks.append({k: result[k] for k in ('start_sm_clock_mhz', 'end_sm_clock_mhz')})
    if len({p['pid'] for p in processes}) != 3 or timing['pid'] in {p['pid'] for p in processes}:
        raise ValueError('boundary cohort reused a process')
    elapsed = measured['candidate_ms']
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('invalid nontrace reference latency')
    window = statistics.median(windows)
    difference = window/(elapsed*1e6)-1
    return dict(evidence='verified diagnostic only', model='bert', batch=measured['batch'],
        gpu=gpu, library=str(library), reference=str(reference), trace_ns=window,
        nontrace_ms=elapsed, relative_difference=difference, passed=abs(difference) <= .05,
        processes=processes, clocks=clocks, process_windows_ns=windows,
        nontrace_source=str(nontrace_path), cohort_source=str(cohort_path),
        scope='Three fresh20/50 children of one guarded reservation;each sample checks exact '
              'output bytes against the nontrace artifact;fresh same-card50/200 reference;'
              'qualification only,not correctness or performance admission')
