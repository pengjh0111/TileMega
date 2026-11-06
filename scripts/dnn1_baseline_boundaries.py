"""Observe one process-and-memory boundary without a later telemetry re-read."""
from __future__ import annotations
import time


def empty_boundary(row):
    return isinstance(row.get('owners'), dict) and not row['owners'] and 0 <= row.get('used_mib', float('inf')) <= 1


def idle_boundary(cell, timeout_s=10., interval_s=.25, *, observe=None, clock=None, sleep=None):
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    if observe is None:
        from dnn1_baselines import exit_context_snapshot
        observe = lambda value: exit_context_snapshot(value, max(.001, timeout_s - (clock() - start)))
    start = clock()
    record = dict(started=time.time(), timeout_s=timeout_s, interval_s=interval_s,
        required_empty_samples=2, memory_floor_mib=1, samples=[],
        attribution='unknown; observations only', power_monitoring=False)
    streak = 0
    while True:
        try:
            row = observe(cell)
            if not isinstance(row.get('owners'), dict) or 'used_mib' not in row:
                raise ValueError('boundary telemetry omitted owners or memory')
        except Exception as error:
            record.update(status='telemetry_unavailable', error=str(error))
            break
        record['samples'].append(dict(elapsed_s=clock()-start, telemetry=row))
        streak = streak + 1 if empty_boundary(row) else 0
        if streak >= 2:
            record['status'] = 'empty_observed'
            break
        remaining = timeout_s - (clock()-start)
        if remaining <= 0:
            record['status'] = 'timeout'
            break
        sleep(min(interval_s, remaining))
    record.update(ended=time.time(), elapsed_s=clock()-start)
    return record


def final_boundary(drain):
    if drain.get('status') != 'empty_observed' or len(drain.get('samples', [])) < 2:
        raise ValueError('two process-and-memory idle samples were not observed')
    if any(not empty_boundary(s['telemetry']) for s in drain['samples'][-2:]):
        raise ValueError('final resource observations were not idle')
    return drain['samples'][-1]['telemetry']
