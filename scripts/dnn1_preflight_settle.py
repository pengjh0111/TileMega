"""Inactive preflight candidate; no current collector or provider imports it."""
from __future__ import annotations

import copy
import math
import time


def settle(observe, *, required_samples, interval_s, timeout_s,
           needs_free_mib, admission_ok, clock=time.monotonic, sleep=time.sleep):
    """Retain every observation and require consecutive strict idle samples.

    A caller must hold its normal physical lock and yield to new reservations.
    This helper launches nothing and never assigns reported PID ownership.
    """
    if (required_samples < 1 or interval_s <= 0 or timeout_s <= 0 or
            needs_free_mib < 0):
        raise ValueError('positive sampling protocol and nonnegative memory required')
    start = clock()
    record = dict(required_samples=required_samples, interval_s=interval_s,
                  timeout_s=timeout_s, memory_floor_mib=1, samples=[],
                  power_monitoring=False)
    streak = 0
    while True:
        if not admission_ok():
            record['status'] = 'yield_reservation'
            break
        if clock() - start >= timeout_s:
            record['status'] = 'timeout'
            break
        try:
            rows = copy.deepcopy(observe())
            if not isinstance(rows, list) or not rows:
                raise ValueError('preflight omitted GPU observations')
            for row in rows:
                if (not isinstance(row.get('owners'), dict) or
                        not math.isfinite(float(row['used_mib'])) or
                        not math.isfinite(float(row['free_mib'])) or
                        row['used_mib'] < 0 or row['free_mib'] < 0):
                    raise ValueError('preflight omitted valid owners/memory')
        except Exception as error:
            record.update(status='telemetry_unavailable', error=repr(error))
            break
        record['samples'].append(dict(elapsed_s=clock() - start, rows=rows))
        if any(row['owners'] or row['free_mib'] < needs_free_mib for row in rows):
            record['status'] = 'occupied'
            break
        streak = streak + 1 if all(row['used_mib'] <= 1 for row in rows) else 0
        if clock() - start >= timeout_s:
            record['status'] = 'timeout'
            break
        if streak == required_samples:
            record['status'] = 'idle_observed'
            record['final_rows'] = rows
            break
        sleep(min(interval_s, timeout_s - (clock() - start)))
    record.update(elapsed_s=clock() - start, final_streak=streak)
    return record
