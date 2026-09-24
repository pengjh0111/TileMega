#!/usr/bin/env python3
"""Wait for GPU memory and an idle device before fresh R9 measurements.

The admission estimate is fixture bytes plus 2048 MiB for runtime buffers,
events and allocator headroom. It is not a measured peak or a correctness
gate. Every observation is retained; admission never changes a test result.
"""
import json
import pathlib
import subprocess
import time


def required_mib(fixture):
    size = sum(p.stat().st_size for p in pathlib.Path(fixture).rglob('*') if p.is_file())
    return (size + 2**20 - 1) // 2**20 + 2048


def sample():
    command = ['nvidia-smi', '--id=0',
               '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits']
    raw = subprocess.check_output(command, text=True, timeout=15).strip()
    free, busy = map(int, raw.split(','))
    return dict(free_mib=free, utilization_percent=busy, command=command)


def wait_for_device(fixture, log, *, probe=sample, sleep=time.sleep, needed=None):
    need = required_mib(fixture) if needed is None else needed
    consecutive = 0
    count = 0
    with pathlib.Path(log).open('a') as out:
        while consecutive < 3:
            observation = probe()
            ready = observation['free_mib'] >= need and observation['utilization_percent'] <= 5
            consecutive = consecutive + 1 if ready else 0
            out.write(json.dumps(dict(observed_ns=time.time_ns(), need_mib=need,
                                     consecutive_ready=consecutive, **observation)) + '\n')
            out.flush()
            if count % 15 == 0:
                print(f"GPU_ADMISSION need_mib={need} free_mib={observation['free_mib']} "
                      f"utilization={observation['utilization_percent']} ready={consecutive}/3", flush=True)
            count += 1
            if consecutive < 3:
                sleep(2)


def resource_failure(text):
    # An OOM after hashes or a numeric mismatch must never trigger a retry.
    return 'out of memory' in text.lower() and 'E2E_HASH' not in text and 'E2E_TIME' not in text
