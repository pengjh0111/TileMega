"""Distinguish a post-exit utilization sample from persistent ownerless saturation."""
import json
import sys
import time

from bootstrap import FRAME, HERE


def check():
    sys.path.insert(0,str(FRAME))
    from gpu_guard import gpu
    policy=json.loads((HERE/'guard_policy.json').read_text())
    row=gpu()
    if row['utilization_pct']<=policy['max_util_pct'] or row['owners']:
        return row
    for _ in range(policy['samples']-1):
        time.sleep(policy['interval_s'])
        row=gpu()
        if row['utilization_pct']<=policy['max_util_pct'] or row['owners']:
            return row
    raise RuntimeError('persistent ownerless GPU saturation; no reset or guard relaxation')
