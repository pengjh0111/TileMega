"""Exercise one owned SDK stream per logical rank, including events and copies."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--helper', type=Path, required=True)
a = p.parse_args()
if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
    raise RuntimeError('SDK stream probe requires the existing group guard')
group = list(map(int, os.environ['TILEMEGA_DEVICE_GROUP'].split(',')))
if len(group) not in (2, 4):
    raise ValueError('stream probe requires TP2 or TP4')
spec = importlib.util.spec_from_file_location('mx7_stream_trial', a.helper)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
owners, rows = [], []
try:
    for device in range(len(group)):
        with torch.cuda.device(device):
            begin = time.perf_counter()
            owner = helper.OwnedTpStream(device)
            owners.append(owner)
            properties = owner.sdk.properties(owner.handle)
            assert properties == dict(flags=1, priority=0)
            assert owner.stream.cuda_stream == owner.handle
            source = torch.arange(1024, dtype=torch.int32)
            target = torch.empty(1024, dtype=torch.int32, device=torch.device('cuda', device))
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(owner.stream):
                start.record(owner.stream)
                target.copy_(source, non_blocking=True)
                target.add_(1)
                end.record(owner.stream)
            owner.stream.synchronize()
            assert torch.equal(target.cpu(), source + 1)
            rows.append(dict(logical_device=device, physical_device=group[device],
                             flags=properties['flags'], priority=properties['priority'],
                             initialization_and_copy_ms=(time.perf_counter()-begin)*1000,
                             event_ms=start.elapsed_time(end), exact_integer_copy_and_add=True))
finally:
    for owner in reversed(owners):
        owner.close()
assert all(owner.handle is None for owner in owners)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
print(json.dumps(dict(**{"pass": True}, PID=os.getpid(), gpu_indices=group, world=len(group),
                      streams=len(owners), rows=rows, helper_sha256=sha(a.helper),
                      runtime_library=helper.MacaStreamRuntime.library_path,
                      runtime_sha256=sha(helper.MacaStreamRuntime.library_path),
                      timing_eligible=False)), flush=True)
