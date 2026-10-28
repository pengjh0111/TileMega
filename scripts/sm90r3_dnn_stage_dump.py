#!/usr/bin/env python3
"""N-2 diagnostic: the magnitude of every native forward buffer after a launch.

Reads each bound buffer back through `tm_plan_read_buffer` and prints its
extent, finite fraction and absolute mean/max in plan order, which localizes
the first stage whose output is unwritten or wrong.
"""
import argparse
import ctypes as C
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from tilemega.serving.plan import PlanLibrary
from tilemega.serving.weights import load_weights

DTYPES = {0: (torch.bfloat16, 2), 1: (torch.float32, 4), 2: (torch.int32, 4)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'export', 'bridge', 'inputs'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    library = PlanLibrary(a.library)
    bridge = json.loads(a.bridge.read_text())
    nodes = {node['name']: node for node in bridge['nodes']}
    signature = bridge['signature']
    inputs = [i['name'] for i in signature['inputs'] if i['kind'] == 'USER_INPUT']
    outputs = [o['name'] for o in signature['outputs'] if o['kind'] == 'USER_OUTPUT']
    tensors = load_weights(a.export / 'checkpoint', library, device='cuda')
    values = load_file(str(a.inputs), device='cuda')
    for name in inputs:
        tensors[name] = values[name].contiguous()
    for name in outputs:
        node = nodes[name]
        tensors[name] = torch.zeros([a.batch, *map(int, node['shape'][1:])],
                                    dtype=torch.bfloat16, device='cuda')
    plan = library.create(a.batch, {n: v.data_ptr() for n, v in tensors.items()}, 0)
    read = library.lib.tm_plan_read_buffer
    read.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint64]
    read.restype = C.c_int
    with plan:
        plan.set_steps([0])
        plan.launch(0, 1, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        rows = []
        for index, info in enumerate(library.buffers):
            dtype, size = DTYPES[info.dtype]
            count = info.elements_constant + a.batch * info.elements_per_batch
            if not count:
                continue
            host = torch.empty(int(min(count, 1 << 22)), dtype=dtype)
            code = read(plan.handle, index, host.data_ptr(), host.numel() * size)
            if code:
                rows.append(dict(index=index, name=info.name, error=code))
                continue
            value = host.float()
            rows.append(dict(index=index, name=info.name, role=info.role, elements=int(count),
                             read=int(host.numel()), finite=float(torch.isfinite(value).float().mean()),
                             absmean=float(value.abs().mean()), absmax=float(value.abs().max()),
                             zeros=float((value == 0).float().mean())))
    for row in rows:
        print(json.dumps(row))
    if a.out:
        a.out.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
