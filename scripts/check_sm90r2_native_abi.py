#!/usr/bin/env python3
"""Check host buffer and workload compatibility without creating a GPU plan."""
import argparse
import ctypes as C
import json
from pathlib import Path
from types import SimpleNamespace

from tilemega.serving.memory import allocation_metadata, memory_report
from tilemega.serving.plan import Buffer, BufferInfo, PlanLibrary
from tilemega.serving.weights import weight_recipes


def host_library(path):
    lib = C.CDLL(str(path.resolve()))
    lib.tm_plan_buffer.argtypes = [C.c_uint, C.POINTER(BufferInfo)]
    lib.tm_plan_buffer.restype = C.c_int
    buffers = []
    for index in range(10000):
        item = BufferInfo()
        if lib.tm_plan_buffer(index, C.byref(item)):
            break
        buffers.append(Buffer(item.name.decode(), item.role, item.dtype,
                              item.elements_constant, item.elements_per_batch,
                              item.pack_json.decode() if item.pack_json else ''))
    else:
        raise ValueError('buffer table lacks a bound')
    return SimpleNamespace(buffers=buffers, lib=lib, path=path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefill', type=Path, required=True)
    p.add_argument('--decode', type=Path, required=True)
    p.add_argument('--model-config', type=Path, required=True)
    a = p.parse_args()
    libs = [host_library(path) for path in (a.prefill, a.decode)]
    manifests = [json.loads(Path(str(path)+'.plan.json').read_text())
                 for path in (a.prefill, a.decode)]
    batch = manifests[1]['runtime_batch']
    recipes = weight_recipes(*libs)
    names = set(recipes) | {b.name for lib in libs for b in lib.buffers if b.role == 1}
    for lib, manifest in zip(libs, manifests):
        # Exercise the public Python pointer binder against symbolic addresses.
        # Replacing only its native create call prevents CUDA initialization.
        proxy = PlanLibrary.__new__(PlanLibrary)
        proxy.buffers = lib.buffers
        proxy.info = SimpleNamespace(batch_lo=manifest['batch_lo'], batch_hi=manifest['batch_hi'])
        proxy.lib = SimpleNamespace(tm_plan_create=lambda *_: 1)
        proxy.create(batch, {name: 16 for name in names}, 0)
    report = memory_report(json.loads(a.model_config.read_text()),
                           [allocation_metadata(m, lib) for m, lib in zip(manifests, libs)],
                           batch, manifests[1]['capacity'])
    print(json.dumps(dict(evidence='verified host ABI checks; no GPU plan created',
                         buffers=[len(lib.buffers) for lib in libs],
                         recipes=len(recipes), allocation=report), indent=2))


if __name__ == '__main__':
    main()
