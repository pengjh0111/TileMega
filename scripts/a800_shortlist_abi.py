"""Read packed-weight descriptors from a compiled shortlist without a GPU."""
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

from tilemega.serving.memory import allocation_metadata, memory_report
from tilemega.serving.plan import Buffer, BufferInfo
from tilemega.serving.weights import weight_recipes


def buffers(path):
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    lookup = library.tm_plan_buffer
    lookup.argtypes = [ctypes.c_uint, ctypes.POINTER(BufferInfo)]
    lookup.restype = ctypes.c_int
    records = []
    for index in range(65536):
        raw = BufferInfo()
        if lookup(index, ctypes.byref(raw)):
            break
        records.append(Buffer(raw.name.decode(), raw.role, raw.dtype, raw.elements_constant,
                              raw.elements_per_batch, (raw.pack_json or b'').decode()))
    else:
        raise ValueError(f'unterminated native buffer table in {path}')
    if not records:
        raise ValueError(f'no native weight descriptors in {path}')
    return SimpleNamespace(buffers=records)


def read_plan(path):
    return json.loads(Path(str(path) + '.plan.json').read_text())


def check_shortlist(prefill, prepared, config, batch):
    prefix = () if prefill == 'prompt' else (Path(prefill),)
    prepared = json.loads(prepared.read_text())
    rows = []
    for item in prepared['receipt']['candidates']:
        if item['status'] != 'compiled':
            continue
        binary = Path(item['binary'])
        libraries = [buffers(path) for path in (*prefix, binary)]
        weights = weight_recipes(*libraries)
        manifests = [read_plan(path) for path in (*prefix, binary)]
        meta = [allocation_metadata(plan, lib) for plan, lib in zip(manifests, libraries)]
        report = memory_report(config, meta, batch, manifests[-1]['capacity'])
        rows.append(dict(rank=item['rank'], binary_sha256=item['binary_sha256'],
                         distinct_weight_buffers=len(weights), allocation=report))
    if not rows:
        raise ValueError('prepared shortlist has no validated packing ABI')
    return rows
