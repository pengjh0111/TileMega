"""Read packed-weight descriptors from a compiled shortlist without a GPU."""
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

from tilemega.bench.scheduler import write_json
from tilemega.build.identity import sha
from tilemega.serving.host_preparation import prepare
from tilemega.serving.plan import Buffer, BufferInfo


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


def freeze_host_preparation(config, batch, paths, out):
    plans = [read_plan(path) for path in paths]
    identities = [json.loads(Path(str(path)+'.identity.json').read_text()) for path in paths]
    for path, identity in zip(paths, identities, strict=True):
        if identity['binary_sha256'] != sha(path) or identity['manifest_sha256'] != sha(str(path)+'.plan.json'):
            raise ValueError(f'host preparation plan changed at {path}')
    record = prepare(config, batch, plans, identities, [buffers(path) for path in paths])
    if out.exists():
        if json.loads(out.read_text()) != record:
            raise ValueError(f'host preparation changed at {out}')
    else:
        write_json(out, record)
    return record


def check_shortlist(prefill, prepared, config, batch):
    prefix = () if prefill == 'prompt' else (Path(prefill),)
    prepared = json.loads(prepared.read_text())
    rows = []
    for item in prepared['receipt']['candidates']:
        if item['status'] != 'compiled':
            continue
        binary = Path(item['binary'])
        host = binary.with_suffix('.host-preparation.json')
        report = freeze_host_preparation(config, batch, (*prefix, binary), host)
        rows.append(dict(rank=item['rank'], binary_sha256=item['binary_sha256'],
                         distinct_weight_buffers=len(report['weight_recipes']), allocation=report['allocation'],
                         host_preparation=str(host), host_preparation_sha256=sha(host)))
    if not rows:
        raise ValueError('prepared shortlist has no validated packing ABI')
    return rows
