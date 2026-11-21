"""Read packed-weight descriptors from a compiled shortlist without a GPU."""
import ctypes
import json
from pathlib import Path
from functools import lru_cache
import tempfile
from types import SimpleNamespace

from tilemega.bench.scheduler import write_json
from tilemega.build.identity import sha
from tilemega.serving.host_preparation import encode_schedule, prepare
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


@lru_cache(maxsize=4)
def prefill_schedule(path, binary_sha256, batch, sms, residency):
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    version = getattr(library, 'tm_plan_schedule_cache_version', None)
    if version is None or not version():
        return None
    if version() != 1 or sms <= 0 or residency <= 0:
        raise ValueError('unknown prefill schedule cache or incomplete target resources')
    export = library.tm_plan_export_schedule
    export.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    export.restype = ctypes.c_int
    with tempfile.TemporaryDirectory(prefix='tilemega-prefill-schedule-') as folder:
        file = Path(folder)/'schedule.bin'
        if export(batch, sms, residency, str(file).encode()):
            raise ValueError('prefill schedule export failed')
        return encode_schedule(file.read_bytes(), binary_sha256)


def freeze_host_preparation(config, batch, paths, out):
    plans = [read_plan(path) for path in paths]
    identities = [json.loads(Path(str(path)+'.identity.json').read_text()) for path in paths]
    for path, identity in zip(paths, identities, strict=True):
        if identity['binary_sha256'] != sha(path) or identity['manifest_sha256'] != sha(str(path)+'.plan.json'):
            raise ValueError(f'host preparation plan changed at {path}')
    record = prepare(config, batch, plans, identities, [buffers(path) for path in paths])
    if len(paths) == 2 and identities[0]['nvcc_arch'] == 'sm_80':
        sms = plans[0].get('target_spec', {}).get('resources', {}).get('num_sms', 0)
        plan_residency = plans[0].get('residency') or 1
        schedule = prefill_schedule(paths[0], identities[0]['binary_sha256'], batch, sms, plan_residency)
        if schedule is not None:
            record['prefill_schedule'] = schedule
    if out.exists():
        if json.loads(out.read_text()) != record:
            raise ValueError(f'host preparation changed at {out}')
    else:
        write_json(out, record)
    return record


def check_shortlist(prefill, prepared, config, batch):
    prefix = () if prefill == 'prompt' else (Path(prefill),)
    audit = prepared.parent/'candidate-coverage.jsonl'
    prepared = json.loads(prepared.read_text())
    rows = []
    for item in prepared['receipt']['candidates']:
        if item['status'] != 'compiled':
            continue
        binary = Path(item['binary'])
        host = binary.with_suffix('.host-preparation.json')
        coverage = dict(schema=1, check='linked_abi', rank=item['rank'],
                        coordinates_sha256=item['coordinates_sha256'], binary_sha256=item['binary_sha256'])
        try:
            report = freeze_host_preparation(config, batch, (*prefix, binary), host)
        except ValueError as error:
            if audit.is_file():
                with audit.open('a') as stream:
                    stream.write(json.dumps(dict(coverage, status='abi_incompatible', reason=str(error)))+'\n')
            raise
        if audit.is_file():
            with audit.open('a') as stream:
                stream.write(json.dumps(dict(coverage, status='abi_compatible'))+'\n')
        rows.append(dict(rank=item['rank'], binary_sha256=item['binary_sha256'],
                         distinct_weight_buffers=len(report['weight_recipes']), allocation=report['allocation'],
                         host_preparation=str(host), host_preparation_sha256=sha(host)))
    if not rows:
        raise ValueError('prepared shortlist has no validated packing ABI')
    return rows
