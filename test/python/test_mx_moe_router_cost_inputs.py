"""Router costs require exact source, build and actual device-query evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'experiments/mx5/moe_router_stage_bench.py'
spec = importlib.util.spec_from_file_location('router_cost', DRIVER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture(tmp_path):
    def blob(name, data):
        path = tmp_path / name
        path.write_bytes(data)
        return ref(path)
    def document(name, data):
        return blob(name, json.dumps(data).encode())
    resources = dict(registers=120, stack_bytes=0, static_smem_bytes=0)
    common = dict(threads=512, tn=32, wn=1, d=1, stream_k=1, dispatch_gate_tn=64,
        dispatch_down_tn=64, source_digest='a' * 64)
    mir = blob('mir.json', b'{}')
    kernel = dict(batch=1, status='CPU_resource_fit', resources=resources, MIR=mir)
    base = dict(common, tag='base', pw=0, binary=blob('base.so', b'base'), kernels=[kernel])
    build = dict(common, tag='selected', pw=1, binary=blob('selected.so', b'body'), kernels=[kernel])
    key = dict(common, tag='selected', pw=1, b=1, r_next=120, library=build['binary']['path'],
        binary_sha256=build['binary']['sha256'], build_record=document('build.json', build),
        r_next_reference=document('base.json', base))
    queries = [dict(tag=item['tag'], b=1, binary=item['binary'], registers=120, local_bytes=0,
        static_smem=0, resident=1, entry_returncode=0, fit=True) for item in (base, build)]
    legacy = blob('legacy.so', b'legacy')
    epoch = dict(kind='mx5_moe_router_source_bound_cost_epoch', final=False,
        R_cap=dict(t512=248, t576=220), driver=ref(DRIVER), legacy=legacy,
        compiler=blob('cucc', b'compiler'), packing_helper=blob('weights.py', b'packing'),
        root=str(tmp_path), inputs={}, query_records=[], eligible_keys=queries, source_digest='a' * 64)
    return epoch, key, Path(legacy['path'])


def validate(tmp_path, epoch, key, legacy):
    e, k = tmp_path / 'epoch.json', tmp_path / 'keys.json'
    e.write_text(json.dumps(epoch))
    k.write_text(json.dumps([key]))
    return module.validate_cost_inputs(e, k, legacy)


def test_exact_router_cost_request(tmp_path):
    epoch, key, legacy = fixture(tmp_path)
    result = validate(tmp_path, epoch, key, legacy)
    assert result['source_digest'] == epoch['source_digest']
    assert result['legacy_binary_sha256'] == epoch['legacy']['sha256']


@pytest.mark.parametrize('change', [
    lambda e, k: e['R_cap'].update(t576=160),
    lambda e, k: k.update(pw=0),
    lambda e, k: k.update(r_next=121),
    lambda e, k: k.update(binary_sha256='b' * 64),
    lambda e, k: k.update(source_digest='b' * 64),
    lambda e, k: e['eligible_keys'][-1].update(entry_returncode=8, fit=False),
    lambda e, k: e['eligible_keys'][-1].update(registers=121),
    lambda e, k: e['eligible_keys'][0].update(local_bytes=4),
])
def test_router_cost_rejects_changed_contract(tmp_path, change):
    epoch, key, legacy = fixture(tmp_path)
    change(epoch, key)
    with pytest.raises(ValueError):
        validate(tmp_path, epoch, key, legacy)


def test_router_cost_rejects_replaced_legacy_binary(tmp_path):
    epoch, key, legacy = fixture(tmp_path)
    legacy.write_bytes(b'replaced')
    with pytest.raises(ValueError, match='input changed'):
        validate(tmp_path, epoch, key, legacy)
