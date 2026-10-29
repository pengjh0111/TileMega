"""New MoE shapes cannot borrow a different regular-stage query or R_next."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('regular_cost', ROOT / 'experiments/mx5/moe_regular_cost_measure.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(kind):
    if kind == 'gemm':
        key = dict(n=5120, k=2048, b=1, tn=32, wn=1, depth=1, threads=512, prefetch=1, op=0, norm=1, part=0)
    else:
        key = dict(dim=128, q=8, b=1, hkv=4, length=65, b_min=1, qk_norm=1, capacity=1088, threads=512, prefetch_kind=0)
    qkey = dict(key, library=dict(path='/body.so', sha256='b' * 64), source_sha256='a' * 64)
    query = dict(key=qkey, registers=200, local_bytes=0, static_smem=0, resident=1, returncode=0)
    reference = copy.deepcopy(query)
    reference['key']['prefetch' if kind == 'gemm' else 'prefetch_kind'] = 0
    reference['registers'] = 180
    key.update(kind=kind, library='/body.so', binary_sha256='b' * 64, r_next=180)
    return key, query, reference


@pytest.mark.parametrize('kind', ['gemm', 'attention'])
def test_exact_new_shape_queries(kind):
    key, query, reference = fixture(kind)
    module.validate_key(key, query, reference, 'a' * 64)


@pytest.mark.parametrize('kind', ['gemm', 'attention'])
@pytest.mark.parametrize('change', [
    lambda k, q, r: q['key'].update(b=16),
    lambda k, q, r: r['key'].update(threads=576),
    lambda k, q, r: r['key'].update(source_sha256='c' * 64),
    lambda k, q, r: k.update(binary_sha256='c' * 64),
    lambda k, q, r: k.update(r_next=181),
    lambda k, q, r: q.update(local_bytes=4),
    lambda k, q, r: q.update(resident=2),
    lambda k, q, r: q.update(registers=256),
])
def test_new_shape_costs_reject_mismatched_evidence(kind, change):
    key, query, reference = fixture(kind)
    change(key, query, reference)
    with pytest.raises(ValueError):
        module.validate_key(key, query, reference, 'a' * 64)


@pytest.mark.parametrize('kind', ['gemm', 'attention'])
def test_failed_zero_prefetch_reference_is_retained_without_rejecting_the_payload(kind):
    key, query, reference = fixture(kind)
    reference['local_bytes'] = 32
    module.validate_key(key, query, reference, 'a' * 64)
