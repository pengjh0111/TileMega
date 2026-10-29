"""Regular aggregates preserve the actual frozen dense-driver contracts."""
import copy
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('regular_table', ROOT / 'experiments/mx5/moe_regular_cost_table.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(kind):
    if kind == 'gemm':
        key = dict(n=5120, k=2048, b=1, tn=32, wn=1, depth=1, threads=512, prefetch=1, op=0, norm=1, part=0)
    else:
        key = dict(dim=128, q=8, b=1, hkv=4, length=65, b_min=1, qk_norm=1, capacity=1088, threads=512, prefetch_kind=0)
    qkey = dict(key, source_sha256='a' * 64, library=dict(path='/body.so', sha256='b' * 64))
    query = dict(key=qkey, registers=200, local_bytes=0, static_smem=0, resident=1, returncode=0)
    reference = copy.deepcopy(query)
    reference['key']['prefetch' if kind == 'gemm' else 'prefetch_kind'] = 0
    reference['registers'] = 180
    key.update(kind=kind, library='/body.so', binary_sha256='b' * 64, r_next=180)
    row = dict(key=key, round=1, executor='mx_static_stage_bench', implementation='mx5_' + kind + '_v2',
        source_sha256='a' * 64, binary_sha256='b' * 64, threads=512,
        mtreg=200, local_bytes=0, static_smem_bytes=0, resident_ctas=1,
        copies=512, iterations=514, working_set_bytes=256 * 1024**2, median_ns=100, p90_ns=101)
    row['prefetch' if kind == 'gemm' else 'prefetch_kind'] = key['prefetch' if kind == 'gemm' else 'prefetch_kind']
    if kind == 'gemm':
        row.update(reference_backend='torch_cpu', reference_precision='float32', reference_oracle_version='gemm_cpu_fp32_v1')
    else:
        row['validation'] = dict(context_atol=.006, context_rtol=.016, new_k_atol=.016, new_k_rtol=.016,
            historical_kv_bits=True, repeated_context_bits=True, barrier_and_LA_counts=True)
    return row, key, dict(source_sha256={kind: 'a' * 64}), query, reference


@pytest.mark.parametrize('kind', ['gemm', 'attention'])
def test_actual_regular_driver_fields(kind):
    row, key, epoch, query, reference = fixture(kind)
    module.validate_sample(row, key, epoch, 1, query, reference)


@pytest.mark.parametrize('kind', ['gemm', 'attention'])
@pytest.mark.parametrize('field,value', [('round', 2), ('implementation', 'legacy'), ('mtreg', 201),
    ('binary_sha256', 'c' * 64), ('local_bytes', 4), ('working_set_bytes', 1),
    ('iterations', 512), ('median_ns', math.nan), ('p90_ns', 99)])
def test_regular_aggregate_rejects_altered_measurements(kind, field, value):
    row, key, epoch, query, reference = fixture(kind)
    row[field] = value
    with pytest.raises(ValueError):
        module.validate_sample(row, key, epoch, 1, query, reference)


def test_regular_gemm_cannot_change_its_original_oracle():
    row, key, epoch, query, reference = fixture('gemm')
    row['reference_backend'] = 'torch_gpu'
    with pytest.raises(ValueError, match='oracle'):
        module.validate_sample(row, key, epoch, 1, query, reference)


def test_regular_attention_cannot_relax_validation():
    row, key, epoch, query, reference = fixture('attention')
    row['validation']['context_atol'] = .1
    with pytest.raises(ValueError, match='validation'):
        module.validate_sample(row, key, epoch, 1, query, reference)
