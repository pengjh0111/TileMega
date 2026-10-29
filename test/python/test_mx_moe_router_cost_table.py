"""Router aggregate times require actual source, resources and byte checks."""
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('router_table', ROOT / 'experiments/mx5/moe_router_cost_table.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    key = dict(threads=512, n=128, k=2048, b=1)
    epoch = dict(source_digest='a' * 64, driver=dict(sha256='b' * 64), legacy=dict(sha256='c' * 64))
    ref = dict(sha256='d' * 64)
    query = dict(entry_returncode=0, fit=True, local_bytes=0, static_smem=0, resident=1, registers=120)
    row = dict(key=key, round=1, PID=123, cost_protocol=True, source_bound=True, raw_sample_only=True,
        diagnostic_only=False, qualified=False, final=False, implementation='mx5_moe_router_v2',
        executor='mx_static_stage_bench', source_digest='a' * 64, epoch_sha256='d' * 64,
        driver_sha256='b' * 64, legacy_binary_sha256='c' * 64,
        resources=dict(MTreg=120, stack=0, static_smem=0, resident=1),
        FP32_reference_passed=True, legacy_topk_weights_bitwise=True, dispatch_verified=True,
        repeat_bitwise=True, comparison='raw_bytes_equal', copies=512,
        addressed_rotating_weight_bytes=256 * 1024**2, iterations=514, median_ns=100, p90_ns=101, spread_ns=2)
    return row, key, epoch, ref, query


def test_router_raw_contract():
    row, key, epoch, ref, query = fixture()
    module.validate_sample(row, key, epoch, ref, 123, 1, query)


@pytest.mark.parametrize('field,value', [('PID', 124), ('round', 2), ('source_bound', False),
    ('diagnostic_only', True), ('qualified', True), ('legacy_topk_weights_bitwise', False),
    ('comparison', 'float_value_equal'), ('median_ns', math.nan), ('p90_ns', 99),
    ('addressed_rotating_weight_bytes', 1), ('iterations', 512), ('legacy_binary_sha256', 'e' * 64)])
def test_rejects_prototypes_and_changed_samples(field, value):
    row, key, epoch, ref, query = fixture()
    row[field] = value
    with pytest.raises(ValueError):
        module.validate_sample(row, key, epoch, ref, 123, 1, query)


@pytest.mark.parametrize('field,value', [('entry_returncode', 8), ('fit', False), ('local_bytes', 4),
    ('static_smem', 4), ('resident', 2), ('registers', 256)])
def test_rejects_failed_actual_queries(field, value):
    row, key, epoch, ref, query = fixture()
    query[field] = value
    with pytest.raises(ValueError):
        module.validate_sample(row, key, epoch, ref, 123, 1, query)


@pytest.mark.parametrize('card', (0, 3, True, '7'))
def test_rejects_unpermitted_or_untyped_card_before_reading_inputs(tmp_path, card):
    with pytest.raises(ValueError, match='physical card'):
        module.aggregate(tmp_path / 'absent', physical_card=card)
