"""Cost qualification rejects incomplete scopes and malformed raw samples."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('moe_cost_table', ROOT / 'experiments/mx5/moe_grouped_cost_table.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample():
    key = dict(n=2048, k=768, b=1, n_active=8)
    epoch = dict(source_digest='a' * 64, driver=dict(sha256='b' * 64))
    ref = dict(sha256='c' * 64)
    row = dict(key=key, round=1, PID=123, cost_protocol=True, source_bound=True, raw_sample_only=True,
        final=False, implementation='mx5_moe_grouped_v2', executor='mx_static_stage_bench',
        source_digest=epoch['source_digest'], epoch_sha256=ref['sha256'], driver_sha256=epoch['driver']['sha256'],
        resources=dict(MTreg=120, stack=0, static_smem=0, resident=1), reference_passed=True, repeat_bitwise=True,
        n_active=8, row_counts=[1] * 8, copies=11, addressed_rotating_weight_bytes=11 * 8 * 2048 * 768 * 2,
        iterations=128, median_ns=100, p90_ns=101, spread_ns=2)
    return row, key, epoch, ref


def test_sample_protocol_accepts_only_the_explicit_raw_contract():
    row, key, epoch, ref = sample()
    module.validate_sample(row, key, epoch, ref, 123, 1, 120)


@pytest.mark.parametrize('field,value', [('PID', 124), ('round', 2), ('source_bound', False),
    ('cost_protocol', False), ('repeat_bitwise', False), ('median_ns', math.nan),
    ('p90_ns', 99), ('addressed_rotating_weight_bytes', 1), ('copies', 128), ('iterations', 11)])
def test_bad_samples_are_not_aggregated(field, value):
    row, key, epoch, ref = sample()
    row[field] = value
    with pytest.raises(ValueError):
        module.validate_sample(row, key, epoch, ref, 123, 1, 120)


def test_missing_batches_never_produce_qualified_costs(tmp_path):
    scope = ROOT / 'runs/mx5/moe_grouped_cost_scope_v1'
    inputs = scope / 'ACTUAL_SOURCE_BOUND_COST_INPUTS_CPU_VALIDATED.json'
    if not inputs.is_file():
        pytest.skip('requires the actual source-bound grouped cost epoch')
    (tmp_path / inputs.name).write_bytes(inputs.read_bytes())
    with pytest.raises(ValueError, match='incomplete'):
        module.aggregate(tmp_path, require_complete=True)
    prefix = module.aggregate(tmp_path, require_complete=False)
    assert not prefix['qualified'] and not prefix['complete_scope'] and not prefix['whole_grouped_domain_complete']
    assert not prefix['full_model_domain'] and not prefix['target_defaults_changed'] and not prefix['body_bitwise_gate']
    assert prefix['rows'] == [] and prefix['coverage']['missing_scope_keys'] == 772


def recovered_branch(tmp_path, mutate=None):
    binary = tmp_path / 'body.so'
    binary.write_bytes(b'fixture binary')
    common = dict(threads=576, gate_up=0, tn=16, wn=8, d=1, source_digest='a' * 64,
        binary=module.record(binary), r_next=100)
    primary = dict(common, pw=16, resources=dict(registers=168, stack_bytes=16, static_smem_bytes=0))
    selected = dict(common, pw=8, resources=dict(registers=160, stack_bytes=0, static_smem_bytes=0))
    if mutate:
        mutate(primary, selected)
    path = tmp_path / 'primary.json'
    path.write_text(json.dumps(primary))
    case = dict(case=dict(primary_not_tried=False, prior=[module.record(path)], pw=16, d=1),
        attempts=[selected], disposition='reduced_prefetch_fit')
    closure = tmp_path / 'closure.json'
    closure.write_text(json.dumps(dict(source_digest='a' * 64, cases=[case])))
    key = dict(resource_closure=module.record(closure), pw=8, r_next=100)
    return key, selected


def test_recovered_cost_retains_the_failed_primary_and_prescribed_halving(tmp_path):
    key, build = recovered_branch(tmp_path)
    branch = module.resource_branch(key, build)
    assert branch['selected_step'] == 1
    assert [attempt['pw'] for attempt in branch['attempts']] == [16, 8]
    assert branch['attempts'][0]['resources']['stack_bytes'] == 16


@pytest.mark.parametrize('mutation', [
    lambda first, last: first['resources'].update(stack_bytes=0),
    lambda first, last: last.update(pw=4),
    lambda first, last: first.update(source_digest='b' * 64),
    lambda first, last: last['resources'].update(stack_bytes=4),
])
def test_recovered_cost_cannot_skip_a_fitting_primary_or_change_the_branch(tmp_path, mutation):
    key, build = recovered_branch(tmp_path, mutation)
    with pytest.raises(ValueError):
        module.resource_branch(key, build)
