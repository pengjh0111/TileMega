"""A MoE top-four candidate owns its matching prompt/decode weight recipes."""
import copy
import json
from pathlib import Path

import pytest

from tilemega.moe.plan_pair import validate_plan_pair

ROOT = Path(__file__).resolve().parents[2]


def pair():
    common = dict(program_kind='moe', batch_lo=1, batch_hi=1, capacity=1088, kv_layout='vt',
        kv_storage='block_fragments', layers=24, stages_per_step=145, experts=128, top_k=8,
        model='qwen3_30b_a3b_l24', head_dim=128, heads_kv=4, q_per_kv=8, watchdog=0,
        buffers=[dict(name='w' + str(i), role='external', dtype='bf16', constant=32, per_batch=0,
            recipe=dict(kind='tile_pages', tile_n=64)) for i in range(246)])
    prefill = dict(copy.deepcopy(common), phase='prefill', executor='mx_static_prompt', seq=64, prompt_barriers=9217)
    decode = dict(copy.deepcopy(common), phase='decode', executor='mx_static', seq=1, prompt_barriers=0)
    return prefill, decode


def test_matching_candidate_pair():
    prefill, decode = pair()
    result = validate_plan_pair(prefill, decode, 1, expected_watchdog=0)
    assert result['shared_binding_contract'] and result['external_buffers'] == 246
    assert result['not_correctness_or_solver_acceptance']


@pytest.mark.parametrize('change', [
    lambda p, d: p.update(executor='legacy'),
    lambda p, d: p.update(seq=1),
    lambda p, d: p.update(batch_hi=16),
    lambda p, d: p.update(kv_storage='row_major'),
    lambda p, d: p.update(heads_kv=8),
    lambda p, d: p.update(prompt_barriers=9216),
    lambda p, d: p['buffers'][0]['recipe'].update(tile_n=32),
    lambda p, d: p['buffers'][0].update(dtype='f32'),
    lambda p, d: p['buffers'][0].update(constant=64),
    lambda p, d: p['buffers'].pop(),
    lambda p, d: p['buffers'][0].update(name=p['buffers'][1]['name']),
])
def test_candidate_cannot_borrow_a_mismatched_prompt(change):
    prefill, decode = pair()
    change(prefill, decode)
    with pytest.raises(ValueError):
        validate_plan_pair(prefill, decode, 1)


def test_watchdog_contract_is_explicit():
    prefill, decode = pair()
    prefill['watchdog'] = 1
    with pytest.raises(ValueError, match='watchdog'):
        validate_plan_pair(prefill, decode, 1, expected_watchdog=0)


@pytest.mark.parametrize('batch,watchdog', [(1, 0), (16, 0), (1, 1), (16, 1)])
def test_actual_native_control_pairs(batch, watchdog):
    directory = ROOT / 'runs/mx5' / ('moe_native_cli_cpu_v1' if watchdog == 0 else 'moe_manual_watchdog_controls_cpu_v1')
    p = directory / f'B{batch}_p1/plan.so.plan.json'
    d = directory / f'B{batch}_p0/plan.so.plan.json'
    if not p.is_file() or not d.is_file():
        pytest.skip('actual native control plans are absent')
    result = validate_plan_pair(json.loads(p.read_text()), json.loads(d.read_text()), batch, expected_watchdog=watchdog)
    assert result['shared_binding_contract']
