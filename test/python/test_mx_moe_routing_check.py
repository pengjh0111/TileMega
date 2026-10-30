"""CPU checks of exact expert sets and the specified BF16 boundary exception."""
import pytest

from tilemega.moe.native_routing_check import attribute_precursors, bf16_ulp, compare_routes


def logits(eighth=1.0, ninth=.984375):
    return [float(16 - i) for i in range(7)] + [eighth, ninth] + [-2.0] * 119


def test_expert_order_does_not_change_the_set_check():
    assert compare_routes(list(range(8)), list(reversed(range(8))), logits()) == {'status': 'match'}


@pytest.mark.parametrize('eighth,ninth,tie', [(1.0, .99609375, True), (1.0, .984375, True),
    (1.0, .98046875, False), (-1.0, -1.015625, True), (-1.0, -1.0234375, False)])
def test_only_the_two_bf16_ulp_boundary_is_exempt(eighth, ninth, tie):
    r = compare_routes(list(range(7)) + [8], list(range(8)), logits(eighth, ninth))
    assert r['status'] == ('tie' if tie else 'failure')
    assert (r['gap_in_bf16_ulp'] <= 2) == tie


@pytest.mark.parametrize('native', [list(range(7)), [0] * 8, list(range(7)) + [128]])
def test_malformed_native_experts_are_not_a_tie(native):
    with pytest.raises(ValueError, match='distinct expert'):
        compare_routes(native, list(range(8)), logits(1.0, 1.0))


def test_unrounded_reference_logits_are_not_silently_rounded():
    with pytest.raises(ValueError, match='exact BF16'):
        compare_routes(list(range(7)) + [8], list(range(8)), logits(1.0, .999))


def test_precursors_include_ties_and_stay_with_the_same_token():
    rows = [dict(step=0, layer=2, token=0, status='tie'),
            dict(step=0, layer=4, token=0, status='failure'),
            dict(step=1, layer=0, token=0, status='failure'),
            dict(step=1, layer=0, token=1, status='failure'),
            dict(step=1, layer=3, token=0, status='failure')]
    assert attribute_precursors(rows) == dict(precursor_lower_layer=2, precursor_earlier_step=1, none=1)
    assert rows[1]['precursor'] == 'precursor_lower_layer'
    assert rows[2]['precursor'] == 'precursor_earlier_step'
    assert rows[3]['precursor'] == 'none'
    assert rows[4]['precursor_lower_layer'] and rows[4]['precursor_earlier_step']


def test_bf16_ulp_uses_reference_magnitude_and_subnormal_spacing():
    import torch
    reference = torch.tensor([0., -0., 2.**-133, 1., -1., .5, 32.])
    expected = torch.tensor([2.**-133, 2.**-133, 2.**-133, 2.**-7, 2.**-7, 2.**-8, 2.**-2])
    assert torch.equal(bf16_ulp(reference), expected)


def test_named_capture_keeps_bf16_bits_and_validates_final_logits():
    import ctypes
    from types import SimpleNamespace
    import torch
    from tilemega.moe.native_routing_check import read_native_buffers

    expected = dict(route_debug=torch.arange(64 * 8).reshape(64, 1, 1, 8),
                    route_input=torch.tensor([0., -0., 1., -1.]).bfloat16().repeat(64, 1, 1, 1),
                    route_logits=torch.arange(64 * 64).bfloat16().reshape(64, 1, 1, 64),
                    route_weights=torch.full((64, 1, 1, 8), .125, dtype=torch.bfloat16))
    expected['router_logits'] = expected['route_logits'][-1, -1].clone()

    def read(_handle, name, target, size):
        value = expected[name.decode().removeprefix('mx.')].contiguous()
        if value.dtype != torch.bfloat16:
            value = value.to(torch.int32)
        assert size == value.numel() * value.element_size()
        ctypes.memmove(target, value.data_ptr(), size)
        return 0

    engine = SimpleNamespace(decode=SimpleNamespace(handle=None),
                             decode_lib=SimpleNamespace(lib=SimpleNamespace(tm_plan_read_buffer_named=read)))
    actual = read_native_buffers(engine, 1, 1, 4, 64)
    for name, value in expected.items():
        if value.dtype == torch.bfloat16:
            assert torch.equal(actual[name].view(torch.int16), value.view(torch.int16))
        else:
            assert torch.equal(actual[name], value)
    expected['router_logits'].zero_()
    with pytest.raises(ValueError, match='captured final logits'):
        read_native_buffers(engine, 1, 1, 4, 64)
