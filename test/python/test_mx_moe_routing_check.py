"""CPU checks of exact expert sets and the specified BF16 boundary exception."""
import pytest

from tilemega.moe.routing_check import compare_routes


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
