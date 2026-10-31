"""Independent exhaustive ranking and exact MX-6 measurement selection rules."""
import itertools
import math
import pytest
from tilemega.moe.selection_v3 import STAGES,attention_mean,k_best,select_measured


def test_k_best_matches_exhaustive_including_ties():
    lists={name:[dict(ns=x)for x in values]for name,values in zip(STAGES,
        ([2,3,3],[1,4],[1,2],[2,3],[7,9],[4,5],[10,11]))}
    layers=16;factors=[layers]*6+[1]
    expected=sorted((sum(lists[name][r]['ns']*factors[i]for i,(name,r)in enumerate(zip(STAGES,ranks))),ranks)
        for ranks in itertools.product(*[range(len(lists[name]))for name in STAGES]))
    actual=[(row['predicted_ns'],tuple(row['ranks'][name]for name in STAGES))for row in k_best(lists,layers)]
    assert actual==expected


def test_attention_mean_averages_exact_decode_lengths():
    assert attention_mean([65,256,512,768,1088])==576
    assert attention_mean([17]*5)==17
    with pytest.raises(ValueError):attention_mean([1]*4)
    with pytest.raises(ValueError):attention_mean([1,2,float('nan'),4,5])


@pytest.mark.parametrize('fastest,selected',[(99.01,0),(99.,2),(98.99,2)])
def test_one_percent_rule_is_inclusive(fastest,selected):
    result=select_measured([100,101,102,103],[[99,101],[100,102],[fastest-.1,fastest+.1],[102,104]])
    assert result['selected']==selected
    assert result['measured_ns'][2]==fastest


def test_exact_four_and_two_fresh_measurements_required():
    with pytest.raises(ValueError):select_measured([1]*3,[[1,1]]*3)
    with pytest.raises(ValueError):select_measured([1]*4,[[1,1],[1],[1,1],[1,1]])
    with pytest.raises(ValueError):select_measured([1]*4,[[1,math.inf]]*4)
