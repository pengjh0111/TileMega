"""CPU policy tests; synthetic observations do not represent device execution."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('mx5_watchdog_pairs', ROOT/'experiments/mx5/watchdog_pairs.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def pin():
    c = {n: dict(tn=32, wn=1, d=2, prefetch=2, r_next=100, part='kTileRange')
         for n in m.policy.CLASSES}
    c.update(threads=576, b_min=4, attn_prefetch=2)
    c['gate_up']['prefetch'] = 4
    c['head']['prefetch'] = 8
    return dict(mx_decode_v2=c)


def test_prescribed_sequence_and_shared_cap():
    initial = pin()
    first = m.proposed_width(initial, 0)
    assert (first['shared_class'], first['from_pw'], first['to_pw']) == ('head', 8, 4)
    second = m.proposed_width(first['next_pin'], 1)
    assert second['shared_class'] == 'gate_up'  # equal widths: specified class order
    third = m.proposed_width(second['next_pin'], 2)
    assert (third['shared_class'], third['to_pw']) == ('head', 2)
    assert m.proposed_width(third['next_pin'], 3) is None
    assert initial == pin()
    for name in m.policy.CLASSES:
        assert {k:v for k,v in third['next_pin']['mx_decode_v2'][name].items() if k != 'prefetch'} == {
            k:v for k,v in initial['mx_decode_v2'][name].items() if k != 'prefetch'}


@pytest.mark.parametrize('used', [-1, 4, True, 1.0])
def test_bad_budget_rejected(used):
    with pytest.raises(ValueError, match='shared reduction'):
        m.proposed_width(pin(), used)


def test_floor_and_no512_transition():
    p = pin()
    for name in m.policy.CLASSES:
        p['mx_decode_v2'][name]['prefetch'] = p['mx_decode_v2'][name]['d']
    assert m.proposed_width(p, 0) is None
    p['mx_decode_v2']['threads'] = 512
    with pytest.raises(ValueError, match='512'):
        m.proposed_width(p, 0)


@pytest.fixture
def pair_fixture(tmp_path, monkeypatch):
    p = pin()
    preview = tmp_path/'CPU_SYNTHETIC_PREVIEW.json'
    preview.write_text('{}\n')
    class Origin:
        def candidate(self, _):
            return {'pin': p}
    monkeypatch.setattr(m.resolved.Qualified576Origin, 'capture', lambda _: Origin())
    observations = [dict(snapshot={'source_digest':'synthetic'},
                         files={'generated_source':{'sha256':'synthetic'},'source_patch':{'sha256':'synthetic'}}, failed_resources=False),
                    dict(snapshot={'source_digest':'synthetic'},
                         files={'generated_source':{'sha256':'synthetic'},'source_patch':{'sha256':'synthetic'}}, failed_resources=True)]
    monkeypatch.setattr(m, '_normal', lambda *args: copy.deepcopy(observations[args[-2]]))
    return preview, p, observations


def test_wd1_failure_is_not_hidden_by_wd0_fit(pair_fixture):
    preview, p, _ = pair_fixture
    result = m.inspect_pair(preview, 0, p, {'native':0,'resources':0}, {'native':0,'resources':0}, [])
    assert result['failed_resources']
    assert result['proposal']['shared_class'] == 'head'
    assert result['execution_authorized'] is False
    assert result['action'] == 'needs_actual_pair_history_and_exact_width_costs'


@pytest.mark.parametrize('field', ['snapshot', 'files'])
def test_pair_source_or_generated_code_substitution_rejected(pair_fixture, field):
    preview, p, observations = pair_fixture
    observations[1][field] = ({'source_digest':'changed'} if field == 'snapshot' else
                             {'generated_source':{'sha256':'changed'},'source_patch':{'sha256':'synthetic'}})
    with pytest.raises(ValueError, match='differ'):
        m.inspect_pair(preview, 0, p, {'native':0,'resources':0}, {'native':0,'resources':0}, [])


def test_both_fit_still_requires_actual_queries(pair_fixture):
    preview, p, observations = pair_fixture
    observations[1]['failed_resources'] = False
    result = m.inspect_pair(preview, 0, p, {'native':0,'resources':0}, {'native':0,'resources':0}, [])
    assert result['action'] == 'needs_both_actual_queries'
    assert result['proposal'] is None and result['execution_authorized'] is False
