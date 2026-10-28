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


@pytest.fixture
def history_fixture(tmp_path, monkeypatch):
    import json
    def save(path, value):
        path.write_text(json.dumps(value) + '\n')
        return m.record(path)
    original = pin()
    proposal = m.proposed_width(original, 0)
    preview = tmp_path/'CPU_SYNTHETIC_PREVIEW.json'
    preview_ref = save(preview, {'fixture': True})
    table = tmp_path/'CPU_SYNTHETIC_TABLE.json'
    table_ref = save(table, {'fixture': True})
    sources = tmp_path/'CPU_SYNTHETIC_SOURCES.json'
    save(sources, [])
    domain = tmp_path/'CPU_SYNTHETIC_DOMAIN'
    domain.write_text('not an executable\n')
    class Origin:
        sha256 = 'CPU_SYNTHETIC_ORIGIN'
        data = {'preview': {'stage_table': table_ref}, 'layers': 2}
        def candidate(self, _):
            return {'pin': original, 'predicted_stage_ns': {n:100 for n in m.policy.STAGES},
                    'evidence': {n:[{'kind':'head' if n == 'head' else 'gemm', 'key':{}}]
                                 for n in m.policy.CLASSES}}
    monkeypatch.setattr(m.resolved.Qualified576Origin, 'capture', lambda _: Origin())
    monkeypatch.setattr(m.resolved, '_complete_domain', lambda *args: None)
    monkeypatch.setattr(m.policy, 'program_epoch', lambda _: 'CPU_SYNTHETIC_EPOCH')
    observations = []
    chain = []
    for index, p in enumerate([original, proposal['next_pin']]):
        fake = dict(native=preview_ref, resources=preview_ref, files={}, snapshot={'source_files':{}})
        value = dict(origin_preview=preview_ref, candidate_id=0, pin=copy.deepcopy(p), used_reductions=index,
                     failed_resources=index == 0, proposal=copy.deepcopy(proposal) if index == 0 else None,
                     WD0=copy.deepcopy(fake), WD1=copy.deepcopy(fake))
        observations.append(value)
        chain.append(dict(step=index, observation=save(tmp_path/f'pair{index}.json', value)))
    monkeypatch.setattr(m, 'inspect_pair', lambda *args: copy.deepcopy(observations[args[-1]]))
    doc = dict(original_preview=preview_ref, candidate=0, chain=chain, used_reductions=1,
               changed_width_requests=[dict(parent_pair=chain[0]['observation'], proposal=proposal)],
               pair_resources_fit=True)
    path = tmp_path/'history.json'
    save(path, doc)
    return path, domain, sources, doc, observations, save


def test_history_replay_never_grants_execution_without_queries_and_costs(history_fixture):
    path, domain, sources, *_ = history_fixture
    result = m.replay_history(path, domain, sources)
    assert result['missing_queries'] == ['WD0', 'WD1']
    assert [r['stage'] for r in result['missing_costs']] == ['head']
    assert not result['resource_cost_query_ready'] and not result['execution_authorized']


@pytest.mark.parametrize('mutation,match', [
    ('reset', 'shared reduction count'), ('gap', 'noncontiguous'),
    ('wrong_pin', 'prescribed pin'), ('requests', 'width requests')])
def test_history_rejects_reset_gap_pin_and_unwitnessed_requests(history_fixture, mutation, match):
    path, domain, sources, doc, observations, save = history_fixture
    if mutation == 'reset':
        doc['used_reductions'] = 0
    elif mutation == 'gap':
        doc['chain'][1]['step'] = 2
    elif mutation == 'requests':
        doc['changed_width_requests'] = []
    else:
        observations[1]['pin']['mx_decode_v2']['gate_up']['prefetch'] = 2
        doc['chain'][1]['observation'] = save(path.parent/'changed_pair.json', observations[1])
    save(path, doc)
    with pytest.raises(ValueError, match=match):
        m.replay_history(path, domain, sources)


@pytest.mark.parametrize('resident', [0, 1])
def test_history_uses_verified_query_summary_without_inventing_raw_status(history_fixture, monkeypatch, resident):
    path, domain, sources, doc, observations, save = history_fixture
    q = save(path.parent/'CPU_SYNTHETIC_QUERY.json', {'not_a_device_query': True})
    # The trusted helper verifies raw API status, then returns this smaller summary.
    monkeypatch.setattr(m.resolved.exception, '_query', lambda *args: {'resident_ctas': resident})
    if resident == 0:
        with pytest.raises(ValueError, match='actual occupancy'):
            m.replay_history(path, domain, sources, queries={'WD0':q,'WD1':q})
    else:
        result = m.replay_history(path, domain, sources, queries={'WD0':q,'WD1':q})
        assert result['missing_queries'] == []
        assert result['missing_costs'] and result['execution_authorized'] is False
