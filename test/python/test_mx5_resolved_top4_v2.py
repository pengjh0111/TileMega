"""CPU synthetic four-plan consumption; no device execution or actual approval."""
import copy
import importlib.util
import os
from pathlib import Path
import sys

import pytest

ROOT=Path(os.environ.get('TILEMEGA_TEST_ROOT',Path(__file__).resolve().parents[2]))
spec=importlib.util.spec_from_file_location('resolved_v2_d7_fixtures',Path(__file__).with_name('test_mx5_d7_v2.py'));f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
MODULE=Path(os.environ.get('TILEMEGA_RESOLVED_V2_MODULE',ROOT/'python/tilemega/mx5_resolved_top4_v2.py'))
spec=importlib.util.spec_from_file_location('tilemega.mx5_resolved_top4_v2',MODULE);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
m.d7=f.m;m.v1=f.m.base;m.history=f.m.history


def build(h):
    plans=[]
    for index in range(4):
        case=h.case(index);proof=f.save(case['path'],case['doc'])
        plans.append(m.capture_plan(h.h.origin_ref,index,domain_tool=h.domain,history_ref=case['history'],lineage512=case['lineage'],d7_ref=proof))
    doc=m.capture_protocol(h.h.origin_ref,h.domain,plans);return doc,f.save(h.root/'resolved_v2.json',doc)


def test_four_complete_512_d7_candidates_review_ready_but_v1_stays_blocked(tmp_path,monkeypatch):
    h=f.Fixture(tmp_path,monkeypatch);doc,path=build(h)
    result=m.validate_resolved_top4(path['path'],domain_tool=h.domain)
    assert result.action==m.v1.Action.READY,result.reason
    assert [x['candidate_id'] for x in result.data['plans']]==[0,1,2,3]
    assert result.data['selection_anchor']==0 and not result.as_dict()['execution_authorized']
    assert result.data['not_claimed']==f.m.I1_NOT_CLAIMED
    assert m.v1.validate_resolved_top4(path['path'],domain_tool=h.domain).action==m.v1.Action.BLOCKED
    for index in range(4):
        binary=result.data['plans'][index]['final']['binary']
        assert m.check_measurement_artifact(path,index,binary,domain_tool=h.domain).action==m.v1.Action.READY
    swapped=result.data['plans'][1]['final']['binary']
    assert m.check_measurement_artifact(path,0,swapped,domain_tool=h.domain).action==m.v1.Action.BLOCKED


@pytest.mark.parametrize('mutate',[
    lambda d:d['plans'].pop(),lambda d:d['plans'][0].update(candidate_id=4),
    lambda d:d['origin_candidates'][0]['ranks'].__setitem__(0,1),
    lambda d:d['plans'][0].update(route='576'),
    lambda d:d['plans'][0]['final']['resource_budget'].update(total_width_reductions=0,prescribed_576_history_verified=False),
    lambda d:d['resource_budget_policy'].update(reset_on_thread_change=True),
    lambda d:d['plans'][0]['final'].update(binary=d['plans'][1]['final']['binary']),
])
def test_four_id_rank_origin_budget_or_artifact_mutations_rejected(tmp_path,monkeypatch,mutate):
    h=f.Fixture(tmp_path,monkeypatch);doc,path=build(h);mutate(doc);f.save(Path(path['path']),doc)
    assert m.validate_resolved_top4(path['path'],domain_tool=h.domain).action==m.v1.Action.BLOCKED


def test_individually_bound_but_mixed_program_epochs_cannot_form_top4(tmp_path,monkeypatch):
    h=f.Fixture(tmp_path,monkeypatch);plans=[]
    for index in range(4):
        if index==3:
            actor=h.h.h.h.h
            marker=actor.source_root/'CPU_SYNTHETIC_epoch.py';marker.write_text('# CPU-only epoch marker\n')
            actor.source_files['CPU_SYNTHETIC_epoch.py']=f.m.record(marker)['sha256']
        case=h.case(index);proof=f.save(case['path'],case['doc'])
        plans.append(m.capture_plan(h.h.origin_ref,index,domain_tool=h.domain,history_ref=case['history'],lineage512=case['lineage'],d7_ref=proof))
    doc=m.capture_protocol(h.h.origin_ref,h.domain,plans);path=f.save(tmp_path/'mixed.json',doc)
    result=m.validate_resolved_top4(path['path'],domain_tool=h.domain)
    assert result.action==m.v1.Action.BLOCKED and 'epochs' in result.reason,result.reason


def test_fitting_canonical576_needs_real_576_origin_and_never_copies512_prediction(tmp_path,monkeypatch):
    h=f.Fixture(tmp_path,monkeypatch);primary=h.h.derive();pin=primary.data['canonical_primary_pin']
    attempt=h.h.attempt(pin,mt=167,frame=0);fit=h.h.audit([attempt]);assert fit.action==m.history.Action.USE_576
    proof=f.save(tmp_path/'fits576.json',fit.as_dict());preview=h.h.h.preview;other=copy.deepcopy(preview);other['request']['threads']=576
    for candidate in other['candidates']:
        candidate['pin']['mx_decode_v2']['threads']=576
        candidate['predicted_tpot_ns']*=2
        candidate['predicted_stage_ns']={k:v*2 for k,v in candidate['predicted_stage_ns'].items()}
    true_ref=f.save(tmp_path/'true576.json',other)
    doc=m.capture_protocol(h.h.origin_ref,h.domain,[]);doc.update(mode='use_576_solver_path',reroute=dict(candidate_id=0,history=proof,true_576_preview=true_ref))
    path=f.save(tmp_path/'reroute.json',doc);result=m.validate_resolved_top4(path['path'],domain_tool=h.domain)
    assert result.action==m.v1.Action.USE_576,result.reason
    assert result.data['true_576_candidates'][0]['predicted_tpot_ns']==2*result.data['original_candidates'][0]['predicted_tpot_ns']
    doc['reroute']['true_576_preview']=h.h.origin_ref;f.save(Path(path['path']),doc)
    assert m.validate_resolved_top4(path['path'],domain_tool=h.domain).action==m.v1.Action.BLOCKED


def test_genuine_576_ledger_route_is_preserved_in_v2_without_thread_origin_mix(tmp_path,monkeypatch):
    h=f.can.f.Fixture(tmp_path,monkeypatch,threads=576);h.protocol576()
    doc=m.capture_protocol(h.preview_ref,h.domain,h.doc['plans']);path=f.save(tmp_path/'576v2.json',doc)
    result=m.validate_resolved_top4(path['path'],domain_tool=h.domain)
    assert result.action==m.v1.Action.READY,result.reason
    assert all(x['final']['threads']==576 for x in result.data['plans'])
    assert not result.as_dict()['execution_authorized']
