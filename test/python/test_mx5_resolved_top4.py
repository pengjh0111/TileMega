"""CPU synthetic consistency tests. No real table, compiler, GPU or approval.

Artifact cases reuse real receipt/ELF/MIR/query parsers with existing synthetic
native fixtures. Their origin/domain are injected explicitly; domain unit cases
separately exercise real row/reference/width validators against raw JSONL files.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pytest

ROOT=Path(os.environ.get('TILEMEGA_TEST_ROOT',Path(__file__).resolve().parents[2]))
spec=importlib.util.spec_from_file_location('resolved_artifact_fixtures',ROOT/'test/python/test_mx5_thread_exception.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
MODULE=Path(os.environ.get('TILEMEGA_RESOLVED_TOP4_MODULE',ROOT/'python/tilemega/mx5_resolved_top4.py'))
spec=importlib.util.spec_from_file_location('tilemega.mx5_resolved_top4',MODULE)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def save(path,value):
    return f.save(path,value)


class Fixture:
    def __init__(self,root,monkeypatch,threads=576):
        self.root=root;self.h=f.Fixture(root,monkeypatch);self.threads=threads
        self.preview=self.h.preview;self.preview['request'].update(threads=threads,grid=104)
        for candidate in self.preview['candidates']:
            candidate['pin']['mx_decode_v2']['threads']=threads
            for rows in candidate['evidence'].values():
                for row in rows:row['key']['threads']=threads
        self.domain=root/'CPU_SYNTHETIC_DOMAIN';self.domain.write_text('CPU synthetic domain identity, not executed')
        self.table=dict(schema_version=3,scope=dict(model={'capacity':1088},source_sha256={k:self.h.h.stage_sha for k in ('gemm','head','attention')}),
            rows=[row for rows in self.preview['candidates'][0]['evidence'].values() for row in rows],resource_rejections=[])
        self.preview.update(stage_table=save(root/'CPU_SYNTHETIC_table.json',self.table),status='predicted_top4_resource_review_required',input=save(root/'bridge.json',{'CPU_SYNTHETIC':True}))
        save(self.h.preview_file,self.preview);self.preview_ref=m.record(self.h.preview_file)
        monkeypatch.setattr(m,'validate_preview',lambda path,**kw:copy.deepcopy(json.loads(Path(path).read_text())))
        self.h.origin=(m.Qualified576Origin if threads==576 else m.Qualified512Origin).capture(self.preview_ref)
        self.h.h.origin=self.h.origin
        # Baseline-domain verification has independent actual-validator tests below.
        def domain(preview,table,tool):
            assert m.record(tool)==m.record(self.domain)
            return dict(table=preview['stage_table'],scope=table['scope'],resource_rejections=table['resource_rejections'],CPU_SYNTHETIC_DOMAIN_INJECTION=True)
        monkeypatch.setattr(m,'_complete_domain',domain)

    def ledger(self,index=0,*,mt=220,frame=0,query=True):
        ledger=m.legacy.PlanLedger.start(self.h.origin,index)
        bundle=self.h.artifact(index=index,mt=mt,frame=frame,query=query)
        node=m._load(bundle)
        from tilemega.mx5_plan_receipt import assemble
        receipt=assemble(node['native_receipt']['path'],ledger,stage_sources=self.h.h.sources,
            resources=node['resources']['path'],occupancy=node['query']['path'] if node['query'] else None)
        receipt_ref=save(self.root/('scoped%d.json'%index),receipt)
        ledger=ledger.append_attempt(receipt_ref['path'])
        assert not ledger.terminal_error,ledger.as_dict()
        path=self.root/('ledger%d.json'%index);ledger.write(path)
        return m.record(path),bundle

    def protocol576(self):
        plans=[]
        for index in range(4):
            proof,_=self.ledger(index)
            plans.append(m.capture_plan(self.preview_ref,index,proof))
        self.doc=m.capture_protocol(self.preview_ref,self.domain,plans)
        return self.write()

    def write(self):return save(self.root/'resolved.json',self.doc)
    def validate(self):return m.validate_resolved_top4(self.write()['path'],domain_tool=self.domain)


def test_all_four_576_bindings_are_review_only(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);path=h.protocol576();result=h.validate()
    assert result.action==m.Action.READY,result.as_dict()
    assert [p['candidate_id'] for p in result.data['plans']]==list(range(4))
    assert result.data['plans'][0]['original']==h.preview['candidates'][0]
    assert result.as_dict()['execution_authorized'] is False
    binary=result.data['plans'][0]['final']['binary']
    assert m.check_measurement_artifact(path,0,binary,domain_tool=h.domain).action==m.Action.READY


@pytest.mark.parametrize('mutation',[
    lambda d:d['plans'].pop(),
    lambda d:d['plans'][0].update(candidate_id=4),
    lambda d:d['plans'][1].update(candidate_id=True),
    lambda d:d['plans'].reverse(),
    lambda d:d['origin_candidates'][0]['ranks'].__setitem__(0,9),
    lambda d:d['origin_candidates'][0].update(predicted_tpot_ns=1),
    lambda d:d['origin_candidates'][0]['pin']['mx_decode_v2']['qkv'].update(prefetch=1),
    lambda d:d['plans'][0]['final']['effective_prediction'].update(effective_predicted_tpot_ns=0),
    lambda d:d.update(execution_authorized=True),
    lambda d:d['resource_budget_policy'].update(reset_on_thread_change=True),
    lambda d:d['resource_budget_policy'].update(maximum_total_reductions=6),
    lambda d:d['plans'][0].update(route='512_exception'),
])
def test_origin_mutations_missing_or_fifth_candidate_rejected(tmp_path,monkeypatch,mutation):
    h=Fixture(tmp_path,monkeypatch);h.protocol576();mutation(h.doc)
    assert h.validate().action==m.Action.BLOCKED


def test_measurement_artifact_swap_is_not_execution_permission(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);path=h.protocol576()
    swapped=h.doc['plans'][1]['final']['binary']
    result=m.check_measurement_artifact(path,0,swapped,domain_tool=h.domain)
    assert result.action==m.Action.BLOCKED and 'swap' in result.reason
    assert result.as_dict()['execution_authorized'] is False


@pytest.mark.parametrize('field',['binary','query','MIR_log','native_receipt'])
def test_final_sidecar_substitution_rejected(tmp_path,monkeypatch,field):
    h=Fixture(tmp_path,monkeypatch);h.protocol576()
    h.doc['plans'][0]['final'][field]=h.doc['plans'][1]['final'][field]
    assert h.validate().action==m.Action.BLOCKED


def test_saved_epoch_or_attempt_and_query_cannot_be_counterfeited(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);h.protocol576()
    proof=h.doc['plans'][0]['proof'];saved=m._load(proof)
    saved['attempts'][0]['verified']['epoch']='0'*64
    changed=save(tmp_path/'forged-ledger.json',saved);h.doc['plans'][0]['proof']=changed
    assert h.validate().action==m.Action.BLOCKED
    h.doc['plans'][0]['proof']=proof
    query=m._load(h.doc['plans'][0]['final']['query']);query['resident_ctas']=0
    Path(h.doc['plans'][0]['final']['query']['path']).write_text(json.dumps(query))
    assert h.validate().action==m.Action.BLOCKED


@pytest.mark.parametrize('kwargs',[dict(mt=228),dict(frame=8),dict(query=False)])
def test_nonfitting_or_unknown_query_never_becomes_final(tmp_path,monkeypatch,kwargs):
    h=Fixture(tmp_path,monkeypatch);proof,_=h.ledger(**kwargs)
    with pytest.raises(ValueError):m.capture_plan(h.preview_ref,0,proof)


def test_non_mt_frame_failure_is_not_a_width_retry(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch)
    ledger=m.legacy.PlanLedger.start(h.h.origin,0)
    bundle=h.h.artifact(index=0,shared=8,query=False);node=m._load(bundle)
    from tilemega.mx5_plan_receipt import assemble
    receipt=assemble(node['native_receipt']['path'],ledger,stage_sources=h.h.h.sources,resources=node['resources']['path'])
    scoped=save(tmp_path/'bad-shared.json',receipt);ledger=ledger.append_attempt(scoped['path'])
    assert m.legacy.next_plan_retry(ledger).kind==m.legacy.ActionKind.BLOCKED


def test_512_missing_d7_and_fitting576_never_approve512(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch,threads=512)
    minimum=h.h.minimum();final=h.h.artifact()
    audit=h.h.audit(minimum,[final]);assert audit.action==m.exception.Action.NEEDS_D7
    proof=save(tmp_path/'audit.json',audit.as_dict())
    with pytest.raises(ValueError):m.capture_plan(h.preview_ref,0,proof,artifact=final)
    fit=h.h.artifact(h.h.pin(minimum=True),mt=167,frame=0)
    veto=h.h.audit(fit,[]);assert veto.action==m.exception.Action.USE_576
    proof=save(tmp_path/'veto.json',veto.as_dict())
    with pytest.raises(ValueError):m.capture_plan(h.preview_ref,0,proof,artifact=final)


def test_real576_preview_is_required_for_fitting_minimum_reroute(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch,threads=512)
    fit=h.h.artifact(h.h.pin(minimum=True),mt=167,frame=0)
    veto=h.h.audit(fit,[]);proof=save(tmp_path/'veto.json',veto.as_dict())
    h.doc=m.capture_protocol(h.preview_ref,h.domain,[])
    h.doc.update(mode='use_576_solver_path',reroute=dict(candidate_id=0,audit=proof,true_576_preview=h.preview_ref))
    assert h.validate().action==m.Action.BLOCKED
    other=copy.deepcopy(h.preview);other['request']['threads']=576
    for candidate in other['candidates']:
        candidate['pin']['mx_decode_v2']['threads']=576
        candidate['predicted_tpot_ns']*=2
        candidate['predicted_stage_ns']={k:v*2 for k,v in candidate['predicted_stage_ns'].items()}
    h.doc['reroute']['true_576_preview']=save(tmp_path/'true576.json',other)
    result=h.validate()
    assert result.action==m.Action.USE_576,result.reason
    assert result.data['true_576_candidates'][0]['predicted_tpot_ns']==2*h.preview['candidates'][0]['predicted_tpot_ns']
    assert result.data['original_candidates'][0]==h.preview['candidates'][0]
    assert not result.as_dict()['execution_authorized']


def test_domain_row_raw_cost_prescribed_width_and_coverage(tmp_path,monkeypatch):
    # Exercise the actual table validators; only tiny CPU geometry enumeration is injected.
    h=f.f.Harness(tmp_path,monkeypatch,widths={n:16 for n in m.legacy.CLASSES})
    auth=dict(kind='plan_resource_retry',candidate_id=0,shared_class='qkv',to_pw=16)
    catalog=h.add_variant(types.SimpleNamespace(details=m.legacy.FrozenJSON.freeze(dict(authorization=auth))))
    row=json.loads(catalog.read_text())['rows'][0]['row']
    q=m._qualifier();index=q.base_id(row['kind'],row['key'])
    monkeypatch.setattr(q,'scoped_domain',lambda scope,domain:({index:16},{'qkv':[index]}))
    monkeypatch.setattr(m,'_qualifier',lambda:q)
    tool=tmp_path/'synthetic-domain';tool.write_text('CPU synthetic domain identity')
    coverage=dict(expected_keys=1,costed_keys=1,rejected_keys=0,missing_keys=0,by_class={'qkv':dict(legal=1,costed=1,rejected=0)})
    table=dict(rows=[row],resource_rejections=[],scope=dict(source_sha256={row['kind']:h.stage_sha}),coverage=coverage,
        r_next_policy_audit=dict(modern_normal_checked=0,legacy_cost_only_unverifiable=1),provenance=dict(artifacts=[m.record(tool)],tools=[m.record(q.__file__),m.record(q.table.__file__),m.record(q.table.compiler_probe.__file__)]))
    preview=dict(stage_table=save(tmp_path/'table.json',table))
    assert m._complete_domain(preview,table,tool)['coverage']==coverage
    bad=copy.deepcopy(table);bad['rows'][0]['r_next']=1
    with pytest.raises(ValueError):m._complete_domain(preview,bad,tool)
    bad=copy.deepcopy(table);bad['rows'][0]['median_ns']=0
    with pytest.raises(ValueError):m._complete_domain(preview,bad,tool)
    bad=copy.deepcopy(table);bad['rows'][0]['key']['prefetch']=1
    for sample in bad['rows'][0]['rounds']:
        sample['measurement']['key']['prefetch']=1
        raw=tmp_path/('wrong_width_round%d.jsonl'%sample['round']);raw.write_text(json.dumps(sample['measurement'])+'\n')
        sample['input']={**m.record(raw),'line':1}
    with pytest.raises(ValueError,match='resource_selection'):
        m._complete_domain(preview,bad,tool)
    bad=copy.deepcopy(table);bad['rows']=[]
    with pytest.raises(ValueError):m._complete_domain(preview,bad,tool)
    bad=copy.deepcopy(table);bad['rows'].append(copy.deepcopy(row))
    with pytest.raises(ValueError):m._complete_domain(preview,bad,tool)
    bad=copy.deepcopy(table);bad['scope']['source_sha256'][row['kind']]='0'*64
    with pytest.raises(ValueError):m._complete_domain(preview,bad,tool)


def d7(h,index,minimum,final):
    x=m.exception;mn=x._artifact(minimum,h.h.origin,index,minimum=True,require_query=True);fn=x._artifact(final,h.h.origin,index)
    diagnostic=h.h.artifact(h.h.pin(index,minimum=True),index=index,mt=233,frame=0,resident=0,extra=('-mllvm','--max-mtreg-number=255'))
    before=m._load(minimum);after=m._load(diagnostic)
    text='\n'.join((x.SPEC_DIFFERENCE,h.h.origin.sha256,mn['files']['binary']['sha256'],fn['files']['binary']['sha256'],
        'CPU SYNTHETIC before/after only; normal168/frame8 versus forced233/frame0/resident0. Not an executed observation or >227 lower bound.'))
    report=h.root/('CPU_SYNTHETIC_D7_%d.txt'%index);report.write_text(text)
    return save(h.root/('d7_%d.json'%index),dict(kind='mx5_candidate_d7_review',schema_version=1,origin_sha256=h.h.origin.sha256,candidate_id=index,
        family_sha256=mn['family'],spec_difference=x.SPEC_DIFFERENCE,minimum576_binary_sha256=mn['files']['binary']['sha256'],final512_binary_sha256=fn['files']['binary']['sha256'],
        assessment='normal_floor_576_still_spills_at_observed_168',not_claimed='required_registers_gt227_or_global_576_impossible',report=m.record(report),
        steps=[dict(kind='budget_diagnostic',analysis='CPU synthetic controlled flags consistency, no approval',before_native=before['native_receipt'],after_native=after['native_receipt'],
            source_evidence=[dict(source_path='include/tilemega/Backend/Mx/Fixture.h',before=m.record(h.h.h.header),after=m.record(h.h.h.header))],
            after_resources=after['resources'],after_query=after['query'])]))


def test_four_512_candidates_including_exact_cost_bound_child_stay_audit_only(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch,threads=512);plans=[]
    for index in range(4):
        minimum=h.h.artifact(h.h.pin(index,minimum=True),index=index,mt=168,frame=8)
        catalogs=[]
        if index==0:
            parent=h.h.artifact(index=index,mt=256,frame=8,query=False)
            next_=m.exception.next_width_diagnostic(h.h.origin,index,[parent]);assert next_.action==m.exception.Action.NEXT_WIDTH
            final=h.h.artifact(next_.data['next_pin'],index=index,ordinal=1)
            lineage=[parent,final];catalogs=[h.h.catalog(next_.data['authorization'])]
            missing=m.exception.audit_thread_exception(h.h.origin,index,minimum,lineage)
            assert missing.action==m.exception.Action.NEEDS_COSTS
        else:final=h.h.artifact(index=index);lineage=[final]
        report=d7(h,index,minimum,final)
        audit=m.exception.audit_thread_exception(h.h.origin,index,minimum,lineage,catalogs=catalogs,d7=report)
        assert audit.action==m.exception.Action.READY,audit.reason
        proof=save(tmp_path/('audit%d.json'%index),audit.as_dict())
        plans.append(m.capture_plan(h.preview_ref,index,proof,artifact=final))
    h.doc=m.capture_protocol(h.preview_ref,h.domain,plans);path=h.write();result=h.validate()
    assert result.action==m.Action.NEEDS_HISTORY,result.reason
    budget=result.data['plans'][0]['resource_budget']
    assert budget['verified_512_reductions']==1 and budget['verified_576_reductions'] is None
    assert budget['total_width_reductions'] is None and not budget['execution_budget_complete']
    binary=result.data['plans'][0]['final']['binary']
    assert m.check_measurement_artifact(path,0,binary,domain_tool=h.domain).action==m.Action.NEEDS_HISTORY
    truthful=copy.deepcopy(h.doc)
    h.doc['plans'][0]['final']['resource_budget'].update(verified_576_reductions=0,total_width_reductions=1,execution_budget_complete=True)
    assert h.validate().action==m.Action.BLOCKED
    h.doc=truthful;h.write()
    prediction=result.data['plans'][0]['final']['effective_prediction']
    assert prediction['origin_predicted_tpot_ns']==h.preview['candidates'][0]['predicted_tpot_ns']
    assert prediction['effective_predicted_tpot_ns']!=prediction['origin_predicted_tpot_ns']
    assert result.data['selection_anchor']==0 and not result.as_dict()['execution_authorized']
    # Rehashing the altered catalog cannot manufacture three matching actual rounds.
    proof=m._load(h.doc['plans'][0]['proof']);first_catalog=proof['details']['request']['catalogs'][0]
    doc=m._load(first_catalog);doc['rows'][0]['row']['median_ns']=1
    Path(first_catalog['path']).write_text(json.dumps(doc))
    assert m.validate_resolved_top4(path['path'],domain_tool=h.domain).action==m.Action.BLOCKED


def test_576_retry_replays_actual_parent_and_exact_three_fresh_costs(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);ledger=m.legacy.PlanLedger.start(h.h.origin,0)
    from tilemega.mx5_plan_receipt import assemble
    def append(bundle,name):
        nonlocal ledger
        node=m._load(bundle)
        receipt=assemble(node['native_receipt']['path'],ledger,stage_sources=h.h.h.sources,
            resources=node['resources']['path'],occupancy=node['query']['path'] if node['query'] else None)
        path=save(tmp_path/name,receipt);ledger=ledger.append_attempt(path['path'])
    append(h.h.artifact(mt=228,query=False),'parent.json')
    missing=m.legacy.next_plan_retry(ledger);assert missing.kind==m.legacy.ActionKind.NEEDS_STAGE_MEASUREMENTS
    catalog=m.legacy.VariantCatalog.load(h.h.catalog(missing.details.data['authorization'])['path'])
    action=m.legacy.next_plan_retry(ledger,catalog);assert action.kind==m.legacy.ActionKind.BUILD_REDUCED_PLAN
    ledger=ledger.commit_retry(action,catalog)
    append(h.h.artifact(ledger.pin.data,ordinal=1),'child.json')
    path=tmp_path/'ledger0.json';ledger.write(path)
    plans=[m.capture_plan(h.preview_ref,0,m.record(path))]
    for index in range(1,4):
        proof,_=h.ledger(index);plans.append(m.capture_plan(h.preview_ref,index,proof))
    h.doc=m.capture_protocol(h.preview_ref,h.domain,plans);result=h.validate()
    assert result.action==m.Action.READY,result.reason
    assert result.data['plans'][0]['resource_budget']['total_width_reductions']==1
    saved=m._load(plans[0]['proof']);saved['authorizations']*=4;saved['cost_bindings']*=4
    h.doc['plans'][0]['proof']=save(tmp_path/'fourth.json',saved)
    assert h.validate().action==m.Action.BLOCKED


def test_semantically_counterfeit_query_rehashed_through_ledger_is_rejected(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);proof,_=h.ledger()
    saved=m._load(proof);payload=saved['attempts'][0]['payload']
    query=m._load(payload['occupancy']);query['attributes']['numRegs']+=1
    payload['occupancy']=save(tmp_path/'counterfeit_query.json',query)
    receipt=save(tmp_path/'rehash_attempt.json',payload)
    # Legacy ledger checks occupancy status; the new final verifier must also
    # validate actual attribute bytes and binary/MIR values, even after rehashing.
    ledger=m.legacy.PlanLedger.start(h.h.origin,0).append_attempt(receipt['path'])
    assert m.legacy.next_plan_retry(ledger).kind==m.legacy.ActionKind.RESOURCE_FIT
    path=tmp_path/'rehash_ledger.json';ledger.write(path)
    with pytest.raises(ValueError,match='raw runtime attributes disagree'):
        m.capture_plan(h.preview_ref,0,m.record(path))


def test_512_parent_occupancy_failure_is_not_a_retry_authorization(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch,threads=512)
    minimum=h.h.minimum();parent=h.h.artifact(mt=256,frame=8,resident=0)
    next_=m.exception.next_width_diagnostic(h.h.origin,0,[parent])
    assert next_.action==m.exception.Action.NEXT_WIDTH
    final=h.h.artifact(next_.data['next_pin'],ordinal=1)
    catalogs=[h.h.catalog(next_.data['authorization'])]
    report=d7(h,0,minimum,final)
    audit=m.exception.audit_thread_exception(h.h.origin,0,minimum,[parent,final],catalogs=catalogs,d7=report)
    assert audit.action==m.exception.Action.READY,audit.reason
    proof=save(tmp_path/'parent_occ_failure.json',audit.as_dict())
    with pytest.raises(ValueError,match='occupancy failure'):
        m.capture_plan(h.preview_ref,0,proof,artifact=final)


def test_source_archive_memo_is_content_bound_and_rechecked(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);checked={};calls=[];actual=m.legacy._verify_archive
    def verify(*args):calls.append(args[0]['provenance']['source_sha256']);return actual(*args)
    monkeypatch.setattr(m.legacy,'_verify_archive',verify)
    artifact=dict(stage_sources=h.h.h.sources,source_files=h.h.h.source_files)
    m._bind_all_sources(h.table,artifact,checked);m._bind_all_sources(h.table,artifact,checked)
    assert len(calls)==1
    other=copy.deepcopy(artifact);other['source_files']['CPU_SYNTHETIC_epoch_marker']='0'*64
    m._bind_all_sources(h.table,other,checked);assert len(calls)==2
    path=Path(artifact['stage_sources'][0]['archive']['path']);path.write_bytes(path.read_bytes()+b'CPU corruption')
    with pytest.raises(ValueError):m._recheck_archives(checked)
