"""CPU synthetic D7 evidence, not real table/query/measurement/approval."""
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT=Path(os.environ.get('TILEMEGA_TEST_ROOT',Path(__file__).resolve().parents[2]))
spec=importlib.util.spec_from_file_location('d7_canonical_fixtures',ROOT/'test/python/test_mx5_canonical576_history.py');can=importlib.util.module_from_spec(spec);spec.loader.exec_module(can)
MODULE=Path(os.environ.get('TILEMEGA_D7_V2_MODULE',ROOT/'python/tilemega/mx5_d7_v2.py'))
spec=importlib.util.spec_from_file_location('tilemega.mx5_d7_v2',MODULE);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
m.history=can.m;m.base=can.m.resolved;m.old=can.m.old


def save(path,value):return can.save(path,value)
def whole(ref):return m.span(ref,1,len(m.check_file(ref).read_bytes().splitlines()))


def baseline(root):
    folder=root/'CPU_SYNTHETIC_I1';folder.mkdir();source=folder/'stage_bench.cu';source.write_text('// CPU synthetic baseline caller; never compiled\n')
    name='experiments/mx4/stage_bench.cu';sha=hashlib.sha256(name.encode()+b'\0'+source.read_bytes()+b'\0').hexdigest()
    provenance=save(folder/'provenance.json',dict(base_head='CPU_SYNTHETIC',implementations={'mx4_instruction_order':dict(sources=[name],source_sha256=sha)}))
    rounds=[]
    for number in range(1,6):
        path=folder/('round%d.csv'%number)
        with path.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['probe','model','batch','class','mode','median_ns','p90_ns','iterations']);writer.writeheader()
            for mode in ('repeat','program_order'):
                for cls in ('qkv','attn','o','gate_up','down'):
                    writer.writerow(dict(probe='I1',model='qwen3_0_6b',batch=1,**{'class':cls},mode=mode,median_ns=100+number+(10 if mode=='program_order' else 0),p90_ns=150+number,iterations=500))
        rounds.append(m.record(path))
    run=folder/'run.log';run.write_text(''.join('round=%d complete\n'%n for n in range(1,6)))
    journal=folder/'guard.jsonl';journal.write_text(json.dumps(dict(phase='after',exit_code=0))+'\n')
    measured=folder/'CPU_measured_binary';measured.write_bytes(b'CPU synthetic historical binary')
    sized=folder/'CPU_size_binary';sized.write_bytes(b'CPU synthetic historical size binary')
    doc=dict(kind='mx5_baseline_i1_evidence',schema_version=1,provenance=provenance,source_files=[dict(name=name,file=m.record(source))],rounds=rounds,
        run_log=m.record(run),guard_result=save(folder/'guard_result.json',dict(code=0)),guard_journal=m.record(journal),
        code_sizes=save(folder/'sizes.json',{n:dict(bytes=100,symbol='CPU_'+n) for n in ('qkv','attn','o','gate_up','down','program')}),
        measured_binary=m.record(measured),size_binary=m.record(sized))
    return save(folder/'baseline.json',doc)


class Fixture:
    def __init__(self,root,monkeypatch):
        self.root=root;self.h=can.Fixture(root,monkeypatch);self.origin=self.h.origin;self.domain=self.h.domain;self.baseline=baseline(root);self.cases={}
    def case(self,index=0,retry=False):
        folder=self.root/('case%d'%index);folder.mkdir()
        bundle=copy.deepcopy(self.h.bundle);bundle['candidate_id']=index;p0=save(folder/'p0.json',bundle)
        primary=m.history.derive_primary576(self.h.origin_ref,index,p0,domain_tool=self.domain);assert primary.action==m.history.Action.PRIMARY,primary.reason
        pin=primary.data['canonical_primary_pin'];node=self.h.h.h.artifact(pin,index=index,mt=168,frame=8)
        content=m._load(node);content.update(kind='mx5_canonical576_attempt',schema_version=2);attempt=save(folder/'attempt.json',content)
        history=m.history.audit_history576(self.h.origin_ref,index,p0,[attempt],domain_tool=self.domain);assert history.action==m.history.Action.HISTORY,history.reason
        history_ref=save(folder/'history.json',history.as_dict());minimum=history.data['actual_attempts'][-1]
        catalogs=[];lineage=[]
        if retry:
            parent=self.h.h.h.artifact(index=index,mt=256,frame=8,query=False)
            next_=m.old.next_width_diagnostic(self.origin,index,[parent]);assert next_.action==m.old.Action.NEXT_WIDTH
            final_ref=self.h.h.h.artifact(next_.data['next_pin'],index=index,ordinal=1)
            lineage=[parent];catalogs=[self.h.h.h.catalog(next_.data['authorization'])]
        else:final_ref=self.h.h.h.artifact(index=index)
        lineage.append(final_ref);final=m.old._artifact(final_ref,self.origin,index,require_query=True);final['resource_ref']=m._load(final_ref)['resources']
        binding=m.history.bind_final512(history_ref,lineage,domain_tool=self.domain,catalogs=catalogs);assert binding.action==m.history.Action.NEEDS_D7,binding.reason
        diagnostic=self.h.h.h.artifact(pin,index=index,mt=233,frame=0,resident=0,extra=('-mllvm','--max-mtreg-number=255'))
        diag=m._load(diagnostic);header=self.h.h.h.h.header
        step=dict(kind='budget_diagnostic',analysis='CPU synthetic controlled max-register diagnostic; never executed',before_native=minimum['files']['compile_receipt'],after_native=diag['native_receipt'],
            source_evidence=[dict(source_path='include/tilemega/Backend/Mx/Fixture.h',before=m.record(header),after=m.record(header))],after_resources=diag['resources'],after_query=diag['query'])
        report=folder/'review.txt';lines=[history_ref['sha256']+' '+self.origin.sha256+' '+minimum['files']['binary']['sha256']+' '+final['files']['binary']['sha256']+' '+m.old.SPEC_DIFFERENCE+' required_registers_gt227_or_global_576_impossible\n']
        lines+=['%s: CPU synthetic evidence consistency review, no hardware or performance qualification.\n'%name for name in m.CHECKS]
        report.write_text(''.join(lines));report_ref=m.record(report)
        known={}
        for prefix,artifact in (('current576',minimum),('current512',final)):
            for role,key in (('mir','log'),('pin','pin'),('generated','generated_source')):known[prefix+'_'+role]=artifact['files'][key]
            known[prefix+'_resources']=artifact['resource_ref']
        required={'spill':['current576_mir','current576_resources'],'wait_locations':['current576_mir','source'],
            'prefetch_amount':['current576_pin','current576_generated'],'prefetch_issue_point':['source','current576_mir'],
            'partition':['current576_pin','current512_pin'],'i1':['current512_resources','current512_mir']}
        checks=[]
        for number,name in enumerate(m.CHECKS,2):
            evidence=[]
            for role in required[name]:
                evidence.append(dict(role=role,span=whole(m.record(header)),source_path='include/tilemega/Backend/Mx/Fixture.h') if role=='source' else dict(role=role,span=whole(known[role])))
            checks.append(dict(name=name,analysis='CPU synthetic detailed location binding for '+name,report_span=m.span(report_ref,number,number),evidence=evidence))
        doc=dict(kind='mx5_qualified_d7_review',schema_version=2,history=history_ref,origin_sha256=self.origin.sha256,candidate_id=index,
            canonical576=dict(pin=minimum['pin'],binary_sha256=minimum['files']['binary']['sha256'],family_sha256=minimum['family'],p0_bundle=p0),
            final512=dict(artifact=final_ref,pin=final['pin'],binary_sha256=final['files']['binary']['sha256'],family_sha256=final['family']),
            resource_budget=binding.data['resource_budget'],spec_difference=m.old.SPEC_DIFFERENCE,assessment='normal_canonical576_floor_still_spills_at_observed_168',
            not_claimed='required_registers_gt227_or_global_576_impossible',report=report_ref,steps=[step],checks=checks,
            i1=dict(mode=m.I1_MODE,not_claimed=m.I1_NOT_CLAIMED,baseline=self.baseline,current_static=m.current_static(final)))
        case=dict(history=history_ref,lineage=lineage,catalogs=catalogs,doc=doc,path=folder/'d7.json',minimum=minimum,final=final,diagnostic=diagnostic)
        self.cases[index]=case;return case
    def audit(self,case):return m.audit_d7_v2(case['history'],case['lineage'],save(case['path'],case['doc']),domain_tool=self.domain,catalogs=case['catalogs'])


def test_substantive_d7_review_ready_keeps_execution_false_and_i1_scope(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);case=h.case();result=h.audit(case)
    assert result.action==m.Action.READY,result.reason
    assert not result.as_dict()['execution_authorized'] and result.data['i1']['mode']==m.I1_MODE
    assert result.data['i1']['not_claimed']==m.I1_NOT_CLAIMED
    assert result.data['i1']['baseline']['rounds']==5


@pytest.mark.parametrize('part',range(6))
def test_missing_ordered_d7_part_is_not_confirmed(tmp_path,monkeypatch,part):
    h=Fixture(tmp_path,monkeypatch);case=h.case();case['doc']['checks'].pop(part)
    assert h.audit(case).action==m.Action.BLOCKED


@pytest.mark.parametrize('change',[
    lambda d:d.update(confirmed=True),
    lambda d:d['canonical576']['pin']['mx_decode_v2'].update(attn_prefetch=0),
    lambda d:d['resource_budget'].update(remaining_reductions=4),
    lambda d:d['resource_budget']['policy'].update(reset_on_thread_change=True),
    lambda d:d['final512'].update(binary_sha256='0'*64),
    lambda d:d['i1'].update(mode='current_source_timing'),
    lambda d:d['i1']['current_static'].update(program_code_bytes=0),
    lambda d:d.update(not_claimed='required_registers_gt227'),
    lambda d:d['checks'][1]['evidence'].pop(),
])
def test_fake_boolean_pk_budget_artifact_i1_and_lower_bound_claims_rejected(tmp_path,monkeypatch,change):
    h=Fixture(tmp_path,monkeypatch);case=h.case();change(case['doc'])
    assert h.audit(case).action==m.Action.BLOCKED


def test_invalid_diagnostic_flags_and_failed_query_rejected(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);case=h.case();pin=case['minimum']['pin']
    bad=h.h.h.h.artifact(pin,mt=233,frame=0,resident=0,extra=('-mllvm','--max-mtreg-number=256'))
    node=m._load(bad);step=case['doc']['steps'][0];step.update(after_native=node['native_receipt'],after_resources=node['resources'],after_query=node['query'])
    assert h.audit(case).action==m.Action.BLOCKED
    node=m._load(case['diagnostic']);query=m._load(node['query']);query['occupancy_returncode']=1
    step.update(after_native=node['native_receipt'],after_resources=node['resources'],after_query=save(tmp_path/'query_failed.json',query))
    assert h.audit(case).action==m.Action.BLOCKED


def test_live_source_mir_and_history_tamper_not_rehashed_as_approval(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);case=h.case()
    saved=m._load(case['history']);saved['details']['used_576_reductions']=None
    case['history']=save(tmp_path/'unknown_history.json',saved)
    assert h.audit(case).action!=m.Action.READY
    case['history']=case['doc']['history'];log=Path(case['minimum']['files']['log']['path']);log.write_text(log.read_text()+'CPU_TAMPER\n')
    assert h.audit(case).action!=m.Action.READY


def test_missing_baseline_or_source_hash_or_current_mir_anchor_blocks(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);case=h.case();baseline=m._load(h.baseline);baseline['rounds'].pop()
    case['doc']['i1']['baseline']=save(tmp_path/'four_rounds.json',baseline)
    assert h.audit(case).action==m.Action.BLOCKED
    baseline=m._load(h.baseline);provenance=m._load(baseline['provenance']);provenance['implementations']['mx4_instruction_order']['source_sha256']='0'*64
    baseline['provenance']=save(tmp_path/'bad_source.json',provenance);case['doc']['i1']['baseline']=save(tmp_path/'bad_baseline.json',baseline)
    assert h.audit(case).action==m.Action.BLOCKED


def test_changed_width_requires_exact_three_fresh_costs_before_d7_ready(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);case=h.case(retry=True);result=h.audit(case)
    assert result.action==m.Action.READY,result.reason
    assert result.data['resource_budget']['used_512_reductions']==1
    assert result.data['prediction']['origin_predicted_tpot_ns']!=result.data['prediction']['effective_predicted_tpot_ns']
    assert m.audit_d7_v2(case['history'],case['lineage'],save(case['path'],case['doc']),domain_tool=h.domain).action!=m.Action.READY


def test_source_repair_binds_actual_old_bytes_but_not_old_reference_metadata_as_history(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);case=h.case();actor=h.h.h.h.h
    saved=(actor.source_root,actor.source_files,actor.producer)
    before_root=tmp_path/'before_source';leaf=before_root/'include/tilemega/Backend/Mx/Fixture.h';leaf.parent.mkdir(parents=True);leaf.write_text('// CPU synthetic actual old source bytes\n')
    producer=tmp_path/'CPU_old_producer';producer.write_bytes(b'CPU old native producer identity')
    actor.source_root=before_root;actor.source_files={'include/tilemega/Backend/Mx/Fixture.h':m.record(leaf)['sha256']};actor.producer=producer
    pin=copy.deepcopy(case['minimum']['pin'])
    for name in m.old.CLASSES:pin['mx_decode_v2'][name].pop('r_next')
    before=h.h.h.h.artifact(pin,mt=168,frame=16)
    actor.source_root,actor.source_files,actor.producer=saved
    before_node=m._load(before)
    case['doc']['steps']=[dict(kind='source_repair',analysis='CPU synthetic old-source vs current-source native/MIR inspection only',
        before_native=before_node['native_receipt'],after_native=case['minimum']['files']['compile_receipt'],
        source_evidence=[dict(source_path='include/tilemega/Backend/Mx/Fixture.h',before=m.record(leaf),after=m.record(actor.header))])]
    result=h.audit(case);assert result.action==m.Action.READY,result.reason
    assert 'not canonical history' in result.data['steps'][0]['before_reference_scope']
    case['doc']['steps'][0]['source_evidence'][0]['before']=m.record(actor.header)
    assert h.audit(case).action==m.Action.BLOCKED


def test_normal_fitting576_cannot_be_inserted_as_the_failed_budget_start(tmp_path,monkeypatch):
    h=Fixture(tmp_path,monkeypatch);case=h.case()
    fit=h.h.h.h.artifact(case['minimum']['pin'],mt=167,frame=0)
    case['doc']['steps'][0]['before_native']=m._load(fit)['native_receipt']
    result=h.audit(case)
    assert result.action==m.Action.BLOCKED and 'exact bound normal576 failure' in result.reason
