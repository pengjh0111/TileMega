"""Qualified576 history v2, CPU-only; old floor-PF0/unset-Rnext audits are unchanged.

Only actual canonical PRIMARY576 receipts can prove zero prior reductions. A
computed nonzero attention kind stays unchanged through the width-only chain.
A budget-complete result still needs substantive qualified-history D7 evidence.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from . import mx5_resolved_top4 as resolved
from . import mx5_thread_exception as old
from . import mx5_plan_resources as legacy
from .mx5_plan_receipt import load_native
from .mx5_resources import inspect
from .mx5_select import check_file, record

CLASSES=legacy.CLASSES
GRID=(65,256,512,768,1088)


class Action(str,Enum):
    PRIMARY='canonical_primary576_derived'
    NEEDS_P0='needs_modern576_p0_references'
    NEEDS_ATTEMPT='needs_canonical576_attempt'
    NEEDS_QUERY='needs_actual576_query'
    NEXT_576='next_canonical576_width'
    NEXT_512='next_remaining_budget512_width'
    USE_576='use_576_solver_path'
    HISTORY='qualified576_history_bound'
    NEEDS_COSTS='needs_stage_measurements'
    NEEDS_D7='needs_qualified_history_v2_d7_review'
    BLOCKED='blocked_evidence'


@dataclass(frozen=True)
class Outcome:
    action: Action
    reason: str
    frozen: legacy.FrozenJSON

    @property
    def data(self):return self.frozen.data

    def as_dict(self):
        return dict(kind='mx5_qualified576_history_audit',schema_version=2,action=self.action.value,
                    reason=self.reason,details=self.data,execution_authorized=False,final=False,default_enabled=False)


def _need(condition,reason):
    if not condition:raise ValueError(reason)


def _load(ref):return json.loads(check_file(ref).read_text())
def _same(a,b):return legacy.FrozenJSON.freeze(a)==legacy.FrozenJSON.freeze(b)
def _out(action,reason,**data):return Outcome(action,reason,legacy.FrozenJSON.freeze(data))


def _modern_p0(reference,key,kind,source,table):
    _need(reference.get('compiler_probe_only') is True,'modern unexecuted P0 marker is required')
    _need(not any(x in reference for x in ('median_ns','p90_ns','rounds','ns','processes')),'P0 is not a timed cost')
    actual_kind,actual_key,provenance,attrs=table.validate_schema3_reference(reference)
    _need(attrs['static_smem_bytes']==0 and attrs['resident_ctas']==1,
          'P0 static shared memory or occupancy anomaly is not a qualified R_next probe')
    _need(actual_kind==kind and actual_key==key,'P0 exact576 shape/coordinate changed')
    _need(provenance['source_sha256']==source,'P0 source differs from independent512 qualified source epoch')
    raw_ref=reference['probe']['raw_query'];path=check_file(raw_ref);line=raw_ref['line'];lines=path.read_text().splitlines()
    _need(type(line) is int and 1<=line<=len(lines),'P0 needs exact raw JSONL fileSHA+line')
    raw=json.loads(lines[line-1])
    encoding=raw_ref.get('encoding','flat_query_v1');observed=raw
    if encoding=='calibration_inspect_inline_reference_v1':
        # inspect preserves the original inventory key outside and the actual
        # queried PF0 attributes/reference inside; never fabricate a flat row.
        _need(kind=='attention','inline inspect encoding is bounded to attention P0')
        original=copy.deepcopy(reference);original['probe'].pop('raw_query')
        _need(_same(raw.get('r_next_reference'),original),'inline actual P0 reference differs')
        outer=dict(key);outer.pop('prefetch_kind')
        _need(raw.get('kind')==kind and (_same(raw.get('key'),outer) or _same(raw.get('key'),key)),'inline P0 inventory coordinates differ')
        _need(type(raw.get('fits')) is bool and raw['fits']==reference['fits'],'inline P0 fits differs')
        observed=raw['r_next_reference']
    else:_need(encoding=='flat_query_v1','unsupported actual P0 raw encoding')
    for name in ('kind','key','provenance','resources','fits'):
        _need(_same(observed.get(name),reference[name]),'raw P0 differs: '+name)
    _need(raw.get('query_only') is True and type(raw.get('iterations')) is int and raw['iterations']==0
          and type(raw.get('query_status')) is int and raw['query_status']==0,'P0 raw query must be successful zero-execution attributes')
    _need(not any(x in raw for x in ('median_ns','p90_ns','rounds','spread','processes')),'P0 raw query contains latency')
    _need(Path(raw['library']).resolve()==check_file(reference['probe']['binary']),'raw P0 binary differs')
    metadata=_load(reference['probe']['build_metadata']);command=metadata['command']
    _need(command and Path(command[0]).is_absolute(),'P0 compiler must be the actual absolute command executable')
    compiler=record(command[0])
    return dict(reference=reference,raw_query=raw_ref,resources=attrs,compiler=compiler,
                compiler_binding='retained_executable_at_validation',original_build_compiler_sha256=None,
                original_compiler_identity_recorded=False)


def _derive(origin_ref,candidate_id,p0_ref,domain_tool):
    origin=resolved.Qualified512Origin.capture(origin_ref);candidate=origin.candidate(candidate_id)
    preview=origin.data['preview'];table=_load(preview['stage_table'])
    domain=resolved._complete_domain(preview,table,domain_tool)
    bundle=_load(p0_ref)
    _need(set(bundle)=={'kind','schema_version','origin_sha256','candidate_id','references'},'unexpected P0 bundle fields')
    _need(bundle['kind']=='mx5_canonical576_p0' and bundle['schema_version']==2
          and bundle['origin_sha256']==origin.sha256 and type(bundle['candidate_id']) is int
          and bundle['candidate_id']==candidate_id,'P0 bundle origin/ID/version mismatch')
    _need(set(bundle['references'])<=set(legacy.STAGES),'unexpected P0 stage')
    q=resolved._qualifier();scope=copy.deepcopy(table['scope']);scope['threads']=576
    slots,_=q.scoped_domain(scope,domain_tool)
    pin=copy.deepcopy(candidate['pin']);config=pin['mx_decode_v2'];config.update(threads=576,purpose='resource_probe')
    missing=[];proofs={};firsts={}
    for name in legacy.STAGES:
        rows=candidate['evidence'][name];references=bundle['references'].get(name,[])
        _need(isinstance(references,list),'P0 references must be exact row lists')
        expected=[]
        for row in rows:
            key=dict(row['key'],threads=576);key['prefetch_kind' if name=='attention' else 'prefetch']=0
            expected.append((row['kind'],key,row['provenance']['source_sha256']))
        provided={}
        allowed={legacy.digest(dict(kind=k,key=key)) for k,key,_ in expected}
        for reference in references:
            identity=legacy.digest(dict(kind=reference['kind'],key=reference['key']))
            _need(identity in allowed and identity not in provided,'duplicate/out-of-shape modern P0 reference')
            provided[identity]=reference
        bound=[]
        for kind,key,source in expected:
            identity=legacy.digest(dict(kind=kind,key=key))
            if identity not in provided:
                missing.append(dict(stage=name,kind=kind,key=key));continue
            _need(q.base_id(kind,key) in slots,'original geometry is outside legal576 domain')
            bound.append(_modern_p0(provided[identity],key,kind,source,q.table))
        if len(bound)!=len(expected):continue
        proofs[name]=bound
        registers={r['resources']['mtreg'] for r in bound}
        _need(len(registers)==1,'one program stage cannot have inconsistent P0 MTreg across L grid')
        rn=next(iter(registers));budget=min(64,220-rn)
        if name=='attention':
            _need([key['length'] for _,key,_ in expected]==list(GRID),'attention P0 requires exact five L-grid keys')
            config['attn_prefetch']=2 if budget>=64 else 1 if budget>=32 else 0
        else:
            kind,key,_=expected[0];first=slots[q.base_id(kind,key)];firsts[name]=first
            config[name]['r_next']=rn
            config[name]['prefetch']=max(config[name]['d'],min(first,budget//(config[name]['tn']//4)))
    compilers={p['compiler']['sha256'] for values in proofs.values() for p in values}
    _need(len(compilers)<=1,'P0 retained compiler identities differ across actual stage probes')
    facts=dict(origin_preview=origin_ref,origin_sha256=origin.sha256,candidate_id=candidate_id,
        original_candidate=candidate,domain=domain,p0_bundle=p0_ref,p0=proofs,first_chunks=firsts,
        canonical_primary_pin=None if missing else pin,resource_budget_policy=resolved.BUDGET_POLICY.data,
        interpretation='P0 attributes are compiler probes; stack/static/occupancy remain diagnostic, never inferred R_next or costs')
    return origin,facts,missing


def derive_primary576(origin_ref,candidate_id,p0_ref,*,domain_tool):
    try:
        _,facts,missing=_derive(origin_ref,candidate_id,p0_ref,domain_tool)
        return _out(Action.NEEDS_P0 if missing else Action.PRIMARY,
            'exact modern576 P0 references missing' if missing else 'canonical primary derived; no program build, cost or execution qualification',**facts,missing=missing)
    except Exception as error:return _out(Action.BLOCKED,str(error))


def _attempt(ref,origin,candidate_id,expected_pin,ordinal,p0):
    node=_load(ref)
    _need(set(node)=={'kind','schema_version','origin_sha256','candidate_id','ordinal','native_receipt','resources','query','stage_sources'},'unexpected qualified576 attempt fields')
    _need(node['kind']=='mx5_canonical576_attempt' and node['schema_version']==2 and node['origin_sha256']==origin.sha256
          and type(node['candidate_id']) is int and node['candidate_id']==candidate_id
          and type(node['ordinal']) is int and node['ordinal']==ordinal,'qualified576 attempt origin/ordinal/version mismatch')
    native_ref,native=load_native(check_file(node['native_receipt']))
    _need(native['returncode']==0 and native['files'].get('binary') is not None,'failed device compilation is not an MT/frame retry')
    files=dict(native['files'],compile_receipt=native_ref);pin=_load(files['pin']);resolved_pin=_load(files['resolved_pin'])['mx_decode_v2']
    _need(_same(pin,expected_pin),'actual pin differs from computed PRIMARY576 or deterministic width-only child; PF0/unset-R_next shortcuts are forbidden')
    config=pin['mx_decode_v2'];snapshot=_load(files['source_snapshot'])
    _need(snapshot['source_digest']==legacy.digest(snapshot['source_files']),'program source digest differs')
    legacy._snapshot_inputs(snapshot)
    _need(files['source_patch']['sha256']==snapshot['worktree_diff_sha256'],'program source patch changed')
    _need(files['producer']['sha256']==snapshot['compiler_sha256']==origin.data['preview']['producer']['sha256'],'actual producer/source epoch differs from native origin')
    _need(Path(snapshot['compiler']).resolve()==check_file(files['producer']),'actual source compiler path differs')
    options=old._normal_options(native,files,snapshot,0)
    _need(all(p['compiler']['sha256']==files['compiler']['sha256'] for values in p0.values() for p in values),'native compiler differs from retained P0 compiler identity')
    old._configuration(check_file(files['generated_source']).read_text(),origin,candidate_id,pin)
    for name in CLASSES:
        for field in ('tn','wn','d','part','prefetch','r_next'):
            _need(type(resolved_pin[name].get(field)) is type(config[name][field]) and resolved_pin[name][field]==config[name][field],'resolved576 pin changed: '+name+'.'+field)
        key=origin.candidate(candidate_id)['evidence'][name][0]['key']
        for field,value in (('n',key['n']),('k',key['k']),('b',origin.data['preview']['request']['batch'])):
            _need(resolved_pin[name].get(field)==value,'resolved576 shape changed')
        op='argmax' if name=='head' else ('kStore','kResidual','kSwiGLU')[key['op']]
        _need(resolved_pin[name].get('op')==op and resolved_pin[name].get('norm')==bool(key.get('norm',1)),'resolved576 operator changed')
    _need(all(resolved_pin.get(k)==config[k] for k in ('threads','b_min','attn_prefetch')),'resolved576 thread/attention changed')
    checked=set()
    for refs in p0.values():
        for proof in refs:
            row=proof['reference'];source=row['provenance']['source_sha256']
            if source not in checked:legacy._verify_archive(row,node['stage_sources'],snapshot['source_files']);checked.add(source)
    report=inspect(check_file(files['binary']),check_file(files['log']),576)
    _need(_same(report,_load(node['resources'])),'actual576 resource report differs from ELF/same-compile MIR')
    kernel=report['kernel']
    for field in ('registers','scalar_registers','stack_bytes','static_smem_bytes'):
        old._int(kernel[field],field,1 if field=='registers' else 0)
    _need(kernel['name']==old.occupancy.KERNEL and set(report['kernel_instructions'])=={kernel['name']} and len(report['device_images'])==1,'unsupported actual576 kernel/image metadata')
    _need(kernel['static_smem_bytes']==0 and not kernel.get('callees'),'smem/callee failure is not an MT/frame retry')
    mir=old._mir(files['log'],kernel['name'])
    _need(mir['calls']==0,'actual576 MIR call is not a width-retry authorization')
    query=old._query(node['query'],files,node['resources'],576,report) if node['query'] else None
    _need(query is None or query['resident_ctas']==1,'actual576 occupancy anomaly stops this history')
    family=legacy.digest(dict(source=snapshot['source_digest'],patch=files['source_patch']['sha256'],producer=files['producer']['sha256'],compiler=files['compiler']['sha256'],options=options))
    return dict(ref=ref,pin=pin,files=files,resources=kernel,resource_ref=node['resources'],query=query,mir=mir,
                family=family,source_files=snapshot['source_files'],stage_sources=node['stage_sources'])


def _halve(pin):
    child=copy.deepcopy(pin);c=child['mx_decode_v2']
    reducible=[n for n in CLASSES if c[n]['prefetch']>c[n]['d']]
    _need(reducible,'all Pw are already D')
    name=min(reducible,key=lambda n:(-c[n]['prefetch'],CLASSES.index(n)))
    before=c[name]['prefetch'];c[name]['prefetch']=max(c[name]['d'],before//2)
    return child,dict(shared_class=name,from_pw=before,to_pw=c[name]['prefetch'])


def audit_history576(origin_ref,candidate_id,p0_ref,attempt_refs,*,domain_tool):
    request=dict(origin_preview=origin_ref,candidate_id=candidate_id,p0_bundle=p0_ref,
                 attempts=list(attempt_refs),domain_tool=record(domain_tool))
    try:
        origin,facts,missing=_derive(origin_ref,candidate_id,p0_ref,domain_tool)
        if missing:return _out(Action.NEEDS_P0,'exact modern576 P0 references missing',request=request,missing=missing)
        _need(len(attempt_refs)<=4,'at most three total whole-plan reductions; no fourth576 retry')
        pin=facts['canonical_primary_pin'];chain=[];authorizations=[];family=None
        for ordinal,ref in enumerate(attempt_refs):
            if ordinal:
                parent=chain[-1];r=parent['resources']
                _need(r['registers']>227 or r['stack_bytes']>0,'a fitting576 parent cannot authorize a width reduction')
                pin,authorization=_halve(pin);authorizations.append(dict(authorization,parent=parent['ref'],global_reduction_step=ordinal,affected_edges=legacy.affected_edges(authorization['shared_class'],origin.data['layers'])))
            node=_attempt(ref,origin,candidate_id,pin,ordinal,facts['p0'])
            _need(family is None or family==node['family'],'576 source/compiler/options epoch changed across actual attempts')
            family=node['family'];chain.append(node)
        if not chain:return _out(Action.NEEDS_ATTEMPT,'actual canonical PRIMARY576 receipt is missing; a bare floor proves no zero count',request=request,**facts)
        last=chain[-1];r=last['resources'];used=len(authorizations)
        data=dict(request=request,derivation=facts,actual_attempts=chain,authorizations=authorizations,
            used_576_reductions=used,remaining_512_reductions=3-used,family_sha256=family,
            validator_source=record(Path(__file__)),selection_anchor=0)
        if r['registers']<=227 and r['stack_bytes']==0:
            if last['query'] is None:return _out(Action.NEEDS_QUERY,'fitting576 still needs its actual query',**data)
            old._strict_final(last);_need(last['mir']['private']==0,'fitting576 has private MIR accesses')
            return _out(Action.USE_576,'actual576 fits; no512 exception or transferred prediction',**data)
        c=pin['mx_decode_v2'];reducible=any(c[n]['prefetch']>c[n]['d'] for n in CLASSES)
        if reducible:
            _need(used<3,'three whole-plan reductions exhausted before allPwD; no floor shortcut or budget reset')
            next_pin,authorization=_halve(pin)
            return _out(Action.NEXT_576,'one prescribed nonexecuting width diagnostic; no timing cost is fabricated',**data,next_pin=next_pin,next_authorization=authorization)
        if last['query'] is None:return _out(Action.NEEDS_QUERY,'actual failed canonical-attention floor576 query is missing',**data)
        return _out(Action.HISTORY,'actual PRIMARY-to-floor history binds budget only; qualified-history D7/512 execution approval is absent',**data,
            floor_is_actual_primary=used==0,canonical_attention_kind=c['attn_prefetch'],
            spec_difference=old.SPEC_DIFFERENCE if r['registers']==168 else None,
            not_claimed='required_registers_gt227_or_global_576_impossible',
            fallback_observation='literal_mt_gt227' if r['registers']>227 else 'observed_168_stack' if r['registers']==168 else 'stack_only_requires_explicit_review')
    except Exception as error:return _out(Action.BLOCKED,str(error),request=request)


def bind_final512(history_ref,lineage512,*,domain_tool,catalogs=()):
    """Bind remaining budget and independent exact512 costs, not a D7 approval."""
    try:
        saved=_load(history_ref)
        _need(saved.get('kind')=='mx5_qualified576_history_audit' and saved.get('schema_version')==2
              and saved.get('execution_authorized') is False and saved.get('action')==Action.HISTORY.value,'expected qualified-history-v2 budget proof')
        request=saved['details']['request'];_need(record(domain_tool)==request['domain_tool'],'trusted history domain tool changed')
        current=audit_history576(request['origin_preview'],request['candidate_id'],request['p0_bundle'],request['attempts'],domain_tool=domain_tool)
        _need(current.action==Action.HISTORY and _same(current.as_dict(),saved),'576 actual history/count/source proof changed; no reset or claimed zero')
        origin=resolved.Qualified512Origin.capture(request['origin_preview']);index=request['candidate_id']
        artifacts,authorizations=old._lineage(origin,index,lineage512)
        used576=current.data['used_576_reductions'];used512=len(authorizations)
        _need(used576+used512<=3,'512 retries exceed the remaining cross-shape total budget')
        for node in artifacts:
            _need(node['family']==current.data['family_sha256'],'512 source/compiler/options family differs from actual576 history')
            _need(not node['resources'].get('callees') and node['mir']['calls']==0 and (node['query'] is None or node['query']['resident_ctas']==1),'512 callee/occupancy anomaly is not a retry authorization')
        final=artifacts[-1];budget=dict(used_576_reductions=used576,used_512_reductions=used512,remaining_reductions=3-used576-used512,policy=resolved.BUDGET_POLICY.data)
        data=dict(history=history_ref,original_candidate=origin.candidate(index),lineage512=list(lineage512),catalogs=list(catalogs),resource_budget=budget,selection_anchor=0)
        if old._failure(final):
            _need(budget['remaining_reductions']>0,'total budget exhausted;512 still fails and cannot restart')
            next_pin,authorization=old._next(origin,index,final,used512+1)
            return _out(Action.NEXT_512,'one actual failure-authorized width diagnostic within remaining budget',**data,next_pin=next_pin,authorization=authorization)
        old._strict_final(final);_need(final['mir']['private']==0,'final512 contains private MIR accesses')
        costs=old._costs(origin,index,artifacts,authorizations,catalogs)
        if costs['missing']:return _out(Action.NEEDS_COSTS,'exact source-qualified three-fresh-process changed-width costs are missing',**data,**costs)
        return _out(Action.NEEDS_D7,'budget and final512 cost/resource bindings verified; substantive qualified-history-v2 D7 evidence and existing consumer validation are required',**data,**costs,final512=final,
            ranking_claim='descendant of independent512 original top4; no576 prediction or repaired-global-ranking transfer')
    except Exception as error:return _out(Action.BLOCKED,str(error))
