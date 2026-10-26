"""ResolvedTop4 v2 CPU consumption of qualified D7; v1 remains unchanged/off."""
from dataclasses import dataclass
from pathlib import Path

from . import mx5_resolved_top4 as v1
from . import mx5_d7_v2 as d7
from . import mx5_canonical576_history as history
from .mx5_select import check_file,record


@dataclass(frozen=True)
class Outcome(v1.Outcome):
    def as_dict(self):
        value=super().as_dict();value['schema_version']=2;return value


def _out(action,reason,**data):return Outcome(action,reason,v1.legacy.FrozenJSON.freeze(data))
def _need(condition,reason):v1._need(condition,reason)
def _load(ref):return v1._load(ref)


def _final512(plan,origin,index,domain_tool):
    audit=d7.audit_d7_v2(plan['history'],plan['lineage512'],plan['d7'],domain_tool=domain_tool,catalogs=plan['catalogs'])
    _need(audit.action==d7.Action.READY,'qualified-history-v2 D7 is incomplete: '+audit.reason)
    data=audit.data
    _need(data['origin_preview']==origin.data['preview_ref'] and data['original_candidate']==origin.candidate(index),'D7 belongs to another independent origin/ID/ranks')
    artifact=data['final512'];artifact['resource_ref']=_load(artifact['ref'])['resources'];artifact['stage_sources']=_load(artifact['ref'])['stage_sources']
    b=data['resource_budget']
    budget=dict(policy=v1.BUDGET_POLICY.data,verified_576_reductions=b['used_576_reductions'],verified_512_reductions=b['used_512_reductions'],
        total_width_reductions=b['used_576_reductions']+b['used_512_reductions'],prescribed_576_history_verified=True,execution_budget_complete=True,limitation=None)
    summary=v1._summary(artifact,data['prediction'],plan['d7'],budget)
    summary.update(canonical_history=plan['history'],qualified_d7=plan['d7'],i1=data['i1'],limitations=data['limitations'])
    return artifact,summary,budget


def capture_plan(preview_ref,candidate_id,*,domain_tool,proof=None,history_ref=None,lineage512=(),catalogs=(),d7_ref=None):
    threads=_load(preview_ref)['request']['threads']
    if threads==576:return v1.capture_plan(preview_ref,candidate_id,proof)
    origin=v1.Qualified512Origin.capture(preview_ref);origin.candidate(candidate_id)
    plan=dict(candidate_id=candidate_id,route='512_qualified_history_v2',history=history_ref,lineage512=list(lineage512),catalogs=list(catalogs),d7=d7_ref)
    _,plan['final'],_=_final512(plan,origin,candidate_id,domain_tool);return plan


def capture_protocol(preview_ref,domain_tool,plans):
    doc=v1.capture_protocol(preview_ref,domain_tool,plans);doc['schema_version']=2;return doc


def _reroute(doc,origin,table,domain_tool,domain,archives):
    route=doc['reroute'];_need(set(route)=={'candidate_id','history','true_576_preview'},'unexpected576 reroute fields')
    index=route['candidate_id'];origin.candidate(index);saved=_load(route['history']);request=saved['details']['request']
    _need(request['origin_preview']==doc['origin_preview'] and request['candidate_id']==index,'576 reroute belongs to another independent origin/ID')
    current=history.audit_history576(request['origin_preview'],index,request['p0_bundle'],request['attempts'],domain_tool=domain_tool)
    _need(current.action==history.Action.USE_576 and v1._same(current.as_dict(),saved),'fitting576 veto/history is not genuine/live')
    fresh=v1.Qualified576Origin.capture(route['true_576_preview']);other=fresh.data['preview'];other_table=_load(other['stage_table']);preview=origin.data['preview']
    _need(preview['input']==other['input'] and preview['producer']==other['producer'],'true576 input/producer differs')
    _need({k:v for k,v in preview['request'].items() if k!='threads'}=={k:v for k,v in other['request'].items() if k!='threads'},'true576 request differs')
    _need({k:v for k,v in table['scope'].items() if k!='threads'}=={k:v for k,v in other_table['scope'].items() if k!='threads'},'true576 model/source scope differs')
    new_domain=v1._complete_domain(other,other_table,domain_tool);minimum=current.data['actual_attempts'][-1]
    v1._bind_all_sources(table,minimum,archives);v1._bind_all_sources(other_table,minimum,archives);v1._recheck_archives(archives)
    return _out(v1.Action.USE_576,'actual576 fits; restart independent576 solver origin without512 prediction/rank transfer',
        original_preview=doc['origin_preview'],original_candidates=doc['origin_candidates'],original_domain=domain,
        true_576_preview=route['true_576_preview'],true_576_candidates=other['candidates'],true_576_domain=new_domain,selection_anchor=0)


def validate_resolved_top4(path,*,domain_tool):
    try:
        descriptor=record(path);doc=_load(descriptor);archives={}
        fields={'kind','schema_version','mode','origin_preview','domain_tool','origin_candidates','plans','execution_authorized','final','default_enabled','resource_budget_policy'}
        _need(set(doc)==fields|({'reroute'} if doc.get('mode')=='use_576_solver_path' else set()),'unexpected v2 protocol fields')
        _need(doc['kind']=='mx5_resolved_top4' and doc['schema_version']==2 and all(doc[k] is False for k in ('execution_authorized','final','default_enabled')),'explicit audit-only ResolvedTop4 v2 required')
        _need(record(domain_tool)==doc['domain_tool'] and v1._same(doc['resource_budget_policy'],v1.BUDGET_POLICY.data),'domain tool/budget reset changed')
        threads=_load(doc['origin_preview'])['request']['threads'];_need(threads in (512,576),'unsupported origin threads')
        origin=(v1.Qualified576Origin if threads==576 else v1.Qualified512Origin).capture(doc['origin_preview'])
        preview=origin.data['preview'];table=_load(preview['stage_table'])
        _need(v1._same(doc['origin_candidates'],preview['candidates']),'all four original IDs/ranks/pins/predictions/evidence must remain exact')
        domain=v1._complete_domain(preview,table,domain_tool)
        if doc['mode']=='use_576_solver_path':
            _need(threads==512 and not doc['plans'],'576 reroute cannot mix origins/thread rankings')
            return _reroute(doc,origin,table,domain_tool,domain,archives)
        _need(doc['mode']=='resolved_top4','unknown v2 mode')
        plans=doc['plans'];_need(len(plans)==4 and all(type(p['candidate_id']) is int and p['candidate_id']==i for i,p in enumerate(plans)),'exact four original ordered IDs required; no fifth/replacement candidate')
        results=[];family=None
        for index,plan in enumerate(plans):
            if threads==576:
                _need(set(plan)=={'candidate_id','route','proof','final'} and plan['route']=='576','independent576 ledger route required')
                artifact,summary,budget=v1._final576(plan,origin,index)
            else:
                _need(set(plan)=={'candidate_id','route','history','lineage512','catalogs','d7','final'} and plan['route']=='512_qualified_history_v2','qualified-history-v2 route required; v1 unknown history is not upgraded')
                artifact,summary,budget=_final512(plan,origin,index,domain_tool)
            _need(v1._same(plan['final'],summary),'final artifact/query/pin/MIR/precise prediction/D7 bindings changed')
            _need(budget['execution_budget_complete'] and budget['total_width_reductions']<=3,'unknown or reset cross-shape reduction history')
            _need(family is None or family==artifact['family'],'four candidates have mixed source/compiler/options epochs')
            family=artifact['family'];v1._bind_all_sources(table,artifact,archives)
            results.append(dict(candidate_id=index,original=preview['candidates'][index],final=summary,resource_budget=budget))
        v1._recheck_archives(archives)
        return _out(v1.Action.READY,'all four qualified bindings review-ready; no execution/default/selection applied',protocol=descriptor,
            origin_preview=doc['origin_preview'],domain=domain,plans=results,family_sha256=family,selection_anchor=0,
            ranking_claim='descendants of one original exact top4; no repaired-global-ranking claim',
            i1_scope=d7.I1_MODE if threads==512 else 'unchanged576 ledger scope',
            not_claimed=d7.I1_NOT_CLAIMED if threads==512 else None,validator_source=record(Path(__file__)))
    except Exception as error:return _out(v1.Action.BLOCKED,str(error))


def check_measurement_artifact(protocol_ref,candidate_id,binary_ref,*,domain_tool):
    result=validate_resolved_top4(check_file(protocol_ref),domain_tool=domain_tool)
    if result.action!=v1.Action.READY:return result
    try:
        _need(type(candidate_id) is int and candidate_id in range(4),'invalid original candidate ID')
        expected=result.data['plans'][candidate_id]['final']['binary']
        _need(binary_ref==expected and record(check_file(binary_ref))==expected,'measurement artifact swap')
        return _out(v1.Action.READY,'v2 exact artifact binding reverified; execution remains unauthorized',protocol=protocol_ref,candidate_id=candidate_id,binary=binary_ref)
    except Exception as error:return _out(v1.Action.BLOCKED,str(error))
