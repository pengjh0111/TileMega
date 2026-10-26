"""CPU-only ResolvedTop4 v1. Verification never authorizes execution.

A protocol binds origin_preview, domain_tool, origin_candidates (all four exact
native entries), and plans in ID order. Each plan carries route, proof and final:
576 uses a saved PlanLedger; 512_exception uses an audit plus artifact bundle.
A fitting minimum576 instead needs mode=use_576_solver_path and reroute containing
candidate_id, audit and true_576_preview. This restarts the entire independent
576 solver origin; no ranks or predictions are transferred across origins.
Direct-floor512 audits retain unknown prior576 history and block consumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.util
import json
import math
import statistics
from pathlib import Path

from . import mx5_plan_resources as legacy
from . import mx5_thread_exception as exception
from .mx5_plan_receipt import load_native
from .mx5_select import check_file, record, validate_preview

ROOT = Path(legacy.__file__).resolve().parents[2]
STAGES = legacy.STAGES
BUDGET_POLICY = legacy.FrozenJSON.freeze(dict(maximum_total_reductions=3,scope='whole_plan_across_576_and_512',
                     reset_on_thread_change=False,unknown_history='blocks_consumer'))


class Action(str, Enum):
    READY = 'review_ready_resolved_top4'
    USE_576 = 'use_576_solver_path'
    BLOCKED = 'blocked_evidence'
    NEEDS_HISTORY = 'needs_576_retry_history'


@dataclass(frozen=True)
class Outcome:
    action: Action
    reason: str
    frozen: legacy.FrozenJSON

    @property
    def data(self):
        return self.frozen.data

    def as_dict(self):
        return dict(kind='mx5_resolved_top4_validation', schema_version=1,
                    action=self.action.value, reason=self.reason, details=self.data,
                    execution_authorized=False, final=False, default_enabled=False)


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _load(ref):
    return json.loads(check_file(ref).read_text())


def _same(a, b):
    return legacy.FrozenJSON.freeze(a) == legacy.FrozenJSON.freeze(b)


def _out(action, reason, **details):
    return Outcome(action, reason, legacy.FrozenJSON.freeze(details))


def _preview(ref, threads):
    doc = validate_preview(check_file(ref), allow_schema3_diagnostic=True)
    table = _load(doc['stage_table'])
    _need(table['schema_version'] == 3, 'ResolvedTop4 v1 requires an explicit complete schema3 domain; legacy schema2 is unchanged')
    _need(doc['status'] == 'predicted_top4_resource_review_required', 'complete executable top4 unavailable; preserve domain_report')
    _need(doc['request']['threads'] == threads and all(c['pin']['mx_decode_v2']['threads'] == threads for c in doc['candidates']), 'origin thread scope changed')
    _need(len(doc['candidates'])==4 and all(type(c['index']) is int and c['index']==i for i,c in enumerate(doc['candidates'])), 'exact four integer original IDs required')
    layers = None
    for candidate in doc['candidates']:
        _need(len(candidate['ranks']) == 6 and all(type(x) is int and x >= 0 for x in candidate['ranks']), 'invalid original ranks')
        for name in STAGES:
            legacy._positive(candidate['predicted_stage_ns'][name], name)
        ns = legacy._positive(candidate['predicted_tpot_ns'], 'original prediction')
        count = (ns-candidate['predicted_stage_ns']['head']) / sum(candidate['predicted_stage_ns'][n] for n in STAGES if n != 'head')
        _need(round(count) >= 1 and math.isclose(count, round(count), rel_tol=1e-12), 'invalid origin layer count')
        _need(layers is None or layers == round(count), 'origin layer counts differ')
        layers = round(count)
    _need(all(a['predicted_tpot_ns'] <= b['predicted_tpot_ns'] for a,b in zip(doc['candidates'],doc['candidates'][1:])), 'original prediction order changed')
    return doc, table, layers


class Qualified576Origin(legacy.OriginCertificate):
    @classmethod
    def capture(cls, ref):
        preview, _, layers = _preview(ref, 576)
        return cls(legacy.FrozenJSON.freeze(dict(schema_version=1, kind='mx5_plan_origin_certificate',
            native_preview=ref, preview=preview, layers=layers, policy=legacy.POLICY,
            final=False, default_enabled=False, selection_anchor=0,
            ranking_claim='exact original native top4; no repaired-global-ranking claim')))

    def verify(self):
        _need(self.capture(self.data['native_preview']).frozen == self.frozen, '576 immutable native origin changed')


class Qualified512Origin(exception.Origin):
    @classmethod
    def capture(cls, ref):
        preview, _, layers = _preview(ref, 512)
        return cls(legacy.FrozenJSON.freeze(dict(preview_ref=ref, preview=preview, layers=layers)))

    def verify(self):
        _need(self.capture(self.data['preview_ref']).frozen == self.frozen, '512 immutable native origin changed')


def _qualifier():
    path = ROOT/'experiments/mx5/qualify_campaign.py'
    spec = importlib.util.spec_from_file_location('mx5_resolved_domain_qualifier', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cost_row(row, table):
    _need(row.get('processes')==3 and [x['round'] for x in row['rounds']]==[1,2,3], 'base cost requires three actual rounds')
    medians=[];p90s=[];paths=set()
    for sample in row['rounds']:
        kind,key,provenance,resources=table.validate_measurement(sample['measurement'])
        _need(kind==row['kind'] and key==row['key'] and provenance==table.identity(row['provenance']) and resources==table.resources(row['resources'],key['threads']), 'base measured round/key/source/resource differs')
        _need(sample['measurement']['round']==sample['round'], 'base round number differs')
        path=check_file(sample['input']);line=sample['input']['line'];lines=path.read_text().splitlines()
        _need(type(line) is int and 1<=line<=len(lines) and json.loads(lines[line-1])==sample['measurement'], 'base raw measured row differs')
        _need(path not in paths, 'base measurement rounds reuse an input file');paths.add(path)
        medians.append(sample['measurement']['median_ns']);p90s.append(sample['measurement']['p90_ns'])
    median=statistics.median(medians)
    _need(row['median_ns']==median and row['p90_ns']==statistics.median(p90s) and row['spread']==round((max(medians)-min(medians))/median,4), 'base cost/quantile/spread differs from raw rounds')


def _complete_domain(preview, table, domain_tool):
    """Re-enumerate with the caller's trusted, prebuilt CPU geometry tool."""
    q = _qualifier()
    artifacts = table['provenance']['artifacts']
    for item in [*artifacts, *table['provenance']['tools']]:
        check_file(item)
    _need(record(domain_tool) in artifacts, 'trusted geometry tool is absent from qualification artifacts')
    required_tools=[record(q.__file__),record(q.table.__file__),record(q.table.compiler_probe.__file__)]
    _need(all(item in table['provenance']['tools'] for item in required_tools), 'qualification lacks the trusted complete-outcome tool identities')
    expected, classes = q.scoped_domain(table['scope'], domain_tool)
    seen = {}; audit = dict(modern_normal_checked=0, legacy_cost_only_unverifiable=0)
    for rejected, rows in ((False, table['rows']), (True, table['resource_rejections'])):
        for row in rows:
            if rejected:
                kind,key,provenance,_ = q.table.validate_primary_rejection(row)
                _need(not any(k in row for k in ('median_ns','p90_ns')), 'rejection must have no invented latency')
            else:
                kind=row['kind'];key=q.table.canonical(row,kind);provenance=q.table.identity(row['provenance'])
                _cost_row(row,q.table)
            ref_kind,ref_key,ref_provenance,ref_resources=q.table.validate_schema3_reference(row['r_next_reference'])
            zero_key=dict(key);zero_key['prefetch_kind' if kind=='attention' else 'prefetch']=0
            _need(ref_kind==kind and ref_key==zero_key and row['r_next']==ref_resources['mtreg'], 'R_next does not match the exact measured zero-prefetch reference')
            _need(all(ref_provenance[k]==provenance[k] for k in ('executor','implementation','source_sha256')), 'R_next reference source/implementation differs')
            label='modern_normal_checked' if 'compiler_probe_only' in row['r_next_reference'] else 'legacy_cost_only_unverifiable'
            audit[label] += 1
            index=q.base_id(kind,key)
            _need(index in expected and index not in seen, 'duplicate/out-of-domain cost or rejection')
            _need(provenance['source_sha256'] == table['scope']['source_sha256'][kind], 'domain source epoch differs')
            budget=min(64,(220 if key['threads']==576 else 248)-row['r_next'])
            width=(2 if budget>=64 else 1 if budget>=32 else 0) if kind=='attention' else max(key['depth'],min(expected[index],budget//(key['tn']//4)))
            if rejected:
                _need(key['prefetch_kind' if kind=='attention' else 'prefetch']==width, 'rejection is not prescribed primary width')
            else:
                q.table.validate_selected_resource(row,width)
            seen[index]='rejected' if rejected else 'costed'
    _need(set(seen)==set(expected), 'missing legal domain key; no hand-pruning/fallback estimate')
    coverage=dict(expected_keys=len(expected),costed_keys=len(table['rows']),rejected_keys=len(table['resource_rejections']),missing_keys=0,
        by_class={name:dict(legal=len(keys),costed=sum(seen[k]=='costed' for k in keys),rejected=sum(seen[k]=='rejected' for k in keys)) for name,keys in classes.items()})
    _need(_same(coverage,table['coverage']) and _same(audit,table['r_next_policy_audit']), 'complete-domain coverage/reference report differs')
    return dict(table=preview['stage_table'],scope=table['scope'],coverage=coverage,resource_rejections=table['resource_rejections'],
                r_next_policy_audit=audit,domain_tool=record(domain_tool))


def _replay_576(ref, origin, candidate_id):
    saved=_load(ref)
    _need(saved.get('kind')=='mx5_plan_attempt_ledger' and saved.get('schema_version')==1, 'wrong 576 ledger schema')
    _need(_same(saved['origin'],origin.data) and saved['origin_sha256']==origin.sha256 and saved['candidate_id']==candidate_id, '576 ledger origin/ID changed')
    _need(len(saved['authorizations'])<=3 and len(saved['cost_bindings'])==len(saved['authorizations']), 'at most three total width reductions')
    _need(len(saved['attempts'])==len(saved['authorizations'])+1, '576 chain lacks final actual attempt')
    ledger=legacy.PlanLedger.start(origin,candidate_id)
    for index,attempt in enumerate(saved['attempts']):
        ledger=ledger.append_attempt(check_file(attempt['receipt']))
        _need(ledger.attempts[-1].data==attempt, 'counterfeit/changed actual attempt')
        payload=attempt['payload'];native_ref,native=load_native(check_file(payload['compile_receipt']))
        _need(native['returncode']==0, 'compiler failures cannot authorize width retries')
        files=dict(native['files'],compile_receipt=native_ref)
        exception._normal_options(native,files,_load(files['source_snapshot']),0)
        report=_load(payload['resources'])
        _need(not report['kernel'].get('callees'), 'callee/build failure cannot authorize width retries')
        if payload.get('occupancy'):
            query=exception._query(payload['occupancy'],files,payload['resources'],576,report)
            _need(query['resident_ctas']==1, 'occupancy failure is not a width-retry authorization')
        if index<len(saved['authorizations']):
            binding=saved['cost_bindings'][index]
            catalogs={json.dumps(v['catalog'],sort_keys=True) for v in binding['effective_variant_rows'].values()}
            _need(len(catalogs)==1, 'one exact immutable catalog required for retry')
            catalog=legacy.VariantCatalog.load(check_file(json.loads(next(iter(catalogs)))))
            action=legacy.next_plan_retry(ledger,catalog)
            _need(action.kind==legacy.ActionKind.BUILD_REDUCED_PLAN and action.details.data['authorization']==saved['authorizations'][index], 'retry is not justified deterministic largest-Pw halving')
            ledger=ledger.commit_retry(action,catalog)
            _need(ledger.cost_bindings[-1].data==binding, 'counterfeit exact-width prediction/costs')
            document=catalog.frozen.data['document']
            for name,bound in binding['effective_variant_rows'].items():
                authorization=next(a.data for a in reversed(ledger.authorizations) if a.data['shared_class']==name)
                matches=[e for e in document['rows'] if e['row_sha256']==bound['row_sha256'] and e['authorization']==authorization]
                _need(len(matches)==1,'ambiguous fresh-process row')
                exception._fresh(dict(matches[0],qualification_tool=document['qualification_tool']),bound['row'])
    _need(_same(ledger.as_dict(),saved), 'saved ledger changed')
    action=legacy.next_plan_retry(ledger)
    _need(action.kind==legacy.ActionKind.RESOURCE_FIT, 'final 576 ledger is not resource-fit: '+action.reason)
    return ledger,action.details.data['effective_prediction']


def _budget(threads, reductions):
    complete=threads==576
    return dict(policy=BUDGET_POLICY.data,verified_576_reductions=reductions if complete else None,
                verified_512_reductions=0 if complete else reductions,
                total_width_reductions=reductions if complete else None,
                prescribed_576_history_verified=complete,execution_budget_complete=complete,
                limitation=None if complete else 'direct-floor576 probe is not a prescribed576 retry history; prior reductions are unknown')


def _summary(artifact, prediction, proof, budget):
    return dict(threads=artifact['pin']['mx_decode_v2']['threads'],pin=artifact['pin'],binary=artifact['files']['binary'],
                native_receipt=artifact['files']['compile_receipt'],resources=artifact['resource_ref'],query=artifact['query']['ref'],
                MIR_log=artifact['files']['log'],MIR=artifact['mir'],source_snapshot=artifact['files']['source_snapshot'],
                generated_source=artifact['files']['generated_source'],family_sha256=artifact['family'],
                effective_prediction=prediction,proof=proof,resource_budget=budget)


def _final576(plan, origin, candidate_id):
    ledger,prediction=_replay_576(plan['proof'],origin,candidate_id)
    payload=ledger.attempts[-1].data['payload'];ref,native=load_native(check_file(payload['compile_receipt']))
    files=dict(native['files'],compile_receipt=ref);snapshot=_load(files['source_snapshot'])
    options=exception._normal_options(native,files,snapshot,0)
    report=_load(payload['resources']);mir=exception._mir(files['log'],report['kernel']['name'])
    query=exception._query(payload['occupancy'],files,payload['resources'],576,report)
    artifact=dict(pin=ledger.pin.data,files=files,resources=report['kernel'],mir=mir,query=query,
        family=legacy.digest(dict(source=snapshot['source_digest'],patch=files['source_patch']['sha256'],producer=files['producer']['sha256'],compiler=files['compiler']['sha256'],options=options)),
        source_files=snapshot['source_files'],resource_ref=payload['resources'],stage_sources=payload['stage_sources'])
    exception._strict_final(artifact)
    _need(mir['private']==0, 'final576 MIR contains private accesses')
    budget=_budget(576,len(ledger.authorizations))
    return artifact,_summary(artifact,prediction,plan['proof'],budget),budget


def _final512(plan, origin, candidate_id):
    outcome=exception.check_bound_exception(plan['proof'],origin,candidate_id,plan['artifact'],purpose='guarded_measurement_wd0')
    _need(outcome.action==exception.Action.READY, '512 audit is not consumption-ready: '+outcome.reason)
    audit=_load(plan['proof']);request=audit['details']['request']
    parents,_=exception._lineage(origin,candidate_id,request['lineage512'])
    for parent in parents[:-1]:
        _need(not parent['resources'].get('callees'), 'callee failure cannot authorize width retries')
        _need(parent['query'] is None or parent['query']['resident_ctas']==1, 'occupancy failure is not a width-retry authorization')
    artifact=exception._artifact(plan['artifact'],origin,candidate_id,require_query=True)
    exception._strict_final(artifact)
    _need(artifact['mir']['private']==0, 'final512 MIR contains private accesses')
    artifact['resource_ref']=_load(plan['artifact'])['resources'];artifact['stage_sources']=_load(plan['artifact'])['stage_sources']
    budget=_budget(512,len(request['lineage512'])-1)
    return artifact,_summary(artifact,audit['details']['prediction'],plan['proof'],budget),budget


def _bind_all_sources(table, artifact, checked):
    # Archive proof depends on content/source inputs, not the number of rows.
    context=legacy.digest(dict(sources=artifact['stage_sources'],source_files=artifact['source_files']))
    for row in [*table['rows'],*table['resource_rejections']]:
        source=row['provenance']['source_sha256'];key=(source,context)
        if key not in checked:
            legacy._verify_archive(row,artifact['stage_sources'],artifact['source_files'])
            matches=[x['archive'] for x in artifact['stage_sources'] if x['source_sha256']==source]
            _need(len(matches)==1, 'ambiguous archived stage source')
            checked[key]=matches[0]


def _recheck_archives(checked):
    for ref in checked.values():
        check_file(ref)


def validate_resolved_top4(path, *, domain_tool):
    """domain_tool is a caller-trusted prebuilt CPU tool, never code from JSON."""
    try:
        descriptor=record(path);doc=_load(descriptor);archives={}
        _need(doc.get('kind')=='mx5_resolved_top4' and doc.get('schema_version')==1, 'wrong ResolvedTop4 schema')
        _need(all(doc.get(k) is False for k in ('execution_authorized','final','default_enabled')), 'protocol cannot authorize execution/defaults')
        _need(record(domain_tool)==doc['domain_tool'], 'trusted domain tool changed')
        allowed={'kind','schema_version','mode','origin_preview','domain_tool','origin_candidates','plans','execution_authorized','final','default_enabled','resource_budget_policy'}
        _need(set(doc)==allowed | ({'reroute'} if doc['mode']=='use_576_solver_path' else set()), 'unexpected protocol fields; no independent policy overrides')
        _need(_same(doc['resource_budget_policy'],BUDGET_POLICY.data), 'whole-plan reduction budget cannot reset across thread shapes')
        threads=_load(doc['origin_preview'])['request']['threads']
        _need(threads in (512,576), 'invalid original thread shape')
        origin=(Qualified576Origin if threads==576 else Qualified512Origin).capture(doc['origin_preview'])
        preview=origin.data['preview'];table=_load(preview['stage_table'])
        _need(_same(doc['origin_candidates'],preview['candidates']), 'all four exact original IDs/ranks/pins/predictions/evidence must be retained')
        domain=_complete_domain(preview,table,check_file(doc['domain_tool']))
        if doc['mode']=='use_576_solver_path':
            _need(threads==512 and not doc.get('plans'), '576 reroute cannot mix resolved512/576 plans')
            route=doc['reroute'];index=route['candidate_id'];origin.candidate(index)
            audit=_load(route['audit']);request=audit['details']['request']
            result=exception.audit_thread_exception(origin,index,request['minimum576'],request['lineage512'],catalogs=request['catalogs'],d7=request['d7'],watchdog512=request['watchdog512'])
            _need(result.action==exception.Action.USE_576 and _same(result.as_dict(),audit), 'minimum576 is not a verified fitting veto')
            fresh=Qualified576Origin.capture(route['true_576_preview']);other=fresh.data['preview'];other_table=_load(other['stage_table'])
            _need(preview['input']==other['input'] and preview['producer']==other['producer'], 'true576 preview input/producer differs')
            _need({k:v for k,v in preview['request'].items() if k!='threads'}=={k:v for k,v in other['request'].items() if k!='threads'}, 'true576 request differs')
            _need({k:v for k,v in table['scope'].items() if k!='threads'}=={k:v for k,v in other_table['scope'].items() if k!='threads'}, 'true576 model/request/source differs')
            new_domain=_complete_domain(other,other_table,domain_tool)
            minimum=exception._artifact(request['minimum576'],origin,index,minimum=True,require_query=True)
            minimum['stage_sources']=_load(request['minimum576'])['stage_sources']
            _bind_all_sources(table,minimum,archives);_bind_all_sources(other_table,minimum,archives)
            _recheck_archives(archives)
            return _out(Action.USE_576,'independent complete576 origin required;512 predictions/ranks are not transferred',protocol=descriptor,
                original_preview=doc['origin_preview'],original_candidates=doc['origin_candidates'],original_domain=domain,
                true_576_preview=route['true_576_preview'],true_576_candidates=other['candidates'],true_576_domain=new_domain,selection_anchor=0)
        _need(doc['mode']=='resolved_top4', 'unknown protocol mode')
        plans=doc['plans'];_need(len(plans)==4 and all(type(p['candidate_id']) is int and p['candidate_id']==i for i,p in enumerate(plans)), 'exact ordered original four required; no replacement candidate')
        results=[];family=None
        for index,plan in enumerate(plans):
            expected_fields={'candidate_id','route','proof','final'} | ({'artifact'} if threads==512 else set())
            _need(set(plan)==expected_fields, 'unexpected plan fields; no unverified reductions or cost overrides')
            route='576' if threads==576 else '512_exception'
            _need(plan['route']==route, 'independent512 origin cannot be renamed576; use a true576 solver preview')
            artifact,summary,budget=(_final576 if threads==576 else _final512)(plan,origin,index)
            known=sum(n for n in (budget['verified_512_reductions'],budget['verified_576_reductions']) if n is not None)
            _need(known<=3, 'three total whole-plan reductions exhausted')
            _need(_same(plan['final'],summary), 'final pin/prediction/artifact/query/MIR/cost lineage changed')
            _need(family is None or family==artifact['family'], 'plan source/compiler/options epoch differs across original top4')
            family=artifact['family'];_bind_all_sources(table,artifact,archives)
            results.append(dict(candidate_id=index,original=preview['candidates'][index],final=summary,resource_budget=budget))
        _recheck_archives(archives)
        action=Action.READY if all(p['resource_budget']['execution_budget_complete'] for p in results) else Action.NEEDS_HISTORY
        reason='all four bindings verified; execution and selection remain unauthorized' if action==Action.READY else '512 audit bindings retained, but prescribed576 retry history and total budget are unknown; consumption is blocked'
        return _out(action,reason,validator_source=record(Path(__file__)),protocol=descriptor,origin_preview=doc['origin_preview'],
            domain=domain,plans=results,family_sha256=family,selection_anchor=0,ranking_claim='original exact top4 descendants; no globally exact repaired ranking')
    except Exception as error:
        return _out(Action.BLOCKED,str(error))


def check_measurement_artifact(protocol_ref, candidate_id, binary_ref, *, domain_tool):
    """Future consumer binding check only; does not run/authorize measurements."""
    result=validate_resolved_top4(check_file(protocol_ref),domain_tool=domain_tool)
    if result.action!=Action.READY:
        return result
    try:
        _need(type(candidate_id) is int and candidate_id in range(4), 'invalid original candidate ID')
        expected=result.data['plans'][candidate_id]['final']['binary']
        _need(binary_ref==expected and record(check_file(binary_ref))==expected, 'measurement artifact swap')
        return _out(Action.READY,'exact measurement artifact reverified; execution remains unauthorized',protocol=protocol_ref,candidate_id=candidate_id,binary=binary_ref)
    except Exception as error:
        return _out(Action.BLOCKED,str(error))


def capture_plan(preview_ref, candidate_id, proof, *, artifact=None):
    """Construct a plan entry by verification, never by copying console numbers."""
    threads=_load(preview_ref)['request']['threads']
    origin=(Qualified576Origin if threads==576 else Qualified512Origin).capture(preview_ref)
    origin.candidate(candidate_id)
    plan=dict(candidate_id=candidate_id,route='576' if threads==576 else '512_exception',proof=proof)
    if artifact is not None:plan['artifact']=artifact
    _,plan['final'],_=(_final576 if threads==576 else _final512)(plan,origin,candidate_id)
    return plan


def capture_protocol(preview_ref, domain_tool, plans):
    """Return a proposed record; validate_resolved_top4 still replays every proof."""
    preview=_load(preview_ref)
    return dict(kind='mx5_resolved_top4',schema_version=1,mode='resolved_top4',origin_preview=preview_ref,
        domain_tool=record(domain_tool),origin_candidates=preview['candidates'],plans=plans,
        execution_authorized=False,final=False,default_enabled=False,resource_budget_policy=BUDGET_POLICY.data)
