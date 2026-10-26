"""Prepare an independent512 original bound to genuine canonical576 shared history."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import json

from . import mx5_canonical576_history as history
from . import mx5_plan_receipt as receipts
from . import mx5_resolved_runner as runner
from . import mx5_select as common

ROOT=Path(common.__file__).resolve().parents[2]
load,record,check_file,write=common.load,common.record,common.check_file,common.write

def need(value,reason):
    if not value:raise ValueError(reason)

def _context(history_ref,sources_ref,domain_ref):
    saved=load(check_file(history_ref))
    need(saved.get('kind')=='mx5_qualified576_history_audit' and saved.get('schema_version')==2 and saved.get('action')==history.Action.HISTORY.value and saved.get('execution_authorized') is False,'genuine completed canonical576 history required; fitting576 veto cannot prepare512')
    request=saved['details']['request'];need(request['domain_tool']==domain_ref,'trusted history domain tool changed')
    actual=history.audit_history576(request['origin_preview'],request['candidate_id'],request['p0_bundle'],request['attempts'],domain_tool=check_file(domain_ref))
    need(actual.action==history.Action.HISTORY and history._same(actual.as_dict(),saved),'canonical576 actual history changed; no claimed/reset reduction count')
    origin=history.resolved.Qualified512Origin.capture(request['origin_preview']);origin.candidate(request['candidate_id'])
    preview=origin.data['preview'];table=load(check_file(preview['stage_table']))
    domain=history.resolved._complete_domain(preview,table,check_file(domain_ref))
    sources=load(check_file(sources_ref));need(isinstance(sources,list) and sources,'actual stage source inventory required')
    digests=[s['source_sha256'] for s in sources]
    need(len(set(digests))==len(digests) and set(digests)==set(table['scope']['source_sha256'].values()),'stage source inventory differs from independent512 source epoch')
    for source in sources:check_file(source['archive'])
    used=actual.data['used_576_reductions'];remaining=actual.data['remaining_512_reductions']
    need(type(used) is int and 0<=used<=3 and remaining==3-used,'invalid genuine shared reduction budget')
    return dict(origin_preview=request['origin_preview'],origin_sha256=origin.sha256,candidate_id=request['candidate_id'],origin_candidates=preview['candidates'],domain=domain,resource_budget=dict(used_576_reductions=used,used_512_reductions=0,remaining_reductions=remaining,policy=history.resolved.BUDGET_POLICY.data),canonical_family_sha256=actual.data['family_sha256']),origin,sources

def _build(preview,pin,workspace):
    binary=Path(workspace)/'artifact0/plan.so';argv=runner._compile_command(preview,pin,binary);argv[-1]='0'
    return dict(argv=argv,binary=str(binary),native_receipt=str(binary)+'.compile_receipt.json',resources=str(binary)+'.resources.json',pin=pin,ordinal=0,watchdog=0,timeout_seconds=900)

def prepare(history_path,sources,domain,out):
    out=Path(out).resolve();need(not out.exists(),'preserve previous preparation output')
    hr,sr,dr=record(history_path),record(sources),record(domain);context,origin,_=_context(hr,sr,dr)
    out.mkdir(parents=True);folder=out/'artifact0';folder.mkdir();pin=folder/'requested.pin.json';write(pin,origin.candidate(context['candidate_id'])['pin'])
    state=dict(kind='mx5_independent512_original_preparation',schema_version=1,history576=hr,stage_sources=sr,domain_tool=dr,context=context,runtime_target=record(origin.data['preview']['request']['runtime_target']),workspace=str(out),selection_anchor=0,build=_build(origin.data['preview'],record(pin),out),execution_authorized=False,final=False,default_enabled=False)
    write(out/'preparation.json',state);return dict(action='needs_actual_native_receipt',preparation=record(out/'preparation.json'),build=state['build'],resource_budget=context['resource_budget'],execution_authorized=False)

def _prepared(path):
    state=load(path);fields={'kind','schema_version','history576','stage_sources','domain_tool','context','runtime_target','workspace','selection_anchor','build','execution_authorized','final','default_enabled'}
    need(set(state)==fields and state['kind']=='mx5_independent512_original_preparation' and state['schema_version']==1,'wrong independent512 preparation schema')
    need(state['selection_anchor']==0 and all(state[n] is False for n in ('execution_authorized','final','default_enabled')),'preparation cannot change anchor/grant execution')
    context,origin,sources=_context(state['history576'],state['stage_sources'],state['domain_tool']);need(history._same(context,state['context']),'independent origin/history/budget changed')
    need(record(check_file(state['runtime_target']))==record(origin.data['preview']['request']['runtime_target']),'runtime target changed')
    pin=state['build']['pin'];need(check_file(pin)==Path(state['workspace'])/'artifact0/requested.pin.json' and load(check_file(pin))==origin.candidate(context['candidate_id'])['pin'],'only exact independent512 original ordinal0 is supported')
    need(state['build']==_build(origin.data['preview'],pin,state['workspace']),'normal WD0 build command changed')
    return state,origin,sources

def run_build(preparation):
    state,_,_=_prepared(preparation);job=state['build'];binary=Path(job['binary'])
    need(not binary.exists() and not list(binary.parent.glob(binary.name+'.*')) and not(binary.parent/'process').exists(),'actual/partial build exists; preserve and consume it')
    env=dict(os.environ,PYTHONPATH=str(ROOT/'python'),CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for key in list(env):
        if key.startswith('TILEMEGA_'):env.pop(key)
    env.pop('CUDACXX',None);old=os.getcwd()
    try:
        os.chdir(ROOT);process=common.run_command(job['argv'],binary.parent/'process',timeout=job['timeout_seconds'],env=env)
    finally:os.chdir(old)
    return dict(action='native_process_returned_consume_receipt',process=process,native_receipt=job['native_receipt'],no_resource_retry_inferred_from_exit=True,execution_authorized=False)

def consume(preparation,out,occupancy=None):
    state,origin,sources=_prepared(preparation);out=Path(out).resolve();need(not out.exists(),'preserve previous consumption output')
    job=state['build'];native_path=Path(job['native_receipt'])
    if not native_path.is_file():return dict(action='needs_actual_native_receipt',missing=str(native_path),build=job,execution_authorized=False)
    out.mkdir(parents=True)
    try:
        native_ref,native=receipts.load_native(native_path)
        need(native['returncode']==0 and native['files'].get('binary')==record(job['binary']) and native['files']['pin']==job['pin'],'compiler failure/different planned binary or pin cannot authorize resource retry')
        query_ref=record(occupancy) if occupancy else None
        bundle=dict(kind='mx5_thread_exception_artifact',schema_version=1,origin_sha256=origin.sha256,candidate_id=state['context']['candidate_id'],ordinal=0,native_receipt=native_ref,resources=record(job['resources']),query=query_ref,stage_sources=sources)
        artifact_path=out/'artifact.json';write(artifact_path,bundle);artifact_ref=record(artifact_path)
        artifact=history.old._artifact(artifact_ref,origin,state['context']['candidate_id'])
        kernel=artifact['resources'];mir=artifact['mir'];failed=history.old._failure(artifact)
        need(not kernel.get('callees') and mir['rings']>0 and not any(mir[n] for n in ('addrspace0','generic','calls','ring4160','ring_generic')),'actual MIR generic/call/ring/callee gate failed')
        need(failed or mir['private']==0,'fitting program contains private accesses')
        need(artifact['family']==state['context']['canonical_family_sha256'],'actual512 source/compiler/options differs from genuine canonical576 history')
        need(artifact['query'] is None or artifact['query']['resident_ctas']==1,'actual occupancy anomaly is not a width-retry authorization')
        files=artifact['files'];spec=dict(kind='mx5_plan_occupancy_inputs',schema_version=1,threads=512,context_kind='compile_receipt',binary=files['binary'],generated_source=files['generated_source'],resources=bundle['resources'],log=files['log'],context=native_ref,source_snapshot=files['source_snapshot'])
        history.old.occupancy.validate_inputs(spec,command=history.old.occupancy.run_text);inputs=out/'occupancy_inputs.json';write(inputs,spec)
        result=dict(kind='mx5_independent512_original_consumption',schema_version=1,preparation=record(preparation),history576=state['history576'],origin_preview=state['context']['origin_preview'],candidate_id=state['context']['candidate_id'],origin_candidates=state['context']['origin_candidates'],lineage512=[artifact_ref],catalogs=[],artifact=artifact_ref,occupancy_inputs=record(inputs),resource_budget=state['context']['resource_budget'],MIR=mir,selection_anchor=0,execution_authorized=False,final=False,default_enabled=False)
        if not failed and query_ref is None:
            result.update(action='needs_actual512_query',needs='owner runs trusted guarded query-only iterations0; consume same actual build again with --occupancy into NEW output; no new attempt/budget reset')
        else:
            bound=history.bind_final512(state['history576'],[artifact_ref],domain_tool=check_file(state['domain_tool']))
            binding_path=out/'binding.json';write(binding_path,bound.as_dict());result.update(action=bound.action.value,reason=bound.reason,binding=record(binding_path))
            result['needs']=('only existing deterministic next-width proposal within genuine remaining shared budget; exact-source independent3fresh changed-width costs and descendant proof still required; no child prepared' if bound.action==history.Action.NEXT_512 else 'existing bind_final512 stops here; preserve failure, no extra retry, budget reset or approval' if bound.action==history.Action.BLOCKED else 'substantive actual D7 v2 evidence/I1/checks and existing resolved_top4_v2.capture_plan/all4 protocol remain mandatory; no512 approval, replacement origin, width costs or retry history created')
        write(out/'result.json',result);return result
    except (ValueError,KeyError,TypeError,OSError) as error:
        result=dict(action='blocked_evidence',reason=str(error),preparation=record(preparation),native_receipt=record(native_path),execution_authorized=False,final=False,default_enabled=False);write(out/'failure.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare')
    for name in ('history','stage-sources','domain-tool','out'):a.add_argument('--'+name,type=Path,required=True)
    a=sub.add_parser('run-build');a.add_argument('--preparation',type=Path,required=True)
    a=sub.add_parser('consume');a.add_argument('--preparation',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--occupancy',type=Path)
    a=p.parse_args()
    try:
        if a.mode=='prepare':result=prepare(a.history,a.stage_sources,a.domain_tool,a.out)
        elif a.mode=='run-build':result=run_build(a.preparation)
        else:result=consume(a.preparation,a.out,a.occupancy)
    except (ValueError,KeyError,TypeError,OSError) as error:result=dict(action='blocked_evidence',reason=str(error),execution_authorized=False,final=False,default_enabled=False)
    print(json.dumps(result,indent=2));return int(result['action']=='blocked_evidence')
if __name__=='__main__':raise SystemExit(main())
