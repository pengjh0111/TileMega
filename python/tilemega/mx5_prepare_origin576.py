"""Prepare one exact independent576 original; no GPU, retries, or execution grants."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

from . import mx5_plan_resources as policy
from . import mx5_plan_receipt as receipts
from . import mx5_resolved_top4 as resolved
from . import mx5_resolved_runner as runner
from . import mx5_select as common

ROOT=Path(common.__file__).resolve().parents[2]
load,record,check_file,write=common.load,common.record,common.check_file,common.write

def need(value,reason):
    if not value:raise ValueError(reason)

def _origin(preview_ref,candidate,sources_ref,domain_ref):
    origin=resolved.Qualified576Origin.capture(preview_ref)
    origin.candidate(candidate)
    preview=origin.data['preview'];table=load(check_file(preview['stage_table']))
    domain=resolved._complete_domain(preview,table,check_file(domain_ref))
    sources=load(check_file(sources_ref))
    need(isinstance(sources,list) and sources,'actual stage archive inventory required')
    digests=[s['source_sha256'] for s in sources]
    need(len(digests)==len(set(digests)) and set(digests)==set(table['scope']['source_sha256'].values()),'stage source inventory differs from independent576 epoch')
    for source in sources:check_file(source['archive'])
    return origin,domain,sources

def _build(preview,pin,workspace):
    binary=Path(workspace)/'artifact0/plan.so'
    argv=runner._compile_command(preview,pin,binary);argv[-1]='0'
    return dict(argv=argv,binary=str(binary),native_receipt=str(binary)+'.compile_receipt.json',resources=str(binary)+'.resources.json',pin=pin,ordinal=0,watchdog=0,timeout_seconds=900)

def prepare(preview,candidate,sources,domain,out):
    out=Path(out).resolve();need(not out.exists(),'preserve previous preparation output')
    pr,sr,dr=record(preview),record(sources),record(domain)
    origin,checked,_=_origin(pr,candidate,sr,dr)
    target=record(origin.data['preview']['request']['runtime_target'])
    out.mkdir(parents=True);folder=out/'artifact0';folder.mkdir()
    pin=folder/'requested.pin.json';write(pin,origin.candidate(candidate)['pin'])
    state=dict(kind='mx5_true576_original_preparation',schema_version=1,origin_preview=pr,origin_sha256=origin.sha256,candidate_id=candidate,origin_candidates=origin.data['preview']['candidates'],stage_sources=sr,domain_tool=dr,domain=checked,runtime_target=target,workspace=str(out),selection_anchor=0,build=_build(origin.data['preview'],record(pin),out),execution_authorized=False,final=False,default_enabled=False)
    write(out/'preparation.json',state)
    return dict(action='needs_actual_native_receipt',preparation=record(out/'preparation.json'),build=state['build'],execution_authorized=False)

def _prepared(path):
    state=load(path)
    fields={'kind','schema_version','origin_preview','origin_sha256','candidate_id','origin_candidates','stage_sources','domain_tool','domain','runtime_target','workspace','selection_anchor','build','execution_authorized','final','default_enabled'}
    need(set(state)==fields and state['kind']=='mx5_true576_original_preparation' and state['schema_version']==1,'wrong true576 preparation schema')
    need(state['selection_anchor']==0 and all(state[n] is False for n in ('execution_authorized','final','default_enabled')),'preparation cannot change anchor/grant execution')
    origin,domain,sources=_origin(state['origin_preview'],state['candidate_id'],state['stage_sources'],state['domain_tool'])
    need(origin.sha256==state['origin_sha256'] and origin.data['preview']['candidates']==state['origin_candidates'] and domain==state['domain'],'original IDs/ranks/predictions/domain changed')
    need(record(check_file(state['runtime_target']))==record(origin.data['preview']['request']['runtime_target']),'runtime target changed')
    pin=state['build']['pin'];need(check_file(pin)==Path(state['workspace'])/'artifact0/requested.pin.json' and load(check_file(pin))==origin.candidate(state['candidate_id'])['pin'],'only the exact original candidate pin is supported')
    need(state['build']==_build(origin.data['preview'],pin,state['workspace']),'normal WD0 native build job changed')
    return state,origin,sources

def run_build(preparation):
    state,_,_=_prepared(preparation);job=state['build'];binary=Path(job['binary'])
    need(not binary.exists() and not list(binary.parent.glob(binary.name+'.*')) and not(binary.parent/'process').exists(),'actual/partial build exists; preserve and consume it')
    env=dict(os.environ,PYTHONPATH=str(ROOT/'python'),CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for key in list(env):
        if key.startswith('TILEMEGA_'):env.pop(key)
    env.pop('CUDACXX',None)
    old=os.getcwd()
    try:
        os.chdir(ROOT);process=common.run_command(job['argv'],binary.parent/'process',timeout=job['timeout_seconds'],env=env)
    finally:os.chdir(old)
    return dict(action='native_process_returned_consume_receipt',process=process,native_receipt=job['native_receipt'],no_resource_retry_inferred_from_exit=True,execution_authorized=False)

def consume(preparation,out,occupancy=None):
    state,origin,sources=_prepared(preparation);out=Path(out).resolve();need(not out.exists(),'preserve previous consumption output')
    job=state['build'];native_path=Path(job['native_receipt'])
    if not native_path.is_file():
        return dict(action='needs_actual_native_receipt',missing=str(native_path),build=job,execution_authorized=False)
    out.mkdir(parents=True)
    try:
        native_ref,native=receipts.load_native(native_path);files=dict(native['files'],compile_receipt=native_ref)
        need(native['returncode']==0 and files.get('binary')==record(job['binary']) and files['pin']==job['pin'],'compiler failure or different actual output/pin; no resource retry inferred')
        snapshot=load(check_file(files['source_snapshot']));resolved.exception._normal_options(native,files,snapshot,0)
        report=policy.inspect(check_file(files['binary']),check_file(files['log']),576);resource_ref=record(job['resources'])
        need(report==load(check_file(resource_ref)),'actual resource report differs from ELF/same-compile MIR')
        kernel=report['kernel'];need(kernel['static_smem_bytes']==0 and not kernel.get('callees'),'static smem/callee failure is not an MT/frame retry')
        mir=resolved.exception._mir(files['log'],kernel['name'])
        need(mir['rings']>0 and not any(mir[n] for n in ('addrspace0','generic','calls','ring4160','ring_generic')),'actual MIR generic/call/ring gate failed')
        failed=kernel['registers']>227 or kernel['stack_bytes']>0
        need(failed or mir['private']==0,'fitting program has private accesses')
        query_ref=record(occupancy) if occupancy else None
        if query_ref:
            query=resolved.exception._query(query_ref,files,resource_ref,576,report)
            need(query['resident_ctas']==1,'occupancy failure is not a width-retry authorization')
        ledger=policy.PlanLedger.start(origin,state['candidate_id'])
        attempt=receipts.assemble(native_path,ledger,stage_sources=sources,resources=check_file(resource_ref),occupancy=check_file(query_ref) if query_ref else None)
        attempt_path=out/'attempt.json';write(attempt_path,attempt);ledger=ledger.append_attempt(attempt_path)
        need(not ledger.terminal_error,'strict PlanLedger receipt rejected: '+str(ledger.terminal_error))
        ledger_path=out/'ledger.json';ledger.write(ledger_path)
        spec=dict(kind='mx5_plan_occupancy_inputs',schema_version=1,threads=576,context_kind='compile_receipt',binary=files['binary'],generated_source=files['generated_source'],resources=resource_ref,log=files['log'],context=native_ref,source_snapshot=files['source_snapshot'])
        resolved.exception.occupancy.validate_inputs(spec,command=resolved.exception.occupancy.run_text)
        inputs=out/'occupancy_inputs.json';write(inputs,spec)
        action=policy.next_plan_retry(ledger)
        result=dict(kind='mx5_true576_original_consumption',schema_version=1,preparation=record(preparation),candidate_id=state['candidate_id'],selection_anchor=0,origin_candidates=state['origin_candidates'],attempt=record(attempt_path),ledger=record(ledger_path),occupancy_inputs=record(inputs),MIR=mir,policy_action=dict(action=action.kind.value,reason=action.reason,details=action.details.data),execution_authorized=False,final=False,default_enabled=False)
        if not failed and query_ref is None:
            need(ledger.attempts[-1].data['verified']['reason']=='resource-fit program lacks bound actual occupancy metadata','unexpected blocked ledger state')
            result.update(action='needs_actual576_query',needs='owner executes trusted guarded query-only iterations0 for occupancy_inputs, then consume again with --occupancy into NEW output; no new build')
        elif action.kind==policy.ActionKind.RESOURCE_FIT:
            plan=resolved.capture_plan(state['origin_preview'],state['candidate_id'],record(ledger_path));plan_path=out/'resolved_plan.json';write(plan_path,plan)
            result.update(action='review_ready_true576_plan',resolved_plan=record(plan_path),needs='collect all four original entries in ID order with existing capture_protocol/validate_resolved_top4; this single plan is not execution authorization')
        else:
            needs=('independently measured exact-source three-fresh-process changed-width costs required by this exact plan_resource_retry; no automatic reduction/build/history' if action.kind==policy.ActionKind.NEEDS_STAGE_MEASUREMENTS else 'existing next_plan_retry stops here; no D7 proof, extra retry history, or512 authorization is synthesized')
            result.update(action=action.kind.value,needs=needs)
        write(out/'result.json',result);return result
    except (ValueError,KeyError,TypeError,OSError) as error:
        result=dict(action='blocked_evidence',reason=str(error),preparation=record(preparation),native_receipt=record(native_path),execution_authorized=False,final=False,default_enabled=False)
        write(out/'failure.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare')
    for name in ('preview','stage-sources','domain-tool','out'):a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--candidate',type=int,choices=range(4),required=True)
    a=sub.add_parser('run-build');a.add_argument('--preparation',type=Path,required=True)
    a=sub.add_parser('consume');a.add_argument('--preparation',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--occupancy',type=Path)
    a=p.parse_args()
    try:
        if a.mode=='prepare':result=prepare(a.preview,a.candidate,a.stage_sources,a.domain_tool,a.out)
        elif a.mode=='run-build':result=run_build(a.preparation)
        else:result=consume(a.preparation,a.out,a.occupancy)
    except (ValueError,KeyError,TypeError,OSError) as error:result=dict(action='blocked_evidence',reason=str(error),execution_authorized=False,final=False,default_enabled=False)
    print(json.dumps(result,indent=2));return int(result['action']=='blocked_evidence')
if __name__=='__main__':raise SystemExit(main())
