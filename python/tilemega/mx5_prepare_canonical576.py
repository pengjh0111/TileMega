"""Prepare/resume real canonical576 diagnostics; no GPU or execution qualification."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys

from . import mx5_canonical576_history as history
from . import mx5_select as common
from . import mx5_plan_receipt as receipts
from . import mx5_resolved_runner as runner

ROOT=Path(common.__file__).resolve().parents[2]
load=common.load
record=common.record
check_file=common.check_file
write=common.write


def need(value,reason):
    if not value:raise ValueError(reason)


def _context(origin,candidate,p0,sources,domain,gpu,workspace):
    need(type(candidate) is int and candidate in range(4),'exact original candidate ID0..3 required')
    need(type(gpu) is int and gpu in (7,6,5,4),'explicit supported query card required')
    preview=common.validate_preview(check_file(origin),allow_schema3_diagnostic=True)
    need(preview['request']['threads']==512,'bounded canonical stepper starts from an independent512 origin; true576 PlanLedger preparation remains separate')
    inventory=load(check_file(sources));need(isinstance(inventory,list) and inventory,'actual stage source inventory required')
    digests=[item['source_sha256'] for item in inventory]
    need(len(set(digests))==len(digests),'ambiguous stage source archives')
    for item in inventory:check_file(item['archive'])
    table=load(check_file(preview['stage_table']));need(set(digests)==set(table['scope']['source_sha256'].values()),'stage archive inventory differs from original qualified source epoch')
    helpers={name:record(path) for name,path in dict(preparer=Path(__file__),history=Path(history.__file__),receipt_parser=Path(receipts.__file__),
        occupancy=Path(history.old.occupancy.__file__),runner=Path(runner.__file__),guard=ROOT/'scripts/mx_gpu_job.sh').items()}
    return dict(origin_preview=origin,candidate_id=candidate,p0_bundle=p0,stage_sources=sources,domain_tool=record(domain),gpu=gpu,workspace=str(Path(workspace).resolve()),
                runtime_target=record(preview['request']['runtime_target']),helpers=helpers,
                gpu_job_log=str(Path(os.environ.get('TILEMEGA_GPU_JOB_LOG') or ROOT/'runs/mx5/gpu_jobs.log').resolve()))


def _check_context(context):
    for name in ('origin_preview','p0_bundle','stage_sources','domain_tool','runtime_target'):check_file(context[name])
    for ref in context['helpers'].values():check_file(ref)
    actual=_context(context['origin_preview'],context['candidate_id'],context['p0_bundle'],context['stage_sources'],check_file(context['domain_tool']),context['gpu'],context['workspace'])
    actual['gpu_job_log']=context['gpu_job_log']
    need(actual==context,'prepared context/source/tool/origin changed')


def _audit(context,attempts):
    return history.audit_history576(context['origin_preview'],context['candidate_id'],context['p0_bundle'],attempts,domain_tool=check_file(context['domain_tool']))


def _pin(audit):
    if audit.action==history.Action.NEEDS_ATTEMPT:return audit.data['canonical_primary_pin']
    need(audit.action==history.Action.NEXT_576,'no canonical576 build is prescribed: '+audit.reason)
    return audit.data['next_pin']


def _build_job(context,pin_ref,ordinal):
    preview=load(check_file(context['origin_preview']));binary=Path(context['workspace'])/('artifact%d'%ordinal)/'plan.so'
    argv=runner._compile_command(preview,pin_ref,binary);argv[-1]='0'
    return dict(argv=argv,timeout_seconds=900,binary=str(binary),native_receipt=str(binary)+'.compile_receipt.json',resources=str(binary)+'.resources.json',
                pin=pin_ref,ordinal=ordinal,watchdog=0,execution='explicit_cpu_native_build_only')


def _query_job(context,folder,inputs):
    output=Path(folder)/'actual_query.json';guard=Path(folder)/'query_guard'
    return dict(argv=[str(ROOT/'scripts/mx_gpu_job.sh'),str(context['gpu']),'300',str(guard),sys.executable,'-m','tilemega.mx5_plan_occupancy',
                     '--inputs',str(check_file(inputs)),'--out',str(output),'--query-only','--iterations','0'],
                output=str(output),inputs=inputs,environment=dict(TILEMEGA_GPU_JOB_LOG=context['gpu_job_log'],NEEDS_MIB='2048'),
                execution='owner_executes_actual_guarded_zero_launch_query',execution_authorized=False)


def _native_process(job):
    path=Path(job['binary']).parent/'process/process.json'
    if not path.exists():return None
    process=load(path);need(process.get('argv')==job['argv'] and type(process.get('returncode')) is int,'actual native process command/status differs')
    report=load(job['resources'])['kernel'];failed=report['registers']>227 or report['stack_bytes']>0
    need(process['returncode']==(1 if failed else 0),'non-MT/frame native process failure or inconsistent exit; no retry authorized')
    return record(path)


def _bundle(context,job,native,query=None):
    return dict(kind='mx5_canonical576_attempt',schema_version=2,origin_sha256=history.resolved.Qualified512Origin.capture(context['origin_preview']).sha256,
        candidate_id=context['candidate_id'],ordinal=job['ordinal'],native_receipt=native,resources=record(job['resources']),query=query,
        stage_sources=load(check_file(context['stage_sources'])))


def _query_inputs(native_ref,native,resources):
    f=native['files']
    return dict(kind='mx5_plan_occupancy_inputs',schema_version=1,threads=576,context_kind='compile_receipt',binary=f['binary'],generated_source=f['generated_source'],
                resources=resources,log=f['log'],context=native_ref,source_snapshot=f['source_snapshot'])


def _step(context,attempts,audit,previous):
    return dict(kind='mx5_canonical576_preparation',schema_version=1,context=context,attempts=attempts,audit=audit.as_dict(),previous_step=previous,
                execution_authorized=False,final=False,default_enabled=False,selection_anchor=0)


def _build_step(context,attempts,audit,previous,folder):
    pin=_pin(audit);ordinal=len(attempts);artifact=Path(context['workspace'])/('artifact%d'%ordinal)
    need(not artifact.exists(),'existing artifact directory is preserved');artifact.mkdir(parents=True)
    path=artifact/'requested.pin.json';created=False
    try:
        with path.open('x') as output:created=True;json.dump(pin,output,indent=2,allow_nan=False);output.write('\n')
        state=_step(context,attempts,audit,previous);state.update(phase='needs_actual_native_receipt',build=_build_job(context,record(path),ordinal))
        write(Path(folder)/'step.json',state);return state
    except BaseException:
        if created:path.unlink()
        if not any(artifact.iterdir()):artifact.rmdir()
        raise


def _validate_step(path,seen=None):
    path=Path(path).resolve();seen=set() if seen is None else seen
    need(str(path) not in seen and len(seen)<10,'cyclic/oversized preparation chain');seen.add(str(path))
    state=load(path);need(state['kind']=='mx5_canonical576_preparation' and state['schema_version']==1 and all(state[k] is False for k in ('execution_authorized','final','default_enabled')) and state['selection_anchor']==0,'wrong preparation schema/policy')
    fields={'kind','schema_version','context','attempts','audit','previous_step','execution_authorized','final','default_enabled','selection_anchor','phase'}
    extra={'needs_actual_native_receipt':{'build'},'needs_actual576_query':{'build','native_receipt','native_process','pending_attempt','query_inputs','query_job'}}.get(state['phase'],{'history576','needs'})
    if 'independent576_origin_job' in state:
        need(state['phase']==history.Action.USE_576.value,'only actual fitting576 can request its independent solver origin')
        extra=extra|{'independent576_origin_job'}
    need(set(state)==fields|extra,'unexpected preparation fields; no approval/cost override flags')
    context=state['context'];_check_context(context);attempts=state['attempts'];need(isinstance(attempts,list) and len(attempts)<=4,'unknown/reset attempt history')
    parent=None
    if state['previous_step']:
        parent=_validate_step(check_file(state['previous_step']),seen);need(parent['context']==context,'context changed across resume')
        if parent['phase']=='needs_actual_native_receipt':
            need(state['phase']=='needs_actual576_query' and attempts==parent['attempts'] and state['build']==parent['build'],'resume lost parent build/history')
        else:
            need(state['phase']!='needs_actual576_query','resume skipped its prescribed build preparation')
            need(parent['phase']=='needs_actual576_query' and attempts[:-1]==parent['attempts'] and len(attempts)==len(parent['attempts'])+1,'resume stripped/reset actual576 history')
            expected=_bundle(context,parent['build'],parent['native_receipt'],record(parent['query_job']['output']))
            need(load(check_file(attempts[-1]))==expected,'consumed attempt differs from actual planned native/query artifacts')
    else:need(not attempts and state['phase']=='needs_actual_native_receipt','only an empty original PRIMARY preparation can start a chain')
    if state['phase']=='needs_actual576_query':
        job=state['build'];native_ref,native=receipts.load_native(check_file(state['native_receipt']))
        need(native_ref==state['native_receipt'] and native['returncode']==0 and native['files'].get('binary')==record(job['binary']) and native['files']['pin']==job['pin'],'native receipt differs from actual planned output/pin')
        need(state['native_process']==_native_process(job),'native process observation changed')
        pending=_bundle(context,job,native_ref);need(load(check_file(state['pending_attempt']))==pending,'pending actual attempt changed')
        audit=_audit(context,[*attempts,state['pending_attempt']]);need(audit.action!=history.Action.BLOCKED,audit.reason)
        spec=_query_inputs(native_ref,native,pending['resources']);need(load(check_file(state['query_inputs']))==spec,'query inputs changed')
        history.old.occupancy.validate_inputs(spec,command=history.old.occupancy.run_text)
        need(state['query_job']==_query_job(context,path.parent,state['query_inputs']),'query command/source/card changed')
    else:
        audit=_audit(context,attempts)
        if state['phase']=='needs_actual_native_receipt':
            job=state['build'];need(job==_build_job(context,job['pin'],len(attempts)) and load(check_file(job['pin']))==_pin(audit),'noncanonical build command/pin/ordinal')
            need(check_file(job['pin'])==Path(context['workspace'])/('artifact%d'%len(attempts))/'requested.pin.json','prepared pin path changed')
        else:
            need(state['phase']==audit.action.value and audit.action in (history.Action.USE_576,history.Action.HISTORY),'unsupported terminal outcome')
            need(load(check_file(state['history576']))==audit.as_dict(),'terminal history material differs from live proof')
            if 'independent576_origin_job' in state:
                need(audit.action==history.Action.USE_576,'failed576 cannot create a fitting576 reroute')
                preview=load(check_file(context['origin_preview']));need('stage_catalog' in preview,'no independent576 source table is bound')
                request=copy.deepcopy(preview);request['request']['threads']=576
                expected=dict(argv=common.native_solve_command(request,path.parent/'independent576/preview'),timeout_seconds=900,
                    purpose='fresh_independent576_solver_origin; no512 pin/rank/prediction transferred')
                need(state['independent576_origin_job']==expected,'independent576 origin command changed')
    need(state['audit']==audit.as_dict(),'saved audit/history/budget changed')
    return state


def prepare(origin,candidate,p0,sources,domain,gpu,out):
    out=Path(out).resolve();need(not out.exists(),'existing preparation output is preserved')
    context=_context(record(origin),candidate,record(p0),record(sources),domain,gpu,out);audit=_audit(context,[])
    if audit.action!=history.Action.NEEDS_ATTEMPT:
        out.mkdir(parents=True);result=dict(action=audit.action.value,reason=audit.reason,details=audit.as_dict(),execution_authorized=False)
        write(out/'needs_materials.json',result);return result
    out.mkdir(parents=True);state=_build_step(context,[],audit,None,out)
    return dict(action=state['phase'],step=record(out/'step.json'),build=state['build'],execution_authorized=False)


def run_build(step):
    state=_validate_step(step);need(state['phase']=='needs_actual_native_receipt','this step has no prescribed native build')
    job=state['build'];binary=Path(job['binary'])
    need(not binary.exists() and not list(binary.parent.glob(binary.name+'.*')) and not (binary.parent/'process').exists(),'actual or partial build output exists; consume/preserve it instead of overwriting')
    env=dict(os.environ,PYTHONPATH=str(ROOT/'python'),CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='')
    for name in ('TILEMEGA_MX_DIAG','TILEMEGA_TRACE'):env.pop(name,None)
    old=os.getcwd()
    try:
        os.chdir(ROOT);process=common.run_command(job['argv'],Path(job['binary']).parent/'process',timeout=job['timeout_seconds'],env=env)
    finally:os.chdir(old)
    return dict(action='native_process_returned_consume_actual_receipt',process=process,process_file=record(Path(job['binary']).parent/'process/process.json'),
                native_receipt=job['native_receipt'],no_fit_or_retry_inferred_from_process_exit=True,execution_authorized=False)


def advance(step,out):
    out=Path(out).resolve();need(not out.exists(),'existing resume output is preserved');state=_validate_step(step)
    context=state['context'];previous=record(step);attempts=state['attempts'];phase=state['phase']
    need(phase in ('needs_actual_native_receipt','needs_actual576_query'),'terminal proof needs its explicit next consumer, not another576 retry')
    job=state['build'];required=Path(job['native_receipt']) if phase=='needs_actual_native_receipt' else Path(state['query_job']['output'])
    if not required.is_file():
        process=Path(job['binary']).parent/'process/process.json'
        if phase=='needs_actual_native_receipt' and process.exists() and load(process)['returncode']!=0:
            return dict(action='blocked_evidence',reason='native process failed without an actual compile receipt; no MT/frame retry inferred',process=record(process),execution_authorized=False)
        return dict(action=phase,step=previous,missing=[str(required)],build=state.get('build'),query_job=state.get('query_job'),execution_authorized=False)
    out.mkdir(parents=True)
    try:
        if phase=='needs_actual_native_receipt':
            native_ref,native=receipts.load_native(required)
            need(native['returncode']==0 and native['files'].get('binary')==record(job['binary']) and native['files']['pin']==job['pin'],'failed compilation or changed planned artifact/pin; no resource retry')
            process_ref=_native_process(job)
            bundle=_bundle(context,job,native_ref);pending=out/'pending_actual_attempt.json';write(pending,bundle)
            audit=_audit(context,[*attempts,record(pending)]);need(audit.action!=history.Action.BLOCKED,audit.reason)
            spec=_query_inputs(native_ref,native,bundle['resources']);history.old.occupancy.validate_inputs(spec,command=history.old.occupancy.run_text)
            inputs=out/'query_inputs.json';write(inputs,spec)
            next_state=_step(context,attempts,audit,previous);next_state.update(phase='needs_actual576_query',build=job,native_receipt=native_ref,native_process=process_ref,pending_attempt=record(pending),query_inputs=record(inputs),query_job=_query_job(context,out,record(inputs)))
        else:
            raw=load(required);need(raw.get('inputs_manifest')==state['query_job']['inputs'],'actual query used another inputs manifest')
            need(raw.get('guard',{}).get('physical_gpu')==str(context['gpu']),'actual query ran on another physical card')
            bundle=_bundle(context,job,state['native_receipt'],record(required));path=out/'consumed_actual_attempt.json';write(path,bundle);attempts=[*attempts,record(path)]
            audit=_audit(context,attempts);need(audit.action!=history.Action.BLOCKED,audit.reason)
            if audit.action==history.Action.NEXT_576:
                next_state=_build_step(context,attempts,audit,previous,out)
                return dict(action=next_state['phase'],step=record(out/'step.json'),build=next_state['build'],execution_authorized=False)
            need(audit.action in (history.Action.USE_576,history.Action.HISTORY),'actual query has not completed canonical576 history: '+audit.reason)
            next_state=_step(context,attempts,audit,previous);next_state['phase']=audit.action.value
            proof=out/'history576.json';write(proof,audit.as_dict());next_state['history576']=record(proof)
            if audit.action==history.Action.USE_576:
                preview=load(check_file(context['origin_preview']))
                if 'stage_catalog' in preview:
                    request=copy.deepcopy(preview);request['request']['threads']=576
                    next_state['independent576_origin_job']=dict(argv=common.native_solve_command(request,out/'independent576/preview'),timeout_seconds=900,
                        purpose='fresh_independent576_solver_origin; no512 pin/rank/prediction transferred')
                next_state['needs']=['independent true576 native solver preview and its own original4; no512 selection or prediction transfer']
            else:
                next_state['needs']=['actual independent512 original/descendant receipts and queries within remaining shared budget',
                    'exact3fresh source-qualified changed-width costs bound to actual plan_resource_retry, without resource_selection',
                    'substantive qualified-history-v2 D7 source/MIR/budget/I1 evidence; no boolean or empty report']
        write(out/'step.json',next_state)
        return dict(action=next_state['phase'],step=record(out/'step.json'),query_job=next_state.get('query_job'),history576=next_state.get('history576'),
                    independent576_origin_job=next_state.get('independent576_origin_job'),needs=next_state.get('needs'),execution_authorized=False)
    except Exception as error:
        result=dict(kind='mx5_canonical576_preparation_failure',action='blocked_evidence',reason=str(error),previous_step=previous,
                    observed_input=record(required),execution_authorized=False,final=False,default_enabled=False)
        write(out/'failure.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare')
    for flag in ('origin','p0','stage-sources','domain-tool','out'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--candidate',type=int,choices=range(4),required=True);p.add_argument('--gpu',type=int,choices=(7,6,5,4),required=True)
    p=sub.add_parser('advance');p.add_argument('--step',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('run-build');p.add_argument('--step',type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.mode=='prepare':result=prepare(args.origin,args.candidate,args.p0,args.stage_sources,args.domain_tool,args.gpu,args.out)
        elif args.mode=='advance':result=advance(args.step,args.out)
        else:result=run_build(args.step)
    except (ValueError,KeyError,TypeError,OSError) as error:
        result=dict(action='blocked_evidence',reason=str(error),execution_authorized=False,final=False,default_enabled=False)
    print(json.dumps(result));return 2 if result['action']=='blocked_evidence' else 0


if __name__=='__main__':raise SystemExit(main())
