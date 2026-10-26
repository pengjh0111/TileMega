"""Explicit, purpose-limited ResolvedTop4 v2 execution and result consumption."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys

from . import mx5_select as common
from . import mx5_resolved_top4_v2 as resolved
from . import mx5_thread_exception as checks
from .mx5_plan_receipt import load_native
from .mx5_resources import inspect
from .build.identity import verify,validate_snapshot_inputs

load=common.load
record=common.record
check_file=common.check_file
write=common.write
run_command=common.run_command
ROOT=Path(common.__file__).resolve().parents[2]


def need(value,reason):
    if not value:raise ValueError(reason)


def _ready(protocol,domain):
    result=resolved.validate_resolved_top4(protocol,domain_tool=domain)
    need(result.action==resolved.v1.Action.READY,'resolved v2 evidence is not READY: '+result.reason)
    return result.data


def _identity(so,pin,request,producer,watchdog,snapshot=None):
    identity=verify(so);validate_snapshot_inputs(identity['source']['root'],identity['source'])
    need(identity['source']['compiler_sha256']==producer['sha256'],'actual serving producer differs from origin')
    if snapshot is not None:need(identity['source']==snapshot,'identity source differs from actual native compile snapshot')
    p=identity['plan'];c=pin['mx_decode_v2']
    need(p.get('executor')=='mx_static' and p.get('version')==5 and p.get('watchdog')==watchdog
         and not p.get('trace_stage') and not p.get('diag') and not identity.get('trace'),'unsupported/instrumented serving artifact')
    need(all(p.get(k)==c[k] for k in ('threads','b_min','attn_prefetch')),'artifact attention/threads differ from actual pin')
    for name in common.CLASSES[:-1]:
        need(all(p['classes'][name].get(k)==v for k,v in c[name].items()),'artifact effective coordinate differs: '+name)
    need(all(p.get(k)==request[k] for k in ('capacity','past_lo','past_hi')) and p['batch_lo']==p['batch_hi']==request['batch'],'artifact decode request differs')
    return identity


def _prefill(path,watchdog):
    identity=verify(path);validate_snapshot_inputs(identity['source']['root'],identity['source'])
    need(identity['plan'].get('phase')=='prefill' and not identity.get('trace'),'invalid prefill identity')
    value=identity.get('defines',{}).get('TILEMEGA_WATCHDOG',str(identity['plan'].get('watchdog',0)))
    need(str(value)==str(watchdog),'prefill watchdog role differs')
    return identity


def _artifact(ready,index,preview):
    final=ready['plans'][index]['final'];so=check_file(final['binary'])
    result=resolved.check_measurement_artifact(ready['protocol'],index,final['binary'],domain_tool=check_file(ready['domain']['domain_tool']))
    need(result.action==resolved.v1.Action.READY,'actual WD0 artifact no longer bound to resolved proof')
    identity=_identity(so,final['pin'],preview['request'],preview['producer'],0,load(check_file(final['source_snapshot'])))
    return dict(binary=final['binary'],identity_id=identity['artifact_id'],pin=final['pin'],source_snapshot=final['source_snapshot'])


def _fresh(preview,descriptor,process):
    need(process['returncode']==0,'native exact top4 revalidation failed')
    argv=process['argv'];prefix=str(check_file(descriptor)).removesuffix('.mx5_candidates.json')
    need(argv==common.native_solve_command(preview,prefix),'native revalidation command differs from exact original request')
    fresh=common.validate_preview(check_file(descriptor),allow_schema3_diagnostic=True)
    for key in ('stage_table','input','producer','request'):
        need(fresh[key]==preview[key],'fresh native origin '+key+' differs')
    need(len(fresh['candidates'])==4,'fresh native solver did not retain exact four')
    for old,new in zip(preview['candidates'],fresh['candidates']):
        need(all(old[k]==new[k] for k in ('index','pin','ranks','predicted_tpot_ns','predicted_stage_ns','evidence')),'fresh native exact top4/anchor changed')
    return fresh


def _compile_command(preview,pin,out):
    q=preview['request']
    return [str(check_file(preview['producer'])),'compile',str(check_file(preview['input'])),str(out),
        '--serving','decode','--emit','serving','--frontend','decoder','--batch',str(q['batch']),
        '--capacity',str(q['capacity']),'--past-range',str(q['past_lo'])+':'+str(q['past_hi']),
        '--runtime-target',q['runtime_target'],'--evaluate-configs',str(check_file(pin)),
        '--selection','predicted','--watchdog','1']


def _source_key(snapshot):return {k:v for k,v in snapshot.items() if k!='patch'}


def _watchdog(so,pin,preview,wd0_snapshot,query_ref=None):
    native_ref,native=load_native(str(so)+'.compile_receipt.json');f=dict(native['files'],compile_receipt=native_ref)
    need(native['returncode']==0 and check_file(f['binary'])==Path(so).resolve() and load(check_file(f['pin']))==pin,'WD1 did not compile the exact effective pin/output')
    snapshot=load(check_file(f['source_snapshot']));need(_source_key(snapshot)==_source_key(wd0_snapshot),'WD1 source snapshot/producer differs from executed WD0')
    need(f['source_patch']['sha256']==snapshot['worktree_diff_sha256'],'WD1 patch content changed')
    checks._normal_options(native,f,snapshot,1)
    report=inspect(so,str(so)+'.ptxas.log',pin['mx_decode_v2']['threads'])
    need(not report['violations'] and report==load(str(so)+'.resources.json'),'WD1 actual resource gate failed')
    mir=checks._mir(f['log'],report['kernel']['name'])
    need(mir['rings']>0 and not any(mir[k] for k in ('private','addrspace0','generic','calls','ring4160','ring_generic')),'WD1 actual MIR gate failed')
    identity=_identity(so,pin,preview['request'],preview['producer'],1,snapshot)
    spec=dict(kind='mx5_plan_occupancy_inputs',schema_version=1,threads=pin['mx_decode_v2']['threads'],context_kind='compile_receipt',
        binary=f['binary'],generated_source=f['generated_source'],resources=record(str(so)+'.resources.json'),log=f['log'],context=native_ref,source_snapshot=f['source_snapshot'])
    if query_ref is not None:
        query=checks._query(query_ref,f,spec['resources'],spec['threads'],report)
        need(query['resident_ctas']==1,'WD1 does not have actual resident1')
    return dict(binary=f['binary'],native_receipt=native_ref,resources=spec['resources'],source_snapshot=f['source_snapshot'],
                identity_id=identity['artifact_id'],pin=pin,query=query_ref),spec


def _guard(process,gpu,timeout):
    need(process['returncode']==0 and process['argv'][:3]==[str(ROOT/'scripts/mx_gpu_job.sh'),str(gpu),str(timeout)],'wrong/failed guarded command or card/timeout')
    folder=Path(process['argv'][3]);result=load(folder/'guard_result.json')
    need(type(result.get('code')) is int and result['code']==0,'guard rejected the actual process')
    pid=int((folder/'pgid').read_text());need(pid>1,'guard child PID missing/invalid')
    return folder,pid


def _guard_files(process):
    folder=Path(process['argv'][3]);refs={name:record(folder/name) for name in ('guard_result.json','pgid','guard.jsonl')}
    rows=[json.loads(line) for line in check_file(refs['guard.jsonl']).read_text().splitlines() if line]
    need(rows and any(row.get('phase')=='preflight' for row in rows) and rows[-1].get('phase')=='after' and rows[-1].get('exit_code')==0,'guard lacks actual successful telemetry sequence')
    return refs


def _process_file(directory,process):
    ref=record(Path(directory)/'process.json');need(load(check_file(ref))==process,'actual process record differs from returned command');return ref


def _check_process(process,ref):need(load(check_file(ref))==process,'persisted process command/status changed')


def _correctness(folder,batch,model):
    c2=load(folder/'c2/mode_check.json');c1=load(folder/'c1/hf_check.json')
    need(c2.get('pass') is True and c2.get('guard_mode')=='exclusive' and c2['batch']==batch and c2['steps']==512
         and c2['mismatches']=={'L1_separate':0,'L1_loop':0},'C2 failed or used another protocol')
    tokens=[load(folder/('c2/tokens_'+mode+'.json')) for mode in ('L1_separate','L1_loop')]
    need(tokens[0]==tokens[1] and len(tokens[0])==batch and all(len(row)==512 and all(type(t) is int and t>=0 for t in row) for row in tokens[0]),'C2 raw token arms differ')
    need(c1.get('pass') is True and Path(c1['model']).resolve()==model and c1['batch']==batch and c1['max_new_tokens']==512 and c1['prompt_len']==64 and c1['positions']==batch*512,'C1 report/workload mismatch')
    need(c1['required_ratio']==.99 and c1['allowed_maximum']==3.0 and math.isfinite(c1['gap_le_0_5_ratio']) and c1['gap_le_0_5_ratio']>=.99
         and math.isfinite(c1['max_gap']) and c1['max_gap']<=3.0,'C1 thresholds failed/changed')
    return {name:record(folder/name) for name in ('c2/mode_check.json','c1/hf_check.json','c2/tokens_L1_separate.json','c2/tokens_L1_loop.json')}


def _selection(ready,pairs,artifacts,watchdogs):
    need(len(pairs)==4 and all(len(x)==2 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in x) for x in pairs),'four pairs of actual process TPOT required')
    means=[x[0]/2+x[1]/2 for x in pairs];fastest=min(range(4),key=lambda i:(means[i],i));chosen=fastest if means[fastest]<=means[0]*.99 else 0
    rows=[]
    for i in range(4):
        p=ready['plans'][i];pred=p['original']['predicted_tpot_ns'];effective=p['final']['effective_prediction']['effective_predicted_tpot_ns']
        rows.append(dict(index=i,ranks=p['original']['ranks'],original_predicted_tpot_ns=pred,effective_predicted_tpot_ns=effective,
            process_tpot_ns=pairs[i],measured_tpot_ns=means[i],pred_over_meas=pred/means[i],effective_pred_over_meas=effective/means[i],artifact=artifacts[i]['binary']))
    return dict(kind='mx5_resolved_top4_measured_selection',schema_version=2,status='measured_selection',selected_index=chosen,
        selection_anchor=0,policy='fastest improves original candidate0 by at least1percent; otherwise retain0',candidates=rows,
        selected_artifact=artifacts[chosen]['binary'],selected_watchdog=watchdogs[chosen]['binary'],selected_pin=ready['plans'][chosen]['final']['pin'],
        final=False,default_enabled=False,target_defaults_changed=False)


def _bound_inputs(manifest):
    for key in ('protocol','domain_tool','model_config','prompts','policy','runtime_target','prefill','prefill_watchdog','guard_wrapper','check_wrapper','guard_policy','runner'):
        check_file(manifest[key])
    for key,path in (('guard_wrapper',ROOT/'scripts/mx_gpu_job.sh'),('check_wrapper',ROOT/'scripts/mx_check_modes.sh'),('guard_policy',ROOT/'configs/bench/guard_policy_c550.json'),('runner',Path(__file__))):
        need(manifest[key]==record(path),'execution helper identity changed: '+key)


def verify_results(path,*,protocol,domain_tool):
    results=load(path);manifest=results['manifest'];need(results['kind']=='mx5_resolved_top4_results' and results['schema_version']==2,'wrong v2 results schema')
    need(manifest['protocol']==record(protocol) and manifest['domain_tool']==record(domain_tool),'results protocol/domain changed')
    _bound_inputs(manifest);ready=_ready(protocol,domain_tool);preview=load(check_file(ready['origin_preview']))
    need(manifest['gpu'] in (4,5,6,7) and manifest['batch']==preview['request']['batch'],'results card/batch differs')
    _check_process(manifest['native_revalidation'],manifest['native_revalidation_file']);_fresh(preview,manifest['fresh_preview'],manifest['native_revalidation'])
    need(_prefill(check_file(manifest['prefill']),0)['artifact_id']==manifest['prefill_identity'] and _prefill(check_file(manifest['prefill_watchdog']),1)['artifact_id']==manifest['prefill_watchdog_identity'],'prefill identity changed')
    need(len(manifest['candidates'])==4 and [x['index'] for x in manifest['candidates']]==list(range(4)),'results lost original four candidate IDs')
    artifacts=[];watchdogs=[]
    for index,entry in enumerate(manifest['candidates']):
        a=_artifact(ready,index,preview);need(a==entry['artifact'],'executed artifact identity changed');artifacts.append(a)
        _check_process(entry['build'],entry['build_file'])
        for ref in entry['build_attempt_files'].values():check_file(ref)
        need(entry['build']['returncode']==0 and entry['build']['argv']==_compile_command(preview,entry['pin_file'],entry['watchdog']['binary']['path']),'WD1 actual build command differs')
        wd,_=_watchdog(check_file(entry['watchdog']['binary']),a['pin'],preview,load(check_file(a['source_snapshot'])),entry['watchdog']['query'])
        need(wd==entry['watchdog'],'WD1 identity/query changed');watchdogs.append(wd)
        q=entry['query_process'];_check_process(q,entry['query_process_file']);_guard(q,manifest['gpu'],300)
        need(_guard_files(q)==entry['query_guard'],'WD1 query guard evidence changed')
        need(q['argv'][4:]==[sys.executable,'-m','tilemega.mx5_plan_occupancy','--inputs',str(check_file(entry['query_inputs'])),'--out',str(check_file(wd['query'])),'--query-only','--iterations','0'],'WD1 query process/inputs differ')
        check=entry['check_process'];_check_process(check,entry['check_process_file']);_guard(check,manifest['gpu'],1700)
        need(_guard_files(check)==entry['check_guard'],'C1/C2 guard evidence changed')
        expected=['bash',str(ROOT/'scripts/mx_check_modes.sh'),manifest['model'],str(check_file(manifest['prefill_watchdog'])),str(check_file(wd['binary'])),str(manifest['batch']),'512',entry['check_out'],str(check_file(manifest['prompts']))]
        need(check['argv'][4:]==expected,'C1/C2 did not execute the exact WD1 artifact')
        need(_correctness(Path(entry['check_out']),manifest['batch'],Path(manifest['model']))==entry['correctness'],'C1/C2 raw evidence changed')
    pairs=[[] for _ in range(4)];pids=set();expected_order=[(r,(j+r)%4) for r in range(2) for j in range(4)]
    need([(p['round'],p['candidate']) for p in manifest['measurements']]==expected_order,'measurements lost original four/two-fresh interleaved order')
    args=argparse.Namespace(root=ROOT,gpu=manifest['gpu'],model=Path(manifest['model']),prefill_so=check_file(manifest['prefill']),prompt_ids=check_file(manifest['prompts']),batch=manifest['batch'],prefill_mode=manifest['prefill_mode'],policy=check_file(manifest['policy']))
    for item in manifest['measurements']:
        index=item['candidate'];process=item['process'];_check_process(process,item['process_file']);folder,pid=_guard(process,args.gpu,900)
        need(_guard_files(process)==item['guard'],'measurement guard evidence changed')
        need(pid==item['measurement_pid'] and pid not in pids,'measurement processes must be fresh distinct guard children');pids.add(pid)
        output=Path(item['output']);need(process['argv']==common.measurement_command(args,check_file(artifacts[index]['binary']),output,folder),'measured artifact/protocol command changed')
        report=load(check_file(item['raw']));need(report.get('measurement_policy',{}).get('guard') is True and not report.get('synthetic_fixture'),'measurement guard disabled or synthetic report')
        value=common.validate_measurement(report,preview,artifacts[index]['identity_id'],manifest['prefill_identity'])
        need(value==item['tpot_ns'],'actual measured TPOT changed');pairs[index].append(value)
    selected=_selection(ready,pairs,artifacts,watchdogs)
    for name,value in (('selection.json',selected),('selected.pin.json',selected['selected_pin'])):
        output=Path(path).parent/name
        if output.exists():need(load(output)==value,'published selection/pin output changed')
    need(results['selection']==selected,'selection/1percent anchor/pin changed')
    return selected


def _publication_targets(folder,selection):
    targets={}
    for role,ref in (('decode',selection['selected_artifact']),('decode_w',selection['selected_watchdog'])):
        binary=check_file(ref)
        for suffix in ('','.identity.json','.cu','.plan.json','.ptxas.log','.build_command.txt'):
            need(Path(str(binary)+suffix).is_file(),'selected artifact sidecar missing: '+suffix)
        for source in [binary,*sorted(binary.parent.glob(binary.name+'.*'))]:
            targets[Path(folder)/role/('plan.so'+source.name[len(binary.name):])]=source
    return targets


def _publish(folder,selection):
    targets=_publication_targets(folder,selection)
    for path in targets:
        need(not path.exists() and not path.is_symlink(),'existing cell artifact is not overwritten: '+str(path))
    created=[];directories=[]
    try:
        for path,target in targets.items():
            for directory in reversed([path.parent,*path.parent.parents]):
                if not directory.exists():directory.mkdir();directories.append(directory)
            path.symlink_to(target);created.append(path)
    except BaseException:
        for path in reversed(created):path.unlink()
        for directory in reversed(directories):directory.rmdir()
        raise


def orchestrate(args):
    need(args.root==ROOT and args.gpu in (4,5,6,7),'runner root/card is not the explicit supported target')
    os.chdir(ROOT)
    ready=_ready(args.protocol,args.domain_tool);preview=load(check_file(ready['origin_preview']));request=preview['request']
    need(args.batch==request['batch'] and (request['capacity'],request['past_lo'],request['past_hi'])==(1088,64,1086),'standard decode workload required')
    from .serving.workload import validate_prompt_ids
    validate_prompt_ids(load(args.prompt_ids),args.batch,64)
    need(load(args.policy).get('guard') is True,'measurement policy guard must be enabled')
    p0=_prefill(args.prefill_so,0);p1=_prefill(args.prefill_watchdog_so,1)
    args.out.mkdir(parents=True,exist_ok=False)
    job_log=Path(os.environ.get('TILEMEGA_GPU_JOB_LOG') or str(ROOT/'runs/mx5/gpu_jobs.log')).resolve()
    manifest=dict(gpu_job_log=str(job_log),protocol=record(args.protocol),domain_tool=record(args.domain_tool),model=str(args.model),model_config=record(args.model/'config.json'),prompts=record(args.prompt_ids),policy=record(args.policy),runtime_target=record(request['runtime_target']),
        prefill=record(args.prefill_so),prefill_watchdog=record(args.prefill_watchdog_so),prefill_identity=p0['artifact_id'],prefill_watchdog_identity=p1['artifact_id'],
        guard_wrapper=record(ROOT/'scripts/mx_gpu_job.sh'),check_wrapper=record(ROOT/'scripts/mx_check_modes.sh'),guard_policy=record(ROOT/'configs/bench/guard_policy_c550.json'),runner=record(Path(__file__)),
        gpu=args.gpu,batch=args.batch,prefill_mode=args.prefill_mode,invocation='explicit_resolved_v2_cli',purpose='C1C2_then_two_fresh_processes_per_original_candidate',audit_execution_authorized=False,candidates=[],measurements=[])
    def snapshot(status,**extra):write(args.out/('state%03d.json'%len(list(args.out.glob('state*.json')))),dict(status=status,manifest=manifest,**extra))
    env=dict(os.environ,PYTHONPATH=str(ROOT/'python'),NEEDS_MIB=str(args.needs_mib),TILEMEGA_GPU_JOB_LOG=str(job_log))
    for k in ('TILEMEGA_MX_DIAG','TILEMEGA_TRACE'):env.pop(k,None)
    cpu=dict(env,CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='');snapshot('preparing')
    try:
        prefix=args.out/'fresh_native/preview';prefix.parent.mkdir()
        process=run_command(common.native_solve_command(preview,prefix),prefix.parent/'process',env=cpu)
        manifest.update(native_revalidation=process,native_revalidation_file=_process_file(prefix.parent/'process',process));snapshot('origin_solve_returned')
        need(process['returncode']==0,'native exact top4 revalidation failed')
        manifest['fresh_preview']=record(str(prefix)+'.mx5_candidates.json');_fresh(preview,manifest['fresh_preview'],process);snapshot('origin_revalidated')
        for index in range(4):
            _bound_inputs(manifest);ready=_ready(args.protocol,args.domain_tool);artifact=_artifact(ready,index,preview)
            directory=args.out/('candidate%d'%index);directory.mkdir();pin=directory/'effective.pin.json';write(pin,artifact['pin']);out=directory/'watchdog/plan.so';out.parent.mkdir()
            entry=dict(index=index,artifact=artifact,pin_file=record(pin));manifest['candidates'].append(entry)
            build=run_command(_compile_command(preview,entry['pin_file'],out),directory/'build',env=cpu);entry.update(build=build,build_file=_process_file(directory/'build',build),build_attempt_files={path.name:record(path) for path in sorted(out.parent.glob(out.name+'.*')) if path.is_file()});snapshot('watchdog_built')
            need(build['returncode']==0,'WD1 resource/build failure; no unapproved retry or artifact substitution')
            wd,spec=_watchdog(out,artifact['pin'],preview,load(check_file(artifact['source_snapshot'])))
            inputs=directory/'query_inputs.json';write(inputs,spec);query=directory/'watchdog_query.json';guard=directory/'query_guard'
            command=[str(ROOT/'scripts/mx_gpu_job.sh'),str(args.gpu),'300',str(guard),sys.executable,'-m','tilemega.mx5_plan_occupancy','--inputs',str(inputs),'--out',str(query),'--query-only','--iterations','0']
            entry['query_inputs']=record(inputs);entry['query_process']=run_command(command,directory/'query_process',timeout=330,env=env);_guard(entry['query_process'],args.gpu,300)
            entry.update(query_process_file=_process_file(directory/'query_process',entry['query_process']),query_guard=_guard_files(entry['query_process']))
            wd,_=_watchdog(out,artifact['pin'],preview,load(check_file(artifact['source_snapshot'])),record(query));entry['watchdog']=wd;snapshot('watchdog_qualified')
            _ready(args.protocol,args.domain_tool);check_out=directory/'correctness';guard=directory/'check_guard'
            command=[str(ROOT/'scripts/mx_gpu_job.sh'),str(args.gpu),'1700',str(guard),'bash',str(ROOT/'scripts/mx_check_modes.sh'),str(args.model),str(args.prefill_watchdog_so),str(out),str(args.batch),'512',str(check_out),str(args.prompt_ids)]
            entry['check_process']=run_command(command,directory/'check_process',timeout=1730,env=env);_guard(entry['check_process'],args.gpu,1700)
            entry.update(check_process_file=_process_file(directory/'check_process',entry['check_process']),check_guard=_guard_files(entry['check_process']))
            entry.update(check_out=str(check_out),correctness=_correctness(check_out,args.batch,args.model));snapshot('C1C2_passed')
        for round_number in range(2):
            for j in range(4):
                index=(j+round_number)%4;ready=_ready(args.protocol,args.domain_tool);a=_artifact(ready,index,preview);_bound_inputs(manifest)
                directory=args.out/('candidate%d/round%d'%(index,round_number));output=directory/'measure';guard=directory/'guard'
                process=run_command(common.measurement_command(args,check_file(a['binary']),output,guard),directory,timeout=930,env=env);_,pid=_guard(process,args.gpu,900)
                raw=record(output/'measurements.json');report=load(check_file(raw))
                need(report.get('measurement_policy',{}).get('guard') is True and not report.get('synthetic_fixture'),'unqualified/synthetic measurement')
                value=common.validate_measurement(report,preview,a['identity_id'],p0['artifact_id'])
                manifest['measurements'].append(dict(candidate=index,round=round_number,process=process,process_file=_process_file(directory,process),guard=_guard_files(process),measurement_pid=pid,output=str(output),raw=raw,tpot_ns=value));snapshot('measured')
        pairs=[[x['tpot_ns'] for x in manifest['measurements'] if x['candidate']==i] for i in range(4)]
        selection=_selection(ready,pairs,[x['artifact'] for x in manifest['candidates']],[x['watchdog'] for x in manifest['candidates']])
        result=dict(kind='mx5_resolved_top4_results',schema_version=2,manifest=manifest,selection=selection)
        path=args.out/'results.json';write(path,result);selected=verify_results(path,protocol=args.protocol,domain_tool=args.domain_tool)
        write(args.out/'selection.json',selected);write(args.out/'selected.pin.json',selected['selected_pin'])
        if args.cell_dir is not None:_publish(args.cell_dir,selected)
        snapshot('measured_selection_complete',selection=record(args.out/'selection.json'),selected_pin=record(args.out/'selected.pin.json'))
        return selected
    except BaseException as error:snapshot('blocked',reason=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-results',type=Path)
    for flag in ('protocol','domain-tool'):parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--cell-dir',type=Path)
    if any(x=='--verify-results' or x.startswith('--verify-results=') for x in sys.argv[1:]):
        args=parser.parse_args();selected=verify_results(args.verify_results,protocol=args.protocol,domain_tool=args.domain_tool)
        if args.cell_dir:
            for name,ref in (('decode',selected['selected_artifact']),('decode_w',selected['selected_watchdog'])):
                need((args.cell_dir/name/'plan.so').resolve()==check_file(ref),'published cell differs from measured selection')
            for path,target in _publication_targets(args.cell_dir,selected).items():
                need(path.is_symlink() and path.resolve()==target.resolve(),'published sidecar differs from measured artifact: '+str(path))
        return
    for flag in ('root','model','prefill-so','prefill-watchdog-so','prompt-ids','policy','out'):parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--batch',type=int,choices=(1,2,4,8,16),required=True);parser.add_argument('--gpu',type=int,choices=(7,6,5,4),required=True)
    parser.add_argument('--prefill-mode',choices=('L1','L2','auto'),default='L2');parser.add_argument('--needs-mib',type=int,default=20480)
    args=parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    orchestrate(args)


if __name__=='__main__':main()
