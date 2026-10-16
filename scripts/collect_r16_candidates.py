#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Collect explicit candidate plans' guarded final rounds without selecting defaults."""
from __future__ import annotations
import argparse
import ast
import csv
import collections
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from r16_measurement_validation import (NATIVE_LLM_SYSTEMS, completed_guard,
    file_sha, native_identity, validate_llm)
from summarize_r16 import dnn_device_identity, record_directory

TERMINAL={'done','failed','skipped','not_run','cancelled'}
DRIVERS={}
METRICS=('e2e_seconds','ttft_seconds','tpot_seconds','output_tokens_per_second')


def read(path):
    return json.loads(Path(path).read_text())


def load_driver(candidate):
    path=Path(candidate['script']);digest=file_sha(path)
    if digest!=candidate['script_sha256']:
        raise ValueError('frozen candidate driver changed')
    helper_path=path.with_name('r16_argument_contract.py')
    helper_digest=candidate.get('argument_contract_sha256')
    if helper_digest is not None and file_sha(helper_path)!=helper_digest:
        raise ValueError('frozen native argument helper changed')
    key=(str(path.resolve()),digest)
    if key not in DRIVERS:
        tree=ast.parse(path.read_text())
        # Only top-level imports execute when loading the frozen validation API.
        for node in tree.body:
            names=[]
            if isinstance(node,ast.Import):names=[x.name.split('.')[0] for x in node.names]
            elif isinstance(node,ast.ImportFrom):names=[(node.module or '').split('.')[0]]
            allowed=('__future__','r16_measurement_validation')+(
                ('r16_argument_contract',) if helper_digest is not None else ())
            if any(n not in sys.stdlib_module_names and n not in allowed for n in names):
                raise ValueError('unsupported frozen driver: non-stdlib top-level import')
        spec=importlib.util.spec_from_file_location('candidate_collector_'+digest,path)
        module=importlib.util.module_from_spec(spec)
        previous=sys.modules.get('r16_argument_contract')
        if helper_digest is not None:
            helper_spec=importlib.util.spec_from_file_location('r16_argument_contract',helper_path)
            helper=importlib.util.module_from_spec(helper_spec)
            helper_spec.loader.exec_module(helper)
            sys.modules['r16_argument_contract']=helper
        try:spec.loader.exec_module(module)
        finally:
            if previous is not None:sys.modules['r16_argument_contract']=previous
            else:sys.modules.pop('r16_argument_contract',None)
        if not callable(getattr(module,'measurement',None)) or not callable(getattr(module,'check_pair',None)):
            raise ValueError('unsupported frozen driver: measurement/check_pair API absent')
        DRIVERS[key]=module
    return DRIVERS[key]


def protocol(plan):
    p=plan['protocol']
    if any(p.get(k)!=v for k,v in [('isl',64),('osl',1024),('warmup',1),('repeats',3)]):
        raise ValueError('unsupported plan: final 64/1024 warmup1/repeats3 required')
    if not any(p.get(k)==3 for k in ('final_fresh_rounds','fresh_timing_rounds','timing_fresh_processes')):
        raise ValueError('unsupported plan: three fresh final processes required')


def scheduled_jobs(plan_path,plan):
    """Only deterministic RF2 finals or explicit plan-local final commands qualify."""
    protocol(plan)
    requests=[]
    if 'run_id' in plan:
        for cell in plan['cells']:
            if len(cell['candidates'])>8:raise ValueError('RF2 plan exceeds eight candidates')
            for c in cell['candidates']:
                requests.append((c,[(f"rf2_{plan['run_id']}_{cell['model']}_B{cell['batch']}_final{rnd}_{c['id']}",rnd,None) for rnd in range(3)]))
        return requests
    explicit=[j for cell in plan['cells'] for j in cell.get('gpu_jobs',[])]
    if not explicit:
        # H3 records its exact launch commands in this known companion, rather
        # than in the plan. No scheduler/report directory search is performed.
        gpus={cell.get('gpu') for cell in plan['cells']}
        gpu=plan.get('gpu',next(iter(gpus)) if len(gpus)==1 else None)
        if gpu is None:raise ValueError('unsupported plan: ambiguous companion queue GPU')
        companion=plan_path.parent/f"queue_gpu{gpu}_preview.json"
        if not companion.is_file():raise ValueError('unsupported plan: no explicit final job commands')
        explicit=read(companion)
    bases=[c for cell in plan['cells'] for c in cell['candidates']]
    groups={}
    for job in explicit:
        argv=job.get('command',[])
        if '--stage' not in argv or argv[argv.index('--stage')+1]!='final':continue
        if not job.get('gpu') or not job.get('timing'):raise ValueError('explicit final job is not guarded timing')
        path=Path(argv[argv.index('--candidate')+1]);c=read(path)
        matched=[b for b in bases if (b['id'],b['model'],b['batch'],b['prefill'],b['decode'],b['snapshot_head'])==(c['id'],c['model'],c['batch'],c['prefill'],c['decode'],c['snapshot_head'])]
        if len(matched)!=1:raise ValueError('explicit final command belongs to another plan candidate')
        for key in ('seed','config_sha256','script_sha256','compiler_sha256','system','arm','variant',
                    'native_system_label','argument_contract_sha256','phase_sources'):
            if matched[0].get(key)!=c.get(key):raise ValueError('explicit final candidate changed plan coordinate '+key)
        if Path(c['file']).resolve()!=path.resolve():raise ValueError('candidate file identity differs from explicit command')
        rnd=int(argv[argv.index('--round-index')+1])
        if rnd not in (0,1,2):raise ValueError('final process round is outside 0/1/2')
        key=(c['model'],c['batch'],c['id'],c['mode'],c['loop'])
        group=groups.setdefault(key,(c,[]));group[1].append((job['name'],rnd,job))
    for c,jobs in groups.values():
        if len(jobs)!=3 or {rnd for _,rnd,_ in jobs}!={0,1,2}:
            raise ValueError('explicit final candidate must declare exactly rounds 0/1/2')
        requests.append((c,sorted(jobs,key=lambda x:x[1])))
    if not requests:raise ValueError('unsupported plan: no final commands')
    return requests


def read_states(root):
    state={};cards={}
    for path in sorted((root/'runs/r16/scheduler').glob('gpu*/state.json')):
        for name,row in read(path).items():
            if name in state:raise ValueError('ambiguous scheduler job name: '+name)
            state[name]=row;cards[name]=int(path.parent.name.removeprefix('gpu'))
    return state,cards


def check_native_alias(candidate,value,directory,expected,actual):
    label=candidate.get('native_system_label',candidate['system'])
    if label not in NATIVE_LLM_SYSTEMS or value['system']!=label:
        raise ValueError('native system label differs from explicit candidate')
    if 'native_system_label' not in candidate:return
    evidence=read(directory/'measurements/candidate-evidence.json')
    for key,wanted in (('system',candidate['system']),('native_system_label',label),
                       ('variant',candidate['variant']),
                       ('artifact_ids',{p:i['artifact_id'] for p,i in expected.items()})):
        if evidence.get(key)!=wanted:
            raise ValueError('candidate alias evidence differs in '+key)
    # The same guarded invocation writes both receipts; the execution receipt
    # binds the outer candidate evidence to the immutable driver's bytes.
    if actual['driver_sha256']!=candidate['script_sha256'] or (
            'driver_sha256' in evidence and evidence['driver_sha256']!=actual['driver_sha256']):
        raise ValueError('candidate alias evidence belongs to another driver')
    contract=read(directory/'measurements/native-argument-contract.json')
    runner=Path(candidate['snapshot_source'])/'python/tilemega/serving/r16_run.py'
    if (Path(contract['runner']).resolve()!=runner.resolve() or
            contract['runner_sha256']!=file_sha(runner) or
            value['configuration']['runner_sha256']!=contract['runner_sha256'] or
            contract.get('outer_system')!=candidate['system'] or
            contract.get('variant')!=candidate['variant']):
        raise ValueError('native alias parser/source evidence differs from candidate')
    parsed=contract['parsed']
    for key,wanted in (('system_label',label),('mode',candidate['mode']),
                       ('decode_loop',candidate['loop']),('max_tokens',1024),
                       ('round_index',value['round_index'])):
        if parsed.get(key)!=wanted:
            raise ValueError('native alias parsed arguments differ in '+key)
    if (int(parsed['batch'])!=candidate['batch'] or
            any(Path(parsed[p]).resolve()!=Path(candidate[p]).resolve() for p in ('prefill','decode')) or
            Path(parsed['out']).resolve()!=(directory/'measurements').resolve()):
        raise ValueError('native alias parsed workload belongs to another candidate')


def validate_phase_sources(candidate, identities):
    declared = candidate.get('phase_sources')
    if declared is None:
        if any(identity['source']['head'] != candidate['snapshot_head']
               for identity in identities.values()):
            raise ValueError('native source differs from declared frozen candidate')
        return
    if set(declared) != {'prefill', 'decode'}:
        raise ValueError('phase sources must bind both prefill and decode')
    for phase, identity in identities.items():
        wanted = declared[phase]
        if set(wanted) != {'head', 'root', 'compiler_sha256'}:
            raise ValueError('phase source requires head, root and compiler SHA')
        actual = identity['source']
        if any(actual.get(key) != value for key, value in wanted.items()):
            raise ValueError('native phase source differs from declared '+phase)
    decode = declared['decode']
    if (decode['head'] != candidate['snapshot_head'] or
            decode['root'] != candidate['snapshot_source'] or
            decode['compiler_sha256'] != candidate['compiler_sha256']):
        raise ValueError('decode phase source differs from candidate compiler')


def inspect_measurement(root,state,cards,name,rnd,candidate):
    record=state[name]
    if record.get('status')!='done' or record.get('exit_code')!=0:
        raise ValueError('final job did not complete successfully: '+str(record.get('status')))
    directory=record_directory(root,name,record)
    guard=completed_guard(directory,record)
    path=directory/'measurements'/f"B{candidate['batch']}"/'measurements.json'
    value,_,own=validate_llm(path,guard,root)
    physical=dnn_device_identity(value,directory,path)
    if value['round_index']!=rnd or value['batch']!=candidate['batch']:
        raise ValueError('system/batch/final round differs from explicit candidate')
    if value['gpu_index']!=candidate['gpu'] or cards[name]!=candidate['gpu']:
        raise ValueError('physical GPU differs from plan or scheduler')
    if value['model_config_sha256']!=candidate['config_sha256'] or value['prompt_ids_sha256']!=file_sha(candidate['prompt_ids']):
        raise ValueError('config or frozen prompt differs from explicit candidate')
    if value['_r16_native']['canonical_model']!=candidate['model']:
        raise ValueError('canonical model differs from plan label')
    config=value['configuration']
    if config['mode']!=candidate['mode'] or config['decode_loop']!=candidate['loop']:
        raise ValueError('executor/device loop differs from explicit candidate')
    checkpoint=read(config['checkpoint_manifest'])
    if checkpoint['seed']!=candidate['seed']:raise ValueError('shared checkpoint seed differs')
    actual=read(directory/'measurements/actual-execution.json')
    if actual['driver_sha256']!=candidate['script_sha256'] or len(actual['calls'])!=8:
        raise ValueError('actual executor receipt belongs to another driver or request count')
    for count in (1,1024):
        calls=[c for c in actual['calls'] if c['tokens']==count]
        if len(calls)!=4 or any(c['executor']!=candidate['mode'] or c['device_loop_used']!=bool(candidate['loop'] and count>1) for c in calls):
            raise ValueError('actual executor/device loop silently differs')
    if type(value['pid']) is not int or value['pid']<=0:raise ValueError('measured process PID is absent')
    expected={p:native_identity(candidate[p],config['identities'][p]) for p in ('prefill','decode')}
    validate_phase_sources(candidate,expected)
    check_native_alias(candidate,value,directory,expected,actual)
    driver=load_driver(candidate)
    pair=driver.check_pair(candidate)
    if {p:i['artifact_id'] for p,i in pair.items()}!={p:i['artifact_id'] for p,i in expected.items()}:
        raise ValueError('frozen driver certified another binary pair')
    normalized=dict(state);normalized[name]=dict(record,out=str(directory))
    measured=driver.measurement(root,normalized,name,candidate)
    if measured['pid']!=value['pid'] or any(measured[k]!=value[k] for k in ('e2e_seconds','ttft_seconds','tpot_seconds')):
        raise ValueError('frozen driver measurement differs from independently validated receipt')
    return dict(job=name,round=rnd,pid=value['pid'],gpu=value['gpu_index'],measurement=str(path),
        **physical,
        system=candidate['system'],native_system_label=candidate.get('native_system_label',candidate['system']),
        variant=candidate.get('variant',{}),
        **{k:value[k] for k in METRICS},source_checkpoint_sha256=value['_r16_native']['source_checkpoint_sha256'],
        native_artifact_ids={p:i['artifact_id'] for p,i in expected.items()},
        binary_sha256={p:i['binary_sha256'] for p,i in expected.items()},versions=value['versions'],
        acceptance=own['acceptance'],ownership_verified=own['ownership_verified'],
        unresolved_host_pids=own['unresolved_host_pids'],source_head=candidate['snapshot_head'],
        phase_sources={p:i['source'] for p,i in expected.items()},
        actual_coordinates={p:dict(implementations=i.get('implementations',{}),
            gemms=i['plan'].get('gemms',[]),
            plan={k:i['plan'].get(k) for k in ('pg','page_bytes','residency','cluster_dim','combine','attention_page','activation_tma','qkv_epilogue','forward_gemm_impl')},
            gemm_implementations={impl:sum(g.get('impl','mma16')==impl for g in i['plan'].get('gemms',[])) for impl in ('mma16','gemv','wgmma')}) for p,i in expected.items()})


def summarize_samples(samples):
    if not samples:return dict(status='not_finalized',fresh_processes=0,protocol_complete=False)
    rounds={s['round'] for s in samples};pids={s['pid'] for s in samples}
    if len(rounds)!=len(samples) or len(pids)!=len(samples):raise ValueError('duplicate final round or measured process PID')
    for key in ('source_checkpoint_sha256','native_artifact_ids','binary_sha256','versions','gpu','source_head','actual_coordinates',
                'system','native_system_label','variant',
                'physical_gpu_uuid','physical_gpu_identity_status','hardware_boot_id'):
        if len({json.dumps(s[key],sort_keys=True) for s in samples})!=1:
            raise ValueError('final rounds disagree on '+key)
    return dict(status='completed' if rounds=={0,1,2} else 'partial',fresh_processes=len(samples),protocol_complete=rounds=={0,1,2},
        **{k:statistics.median(s[k] for s in samples) for k in METRICS},
        source_checkpoint_sha256=samples[0]['source_checkpoint_sha256'],native_artifact_ids=samples[0]['native_artifact_ids'],binary_sha256=samples[0]['binary_sha256'],versions=samples[0]['versions'],actual_coordinates=samples[0]['actual_coordinates'],
        **{k:samples[0][k] for k in ('physical_gpu_uuid','physical_gpu_identity_status','hardware_boot_id')},
        acceptance='strict' if all(s['acceptance']=='strict' for s in samples) else 'provisional_container_pid_namespace',ownership_verified=all(s['ownership_verified'] for s in samples),
        unresolved_host_pids=sorted({p for s in samples for p in s['unresolved_host_pids']}))


def selection_matches(choice,candidate,samples):
    try:
        proofs=[x for x in choice.get('candidates',[]) if x.get('candidate')==candidate['id']]
        if len(proofs)!=1:return False
        recorded=proofs[0].get('samples',[])
        return len(recorded)==3 and {(str(Path(x['measurement']).resolve()),x['pid']) for x in recorded}=={(str(Path(x['measurement']).resolve()),x['pid']) for x in samples}
    except (OSError,ValueError,KeyError,TypeError):return False


def collect(root,plans):
    root=Path(root).resolve();state,cards=read_states(root);rows=[];processes=[];rejections=[];selected=[];seen=set()
    for path in plans:
        path=Path(path).resolve()
        try:
            plan=read(path)
            if 'root' in plan and Path(plan['root']).resolve()!=root:raise ValueError('plan belongs to a different root')
            requests=scheduled_jobs(path,plan)
        except (OSError,ValueError,KeyError,TypeError) as error:
            rejections.append(dict(plan=str(path),status='unsupported',reason=str(error)));continue
        report_path=path.parent/'selection-report.json'
        try:
            report=read(report_path) if report_path.is_file() else {}
            choices={(c['model'],c['batch']):c for c in report.get('cells',[])}
        except (OSError,ValueError,KeyError,TypeError) as error:
            choices={};rejections.append(dict(plan=str(path),status='unsupported_selection_report',reason=str(error)))
        for c,jobs in requests:
            base=dict(plan=str(path),plan_sha256=file_sha(path),model_config_sha256=c['config_sha256'],model=c['model'],batch=c['batch'],candidate=c['id'],system=c['system'],native_system_label=c.get('native_system_label',c['system']),executor=c['mode'],device_loop=c['loop'],gpu=c['gpu'],source_head=c['snapshot_head'],driver_sha256=c['script_sha256'],selected_by_report=False,default_promoted=False)
            samples=[];statuses={};unsupported=c.get('native_system_label',c.get('system')) not in NATIVE_LLM_SYSTEMS
            if unsupported:rejections.append(dict(plan=str(path),candidate=c['id'],status='unsupported',reason='current native validation does not recognize system '+str(c.get('system'))))
            for name,rnd,definition in jobs:
                statuses[name]=state.get(name,{}).get('status','not_enqueued')
                if unsupported or name not in state or state[name].get('status') not in TERMINAL:continue
                if name in seen:
                    rejections.append(dict(plan=str(path),job=name,status='duplicate_input',reason='same final job covered by multiple explicit plans'));continue
                seen.add(name)
                try:
                    sample=inspect_measurement(root,state,cards,name,rnd,c);samples.append(sample)
                    processes.append(dict(plan=str(path),candidate=c['id'],model=c['model'],batch=c['batch'],executor=c['mode'],device_loop=c['loop'],**sample))
                except (OSError,ValueError,KeyError,TypeError,ImportError) as error:
                    rejections.append(dict(plan=str(path),candidate=c['id'],job=name,status=state[name]['status'],reason=str(error)))
            try:summary=summarize_samples(samples)
            except ValueError as error:
                rejections.append(dict(plan=str(path),candidate=c['id'],status='rejected_triplet',reason=str(error)));summary=dict(status='rejected_triplet',fresh_processes=len(samples),protocol_complete=False)
            if unsupported:summary['status']='unsupported'
            elif not samples and any(s in TERMINAL for s in statuses.values()):summary['status']='rejected'
            choice=choices.get((c['model'],c['batch']),{})
            if choice.get('status')=='complete' and choice.get('selected')==c['id']:
                if summary['protocol_complete'] and selection_matches(choice,c,samples):
                    base['selected_by_report']=True;selected.append(dict(**base,selection_report=str(report_path),selection_report_sha256=file_sha(report_path)))
                else:rejections.append(dict(plan=str(path),candidate=c['id'],status='unvalidated_selection',reason='selection-report lacks the same three independently valid final measurement paths/PIDs'))
            rows.append(dict(**base,**summary,job_statuses=statuses))
    return dict(evidence='verified_independent_receipt_analysis',scope='explicit plans only; final N1024 rounds; no pilot replacement or default promotion',plans=[str(Path(p).resolve()) for p in plans],candidates=rows,processes=processes,selections=selected,rejections=rejections)


def compact_coordinates(coordinates):
    result={}
    for phase,row in coordinates.items():
        implementations={k:v for k,v in row['implementations'].items() if k!='gemms'}
        geometries=collections.Counter(tuple((key,g.get(key)) for key in ('impl','tile_m','tile_n','tile_k','stages','split_k')) for g in row.get('gemms',[]))
        result[phase]=dict(implementations=implementations,plan=row['plan'],gemm_implementations=row['gemm_implementations'],geometries=[dict(key,count=count) for key,count in sorted(geometries.items(),key=lambda x:str(x[0]))])
    return result


def csv_rows(rows):
    result=[]
    for r in rows:
        result.append(dict(model=r['model'],batch=r['batch'],candidate=r['candidate'],system=r['system'],native_system_label=r['native_system_label'],status=r['status'],fresh_processes=r['fresh_processes'],protocol_complete=r['protocol_complete'],selected_by_report=r['selected_by_report'],default_promoted=False,executor=r['executor'],device_loop=r['device_loop'],gpu=r['gpu'],
            e2e_ms=r.get('e2e_seconds','')*1000 if 'e2e_seconds' in r else '',ttft_ms=r.get('ttft_seconds','')*1000 if 'ttft_seconds' in r else '',tpot_ms=r.get('tpot_seconds','')*1000 if 'tpot_seconds' in r else '',tokens_per_second=r.get('output_tokens_per_second',''),
            acceptance=r.get('acceptance',''),ownership_verified=r.get('ownership_verified',''),physical_gpu_uuid=r.get('physical_gpu_uuid',''),physical_gpu_identity_status=r.get('physical_gpu_identity_status','unknown'),hardware_boot_id=r.get('hardware_boot_id',''),unresolved_host_pids=json.dumps(r.get('unresolved_host_pids',[])),source_head=r['source_head'],model_config_sha256=r['model_config_sha256'],native_artifact_ids=json.dumps(r.get('native_artifact_ids',{}),sort_keys=True),binary_sha256=json.dumps(r.get('binary_sha256',{}),sort_keys=True),source_checkpoint_sha256=r.get('source_checkpoint_sha256',''),coordinates=json.dumps(compact_coordinates(r.get('actual_coordinates',{})),sort_keys=True,separators=(',',':')),versions=json.dumps(r.get('versions',{}),sort_keys=True),plan=r['plan'],plan_sha256=r['plan_sha256'],driver_sha256=r['driver_sha256']))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--plan',type=Path,action='append',required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result=collect(args.root,args.plan);args.out.mkdir(parents=True,exist_ok=True);(args.out/'candidates.json').write_text(json.dumps(result,indent=2)+'\n');(args.out/'rejections.json').write_text(json.dumps(result['rejections'],indent=2)+'\n');(args.out/'selected-by-report.json').write_text(json.dumps(result['selections'],indent=2)+'\n')
    rows=csv_rows(result['candidates'])
    with (args.out/'candidates.csv').open('w',newline='') as stream:
        if rows:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(candidates=len(rows),completed=sum(x['protocol_complete'] for x in rows),partial=sum(x['status']=='partial' for x in rows),validated_processes=len(result['processes']),rejections=len(result['rejections']),selected_by_report=len(result['selections']),default_promoted=False)))

if __name__=='__main__':main()
