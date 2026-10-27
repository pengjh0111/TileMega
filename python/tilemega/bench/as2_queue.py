"""Single submitter for finite AS2 jobs; idle observation never reserves a device."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

from tilemega.bench.as1_run import devices_from_snapshot, process_identity, snapshot
from tilemega.bench.as2_run import unfinished_owned_jobs

ROOT=Path('/home/d50064638/junhuip')
REPO=ROOT/'TileMega'
RUNS=REPO/'runs/as2'
WORK=ROOT/'tilemega-as2'
CONFIG=REPO/'configs/ascend/as2_focus_compare.json'


def write(path, value):
    temporary=path.with_suffix(path.suffix+'.new')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def controlled(raw):
    path=Path(raw).absolute()
    if path.resolve()!=path or not path.is_relative_to(RUNS) or path==RUNS:
        raise ValueError('Require a canonical path beneath runs/as2')
    return path


def idle_devices(state, required_free_mib, maximum_utilization=0):
    return {i:d for i,d in devices_from_snapshot(state).items()
            if d.get('health')=='OK' and d.get('utilization_percent') is not None
            and d['utilization_percent']<=maximum_utilization and d['free_mib']>=required_free_mib}


def validate_job(job):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',job['id']):raise ValueError('Invalid job identity')
    if job['environment'] not in ('as2','vllm-ascend','sglang-npu'):raise ValueError('Unknown private environment')
    if job.get('physical_device','auto') not in ('auto',0,1):raise ValueError('Unknown physical device')
    if not 0<job['timeout_s']<=900 or job['required_free_mib']<0:raise ValueError('Require finite bounded resources')
    if not isinstance(job['argv'],list) or not job['argv'] or not all(isinstance(x,str) for x in job['argv']):
        raise ValueError('Command must be an argv array')
    controlled(job['output'])
    return job


def initialize(directory):
    directory=controlled(directory);directory.mkdir(parents=True,exist_ok=False)
    (directory/'requests').mkdir();(directory/'observations').mkdir()
    policy=json.loads(CONFIG.read_text());write(directory/'policy.json',policy)
    write(directory/'state.json',dict(schema='tilemega.as2.serial_queue.v1',status='prepared',jobs={},
          device_locked=False,baseline_numerical_comparison=False,
          source_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest()))
    for model_index,model in enumerate(policy['models']):
        manifest=json.loads((WORK/'weights'/model/'manifest.json').read_text())
        cfg=manifest['config']
        for batch_index,batch in enumerate(policy['batches']):
            kv=2*cfg['num_hidden_layers']*batch*cfg['num_key_value_heads']*1152*cfg['head_dim']*2
            required=(manifest['unique_weight_bytes']+kv+(6<<30)+(1<<20)-1)//(1<<20)
            for system_index,system in enumerate(policy['systems']):
                name=f'{model}_b{batch}_{system}'
                out=directory/'results'/name
                common=['--model',model,'--batch',str(batch),'--outputs',str(policy['osl']),
                        '--warmup',str(policy['warmup']),'--repetitions',str(policy['repetitions']),'--out',str(out)]
                if system.startswith('torch_npu'):
                    environment='as2';command=['python','-m','tilemega.bench.as2_eager',*common,
                        '--attention',policy['torch_attention'],'--norm','fused','--cache','static',
                        '--execution','compile' if system.endswith('compile') else 'eager',
                        '--compile-backend',policy['torch_compile_backend'],'--performance-only']
                else:
                    environment='vllm-ascend' if system=='vllm_ascend' else 'sglang-npu'
                    module='as2_vllm' if system=='vllm_ascend' else 'as2_sglang'
                    graph=policy['engine_graph'] if system=='vllm_ascend' else policy.get('sglang_graph','off')
                    command=['python','-m','tilemega.bench.'+module,*common,'--graph',graph]
                    if system=='sglang_npu':
                        command+=['--stream-output',policy.get('sglang_stream_output','delta')]
                        kv=4*cfg['num_hidden_layers']*batch*cfg['num_key_value_heads']*2048*cfg['head_dim']
                        required=(manifest['unique_weight_bytes']+kv+(6<<30)+(1<<20)-1)//(1<<20)
                job=validate_job(dict(id=name,kind='baseline',model_id=model,batch=batch,system=system,
                    environment=environment,argv=command,output=str(out),physical_device='auto',
                    required_free_mib=required,timeout_s=policy['timeout_s'],
                    priority=100+20*model_index+4*batch_index+system_index))
                write(directory/'requests'/f'{name}.json',job)
    (directory/'results').mkdir()
    return directory


def collect_samples(job):
    out=Path(job['output']);samples=[]
    if (out/'samples.jsonl').exists():
        samples=[json.loads(line) for line in (out/'samples.jsonl').read_text().splitlines() if line]
    else:
        samples=[json.loads(p.read_text()) for p in sorted(out.glob('request_*/sample.json'))]
    timed=[s for s in samples if s.get('repetition',-1)>=0 and not s.get('warmup',False)]
    if len(timed)!=3:raise ValueError('Require three complete timed requests')
    batch=job['batch']
    for s in timed:
        if s.get('output_count',1024)!=1024 or s.get('output_tokens',batch*1024)!=batch*1024:
            raise ValueError('Incomplete generation length')
        if s.get('decode_forwards',1023)!=1023:raise ValueError('Incorrect decode count')
        for key in ('ttft_ms','e2e_ms','tpot_ms'):
            if not isinstance(s.get(key),(int,float)) or not math.isfinite(s[key]) or s[key]<=0:
                raise ValueError('Missing/invalid request timing')
        if not math.isclose(s['e2e_ms']-s['ttft_ms'],s['tpot_ms']*1023,rel_tol=1e-6,abs_tol=.001):
            raise ValueError('Request timing boundaries disagree')
    return dict(samples=timed,**{k:statistics.median(s[k] for s in timed) for k in ('ttft_ms','e2e_ms','tpot_ms')},
                throughput_tokens_per_s=statistics.median(batch*1024*1000/s['e2e_ms'] for s in timed))


def ready(job,state,policy):
    if not job.get('wait_for_baselines'):return True
    cells={v.get('request',{}).get('cell_id',k) for k,v in state['jobs'].items() if v['status']=='collected'}
    return all(f'{job["model_id"]}_b{job["batch"]}_{system}' in cells for system in policy['systems'])


def summarize(directory,state):
    policy=json.loads((directory/'policy.json').read_text());rows=[]
    for model in policy['models']:
        for batch in policy['batches']:
            row=dict(model_id=model,batch=batch)
            for system in [*policy['systems'],'tilemega']:
                cell=f'{model}_b{batch}_{system}'
                attempts=[v for k,v in state['jobs'].items() if v.get('request',{}).get('cell_id',k)==cell]
                item=next((v for v in reversed(attempts) if v['status']=='collected'),attempts[-1] if attempts else {})
                request=directory/'requests'/(cell+'.json')
                if not item and request.exists():
                    pending=json.loads(request.read_text())
                    if not pending.get('enabled',True):item=dict(status='blocked-setup',reason=pending.get('deferred_reason'))
                row[system+'_status']=item.get('status','not_run')
                for metric in ('tpot_ms','ttft_ms','e2e_ms','throughput_tokens_per_s'):
                    row[system+'_'+metric]=item.get('metrics',{}).get(metric)
                row[system+'_physical_device']=item.get('physical_device_id')
                row[system+'_interference']=item.get('receipt',{}).get('timing_status')
            for system in policy['systems']:
                baseline=row[system+'_tpot_ms'];native=row['tilemega_tpot_ms']
                row['speedup_vs_'+system]=baseline/native if baseline and native else None
            rows.append(row)
    write(directory/'comparison.json',dict(status='exploratory_not_formal_G_PERF',rows=rows,
        baseline_numerical_comparison=False,all_attempts_retained=True))
    with (directory/'comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    systems=[*policy['systems'],'tilemega']
    lines=['Exploratory full64/1024 request TPOT (ms), median of three timed requests.',
           'Missing cells are not measured. No paired confidence interval or AS2 G-PERF claim.',
           'Resident contexts and interference remain in each job receipt. Baseline numerical comparisons are waived.','',
           '| Model | B | '+' | '.join(systems)+' |',
           '|---|---|'+'---|'*len(systems)]
    for row in rows:
        cells=[f'{row[s+"_tpot_ms"]:.3f}' if row[s+'_tpot_ms'] is not None else row[s+'_status'] for s in systems]
        lines.append(f'| {row["model_id"]} | {row["batch"]} | '+' | '.join(cells)+' |')
    (directory/'comparison.md').write_text('\n'.join(lines)+'\n')


def serious_failure(job_dir,receipt):
    if receipt.get('status') in ('timeout','interrupted'):return 'Own job exceeded a finite bound; device work may remain'
    for name in ('stdout.log','stderr.log'):
        p=job_dir/name
        if p.exists():
            with p.open('rb') as stream:
                stream.seek(max(0,p.stat().st_size-(1<<20)));tail=stream.read().decode(errors='replace')
            if re.search(r'507033|TsdOpen|CreateSession.*fail',tail,re.I):return 'Device session initialization failed'
    after=job_dir/'after.json'
    if after.exists():
        devices=devices_from_snapshot(json.loads(after.read_text()))
        selected=devices.get(receipt.get('physical_device_id'),{})
        if selected and selected.get('health')!='OK':return 'Selected device health changed'
    return None


def run(directory,hours):
    directory=controlled(directory);state=json.loads((directory/'state.json').read_text())
    if state.get('status') in ('running','paused-device-safety'):
        if state.get('status')=='paused-device-safety':raise ValueError('Review retained device failure before a new queue')
        old=state.get('supervisor_identity')
        if old:
            try:
                if process_identity(old['pid'])==old:raise ValueError('The existing project submitter is still running')
            except (FileNotFoundError,ProcessLookupError):pass
    # An interrupted submission is never retried blindly or overwritten.
    if any(x.get('status')=='running' for x in state['jobs'].values()):
        raise ValueError('Reconcile the interrupted own job receipt before resuming')
    state.update(status='running',supervisor_identity=process_identity(os.getpid()),
                 started_at=dt.datetime.now(dt.timezone.utc).isoformat(),finite_monitor_hours=hours)
    write(directory/'state.json',state)
    stop_requested=[]
    # A signal stops new submissions after the current runner has reaped its
    # child. Killing a waiting subprocess.run would orphan its timeout owner.
    def stop_after_current(signum,frame):stop_requested.append(signum)
    signal.signal(signal.SIGINT,stop_after_current);signal.signal(signal.SIGTERM,stop_after_current)
    deadline=time.monotonic()+hours*3600;previous={};policy=json.loads((directory/'policy.json').read_text())
    while time.monotonic()<deadline:
        if stop_requested or (directory/'stop_after_current.json').exists():
            state['status']='stopped_after_current';break
        jobs=[validate_job(json.loads(p.read_text())) for p in (directory/'requests').glob('*.json')]
        remaining=[j for j in jobs if j['id'] not in state['jobs'] and j.get('enabled',True)]
        pending=sorted((j for j in remaining if ready(j,state,policy)),
                       key=lambda j:(j.get('priority',100),j['id']))
        if not pending:
            if remaining:
                state.update(status='blocked-prerequisite',waiting_reason='Uncollected baseline cells prevent the corresponding TileMega row');break
            state['status']='completed_with_deferred' if any(not j.get('enabled',True) for j in jobs) else 'completed'
            break
        if unfinished_owned_jobs():
            state['waiting_reason']='Existing own submission has not finished';write(directory/'state.json',state)
            time.sleep(policy['idle_interval_s']);continue
        observed=snapshot(15);stamp=str(time.time_ns())
        write(directory/'observations'/f'{stamp}.json',observed)
        selected=None
        for job in pending:
            functional=job['kind']=='functional' and not job.get('require_idle',False)
            devices=idle_devices(observed,job['required_free_mib'],100 if functional else policy['maximum_idle_utilization_percent'])
            if job.get('physical_device','auto')!='auto':devices={i:d for i,d in devices.items() if i==job['physical_device']}
            eligible={i:d for i,d in devices.items() if functional or previous.get(i,0)>=policy['idle_observations']-1}
            if eligible:
                selected=(job,max(eligible,key=lambda i:eligible[i]['free_mib']));break
        available=idle_devices(observed,0,policy['maximum_idle_utilization_percent'])
        previous={i:previous.get(i,0)+1 for i in available}
        if selected is None:
            state['waiting_reason']='Waiting for repeated healthy idle observations and sufficient HBM; no device reserved'
            write(directory/'state.json',state);time.sleep(policy['idle_interval_s']);continue
        job,physical=selected
        disk=subprocess.run([sys.executable,str(REPO/'scripts/as2_disk_status.py')],capture_output=True,text=True)
        (directory/f'disk_before_{job["id"]}.json').write_text(disk.stdout)
        if disk.returncode:
            state.update(status='paused-resource',waiting_reason='Project disk preflight failed');break
        job_dir=RUNS/'jobs'/('focus_'+job['id']+'_'+str(time.time_ns()))
        envscript=REPO/'scripts/as2_env.sh' if job['environment']=='as2' else REPO/'scripts/as2_baseline_env.sh'
        shell='source "$1" "$2" || exit; shift 2; unset TORCH_DEVICE_BACKEND_AUTOLOAD; exec python -m tilemega.bench.as2_run "$@"'
        command=['bash','--noprofile','--norc','-c',shell,'as2-job',str(envscript),job['environment'],
                 '--physical-device',str(physical),
                 '--required-free-mib',str(job['required_free_mib']),'--timeout-s',str(job['timeout_s']),
                 '--out',str(job_dir),'--',*job['argv']]
        state['jobs'][job['id']]=dict(status='running',request=job,job_directory=str(job_dir),
            idle_observation=str(directory/'observations'/f'{stamp}.json'),physical_device_id=physical)
        state.pop('waiting_reason',None);write(directory/'state.json',state)
        with (directory/'submission.log').open('a') as log:
            # The existing runner owns the child session and its finite timeout.
            result=subprocess.run(command,stdout=log,stderr=log)
        receipt=json.loads((job_dir/'job.json').read_text()) if (job_dir/'job.json').exists() else dict(status='launch_error')
        item=state['jobs'][job['id']];item.update(status=receipt['status'],returncode=result.returncode,receipt=receipt)
        if result.returncode==0 and job['kind'] in ('baseline','tilemega'):
            try:item.update(metrics=collect_samples(job),status='collected')
            except (ValueError,OSError,json.JSONDecodeError) as exc:item.update(status='invalid_measurement',error=str(exc))
        for path in (job_dir,Path(job['output'])):
            if path.exists():
                subprocess.run([sys.executable,str(REPO/'scripts/as2_register.py'),'--path',str(path),
                    '--stage','AS2-focus','--category','result' if result.returncode==0 else 'failure','--completed'],
                    stdout=subprocess.DEVNULL,check=True)
        pause=serious_failure(job_dir,receipt)
        summarize(directory,state);write(directory/'state.json',state);previous={}
        print(json.dumps(dict(id=job['id'],status=item['status'],physical_device_id=physical,
                              metrics=item.get('metrics'),pause=pause)),flush=True)
        if pause:
            state.update(status='paused-device-safety',waiting_reason=pause);break
    else:state['status']='monitor_budget_finished'
    summarize(directory,state);write(directory/'state.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('init','run','summary'))
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=12)
    args=parser.parse_args()
    if not 0<args.hours<=24:parser.error('Require a finite monitor budget up to 24 hours')
    if args.action=='init':print(initialize(args.directory))
    elif args.action=='run':run(args.directory,args.hours)
    else:summarize(args.directory,json.loads((args.directory/'state.json').read_text()))


if __name__=='__main__':main()
