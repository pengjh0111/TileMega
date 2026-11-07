"""Execute independent selection processes on idle guarded cards."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from sm90d1_common import ROOT, PYTHON, read, write
import sm90r3_llm as llm
import sm90r4_dispatch as dispatch
from sm90x1_campaign import live_start

POLICY=ROOT/'configs/bench/sm90x1_parallel_policy.json'


def enabled(spec):
    policy=read(POLICY)
    return spec['kind']=='sel' and policy.get('parallel_selection_processes') is True and \
        policy.get('cross_gpu_comparisons_allowed') is True


def names(spec):
    return [f"{spec['name']}-{i:02d}-{label}" for i,label in enumerate(spec['order'])]


def isolate(jobs, name):
    selected=[dict(job, after=[]) for job in jobs if job['name']==name]
    if len(selected)!=1:raise ValueError('selection process is missing or duplicated: '+name)
    return selected


def execution():
    return dict(mode='parallel_processes',authorized_by_user=True,
                order_semantics='logical round labels; process execution may overlap',
                authorization=read(POLICY)['authorization'])


def one_process(template, unit, index):
    from sm90x1_hardware import audit_unit, quarantine
    spec=read(template);name=names(spec)[index]
    job=unit/'parallel/jobs'/f'{index:02d}'
    job.mkdir(parents=True,exist_ok=True)
    receipt=job/'process.json'
    if receipt.exists():
        if not audit_unit(job)['passed']:raise ValueError('cached parallel process failed guard audit')
        return read(receipt)
    for path in job.glob('try*/scheduler-process.json'):
        previous=read(path)
        if live_start(previous['pid'])==previous['start']:
            raise RuntimeError('existing parallel scheduler requires recovery before restart: '+str(path))
    for _ in range(3):
        gpu,lease=dispatch.claim('trial',str(job),poll=5)
        tries=list(job.glob('try[0-9]*'));attempt=job/f'try{len(tries)+1}';attempt.mkdir()
        try:
            local=dict(spec,gpu=gpu,concurrency='trial',execution=execution())
            write(attempt/'spec.json',local)
            subprocess.check_call([PYTHON,str(dispatch.LLM),'queue','--spec',str(attempt/'spec.json'),
                                   '--queue-dir',str(attempt/'queue')],cwd=ROOT,stdout=subprocess.DEVNULL)
            queue=next((attempt/'queue').glob('queue_*.json'))
            write(queue,isolate(read(queue),name))
            env=dict(os.environ,**llm.environment(gpu,'trial'))
            with (attempt/'scheduler.log').open('w') as log:
                scheduler=subprocess.Popen([PYTHON,str(ROOT/'scripts/sm90x1_scheduler.py'),
                    '--queue-dir',str(attempt/'queue'),'--out',str(attempt/'scheduler'),
                    '--policy',str(dispatch.POLICY),'--deadline-hours','24'],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(attempt/'scheduler-process.json',dict(pid=scheduler.pid,start=live_start(scheduler.pid)))
                stalled=dispatch.watch(scheduler,attempt/'scheduler/state.json','trial',gpu,60,
                                       poll=5,preflight_handoff=True)
        finally:
            lease.close()
        if stalled:continue
        state=read(attempt/'scheduler/state.json')
        if scheduler.returncode or state[name]['status']!='done':
            write(attempt/'failed.json',dict(state=state,code=scheduler.returncode,at=time.time()))
            continue
        write(job/'unit.json',dict(gpu=gpu,**{'class':'trial','try':attempt.name}))
        if not audit_unit(job)['passed']:
            quarantine(job,'parallel process failed owner audit');continue
        row=dict(state[name],gpu=gpu,process_spec=str(attempt/'spec.json'))
        write(receipt,row)
        return row
    raise RuntimeError('three parallel process attempts failed: '+name)


def adopted_state(unit):
    path=unit/'parallel-adoption.json'
    if not path.exists():return {},False
    adoption=read(path);source=unit/adoption['source_try']
    state=read(source/'scheduler/state.json');spec=read(source/'spec.json')
    rows={name:dict(state[name],gpu=spec['gpu'],process_spec=str(source/'spec.json'))
          for name in adoption['preserved_jobs']}
    failed=[name for name,row in rows.items() if row['status'] not in ('done','running')]
    if failed:raise ValueError('adopted serial process failed: '+str(failed))
    scheduler=adoption['scheduler']
    live=live_start(scheduler['pid'])==scheduler['start']
    return rows,live


def validate_processes(spec,state):
    from sm90x1_hardware import audit_samples
    expected=names(spec)
    if set(state)!=set(expected):raise ValueError('parallel selection process coverage differs')
    allowed=set(dispatch.allowed(read(dispatch.POLICY)))
    for i,(name,label) in enumerate(zip(expected,spec['order'])):
        record=state[name];gpu=record['gpu'];out=Path(record['out'])
        result=llm.job_result(state,name);process=read(out/'job/sm90r3-process.json')
        actual=read(record['process_spec']);identity=read(out/'job/physical-device-identity.json')
        samples=[json.loads(line) for line in (out/'guard.jsonl').read_text().splitlines()]
        if gpu not in allowed or result['gpu']!=gpu or actual['gpu']!=gpu:
            raise ValueError('parallel process device differs from guarded assignment')
        if process['fresh']!=i or process['arm']!=label or actual['order']!=spec['order']:
            raise ValueError('parallel process logical round differs')
        for field in ('decode_sha256','prefill_sha256'):
            if process[field]!=spec['arms'][label][field]:raise ValueError('parallel process artifact changed')
        if not audit_samples(samples)['passed']:raise ValueError('parallel process failed owner audit')
        expected_uuid=read(dispatch.POLICY)['idle_power_mapping_provenance']['uuid_by_gpu'][str(gpu)]
        if identity.get('nvml_uuid')!=expected_uuid or identity.get('cuda_uuid')!=expected_uuid.removeprefix('GPU-'):
            raise ValueError('parallel process physical UUID differs')


def run_unit(template,unit,cls):
    from sm90x1_hardware import audit_unit
    template=Path(template);unit=Path(unit);unit.mkdir(parents=True,exist_ok=True)
    if cls!='trial' or not enabled(read(template)):raise ValueError('parallel timing requires user-authorized trial mode')
    with (unit/'parallel.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (unit/'unit.json').exists():
            if not audit_unit(unit)['passed']:raise ValueError('accepted timing failed re-audit')
            return read(unit/'result.json')
        spec=read(template);all_names=names(spec);adopted,_=adopted_state(unit)
        parallel=unit/'parallel';parallel.mkdir(exist_ok=True)
        workers=int(read(POLICY).get('max_parallel_timing_processes',4))
        if not 1<=workers<=4:raise ValueError('parallel selection worker count must be 1..4')
        pending=[i for i,name in enumerate(all_names) if name not in adopted]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(one_process,template,unit,i):i for i in pending}
            completed={}
            while True:
                adopted,serial_live=adopted_state(unit)
                for future,index in futures.items():
                    if future.done():completed[all_names[index]]=future.result()
                running=[]
                for path in parallel.glob('jobs/*/try*/scheduler/state.json'):
                    running.extend(name for name,row in read(path).items() if row['status']=='running')
                done=sum(r['status']=='done' for r in adopted.values())+len(completed)
                write(parallel/'progress.json',dict(done=done,total=len(all_names),
                    running=running+[n for n,r in adopted.items() if r['status']=='running'],
                    max_workers=workers,at=time.time()))
                if len(completed)==len(futures) and not serial_live and all(r['status']=='done' for r in adopted.values()):break
                time.sleep(5)
        state={**adopted,**completed}
        local=read(next((parallel/'jobs').glob('*/try*/spec.json'))) if pending else read(unit/read(unit/'parallel-adoption.json')['source_try']/'spec.json')
        spec['arms']=local['arms'];spec['execution']=execution()
        validate_processes(spec,state)
        write(parallel/'spec.json',spec);write(parallel/'scheduler/state.json',{name:state[name] for name in all_names})
        subprocess.check_call([PYTHON,str(dispatch.LLM),'collect','--spec',str(parallel/'spec.json'),
            '--state',str(parallel/'scheduler/state.json'),'--out',str(unit/'result.json')],cwd=ROOT)
        result=read(unit/'result.json')
        for label,arm in result['arms'].items():arm['gpus']=sorted({r['gpu'] for r in result['rows'] if r['arm']==label})
        write(unit/'result.json',result)
        gpus=sorted({row['gpu'] for row in state.values()})
        write(unit/'unit.json',dict(mode='parallel_processes',gpu=';'.join(map(str,gpus)),gpus=gpus,
              **{'class':cls,'try':'parallel'},execution=execution(),processes=len(state)))
        if not audit_unit(unit)['passed']:raise ValueError('parallel selection failed final owner audit')
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',type=Path,required=True);p.add_argument('--unit',type=Path,required=True)
    args=p.parse_args();run_unit(args.template,args.unit,'trial')


if __name__=='__main__':main()
