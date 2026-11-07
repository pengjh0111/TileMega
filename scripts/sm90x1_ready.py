#!/usr/bin/env python3
"""Submit each ready trace variant without waiting for its complete cell."""
import argparse
import collections
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sm90d1_common import ROOT, PYTHON, read, write
import sm90r4_dispatch as dispatch
import sm90x1 as experiment
from sm90x1_experiment import gate
from sm90x1_hardware import audit_trace_unit


def inventory(cells, phase, run_dir=experiment.RUN):
    rows=[]
    for cell in cells:
        directory=run_dir/'cells'/cell['cell']/f'phase{phase}'
        plan=directory/'build-plan.json'
        if not plan.exists():continue
        for entry in read(plan):
            artifact=directory/entry['id']
            status='waiting_native'
            if (artifact/'trace/unit.json').exists():status='done'
            elif (artifact/'binding.json').exists():
                status='waiting_trace_build'
                if (artifact/'trace.so').exists() and (artifact/'trace.so.identity.json').exists():
                    status='trace_busy' if dispatch.held(artifact/'trace-unit.lock') else 'ready'
            rows.append(dict(cell=cell['cell'],variant=entry['id'],status=status,
                             **{'class':cell['class']}))
    return rows


def choose(rows, slots, active):
    if slots<=0 or 'machine' in active:return []
    ready=[row for row in rows if row['status']=='ready']
    trials=[row for row in ready if row['class']=='trial']
    if trials:return trials[:slots]
    # Only one machine waiter is admitted; its existing claim drains trial units.
    return [row for row in ready if row['class']=='machine'][:1] if not active else []


def key(row):
    return row['cell'],row['variant']


def plans_ready(cells,phase):
    return all((experiment.RUN/'cells'/c['cell']/f'phase{phase}'/'build-plan.json').exists() for c in cells)


def one(args):
    cell=experiment.cells(args.cell)[0]
    if cell['pilot']!=(args.phase==1):raise ValueError('cell does not belong to this phase')
    plan=read(experiment.directory(cell)/f'phase{args.phase}'/'build-plan.json')
    if args.variant not in {entry['id'] for entry in plan}:raise ValueError('variant is absent from the frozen plan')
    experiment.trace_cell(argparse.Namespace(phase=args.phase,tag=f'phase{args.phase}'),cell,args.variant)
    unit=experiment.directory(cell)/f'phase{args.phase}'/args.variant/'trace'
    if not audit_trace_unit(unit)['passed']:raise ValueError('trace hardware audit failed')


def prepare_one(row,phase):
    artifact=experiment.RUN/'cells'/row['cell']/f'phase{phase}'/row['variant']
    with (artifact/'trace-unit.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(row,status='busy')
        binding=read(artifact/'binding.json')
        experiment.validate_runtime(binding)
        experiment.prepare_trace_binary(artifact,binding,artifact/'trace-nvcc.lock')
    return dict(row,status='prepared')


def prepare(args):
    lock=(experiment.RUN/f'prepare-traces-phase{args.phase}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cells=[c for c in experiment.cells('all') if c['pilot']==(args.phase==1)]
    active={};attempts=collections.Counter();records=[]
    head=read(experiment.RUN/'measurement-head.json')['head']
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        while read(experiment.RUN/'campaign-state.json')['status']=='running':
            if experiment.head()!=head:raise ValueError('measurement HEAD changed')
            for identity,job in list(active.items()):
                if not job.done():continue
                try:record=job.result()
                except Exception as error:record=dict(cell=identity[0],variant=identity[1],status='failed',error=str(error))
                if record['status']=='busy':attempts[identity]-=1
                records.append(record);del active[identity]
            rows=inventory(cells,args.phase)
            for row in rows:
                identity=key(row)
                if len(active)>=args.max_workers:break
                if row['status']!='waiting_trace_build' or identity in active or attempts[identity]>=2:continue
                artifact=experiment.RUN/'cells'/row['cell']/f'phase{args.phase}'/row['variant']
                if dispatch.held(artifact/'trace-unit.lock'):continue
                attempts[identity]+=1;active[identity]=pool.submit(prepare_one,row,args.phase)
            write(experiment.RUN/f'prepare-traces-phase{args.phase}-state.json',dict(status='running',
                max_workers=args.max_workers,active=[dict(cell=k[0],variant=k[1]) for k in active],records=records,at=time.time()))
            if plans_ready(cells,args.phase) and rows and all(r['status'] not in ('waiting_native','waiting_trace_build') for r in rows) and not active:break
            time.sleep(args.poll)
    write(experiment.RUN/f'prepare-traces-phase{args.phase}-state.json',dict(status='complete',records=records,at=time.time()))


def post_one(args):
    directory=experiment.RUN/'cells'/args.cell/f'phase{args.phase}'
    with (directory/'post-flow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return post_locked(args)


def post_locked(args):
    for action,extra in [('analyze',()),('arms',()),('build',('--arms-only',)),('corr',()),('time',())]:
        subprocess.check_call([PYTHON,str(ROOT/'scripts/sm90x1.py'),action,'--phase',str(args.phase),
                               '--cell',args.cell,*extra],cwd=ROOT)
    subprocess.check_call([PYTHON,str(ROOT/'scripts/sm90x1.py'),'tables','--phase',str(args.phase),
                           '--cell','all'],cwd=ROOT)


def post_retry(directory,name,records,now,limit=3,retry_wait=30):
    failures=[r for r in records if r['cell']==name and r['code']!=0]
    if len(failures)>=limit:return False,'attempt_limit'
    if not failures:return True,'first_attempt'
    if now-max(r['at'] for r in failures)<retry_wait:return False,'retry_wait'
    for path in directory.glob('*/corr/result.json'):
        if read(path).get('pass') is False:return False,'correctness_failed'
    for path in directory.glob('*/build.json'):
        if read(path).get('status',0)!=0:return False,'build_failed'
    states=list(directory.glob('*/corr/try*/scheduler/state.json'))+list(directory.glob('time/try*/scheduler/state.json'))
    if not states:return False,'no_process_failure_receipt'
    state=read(max(states,key=lambda p:p.stat().st_mtime_ns))
    for row in state.values():
        if 'out' not in row:continue
        for path in (Path(row['out'])/'job/measurements').glob('c[12].json'):
            if read(path).get('pass') is False:return False,'correctness_failed'
    failed=[r.get('exit_code') for r in state.values() if r['status']=='failed']
    if failed and set(failed)<={247,137,-9,143,-15,75}:return True,'abnormal_process_exit'
    return False,'failure_requires_review'


def post(args):
    lock=(experiment.RUN/f'post-phase{args.phase}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selected=[c for c in experiment.cells('all') if c['pilot']==(args.phase==1)]
    state_path=experiment.RUN/f'post-phase{args.phase}-state.json'
    active={};records=read(state_path).get('records',[]) if state_path.exists() else []
    try:
        while read(experiment.RUN/'campaign-state.json')['status']=='running':
            for name,job in list(active.items()):
                code=job['process'].poll()
                if code is None:continue
                job['handle'].close();records.append(dict(cell=name,code=code,log=job['log'],at=time.time()))
                del active[name]
            data=inventory(selected,args.phase)
            done=[];blocked=[]
            for cell in selected:
                name=cell['cell'];directory=experiment.directory(cell)/f'phase{args.phase}'
                if (directory/'time/unit.json').exists():done.append(name);continue
                if name in active or name in args.defer_cell:continue
                variants=[r for r in data if r['cell']==name]
                if not variants or not all(r['status']=='done' for r in variants):continue
                admit,reason=post_retry(directory,name,records,time.time(),args.max_attempts,args.retry_wait)
                if not admit:blocked.append(dict(cell=name,reason=reason));continue
                if dispatch.held(directory/'post-flow.lock') or any(dispatch.held(directory/f'action-{action}.lock')
                        for action in ('analyze','arms','build','corr','time')):continue
                if len(active)>=args.max_workers:continue
                log=experiment.RUN/'overlap-logs'/f'{time.time_ns()}-post-ready-{name}.log';handle=log.open('w')
                process=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'post-one','--phase',str(args.phase),
                                          '--cell',name],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                active[name]=dict(process=process,handle=handle,log=str(log))
            write(state_path,dict(status='running',active=[dict(cell=k,pid=v['process'].pid) for k,v in active.items()],
                                  complete_cells=done,blocked=blocked,records=records,at=time.time()))
            if len(done)==len(selected) and not active:break
            time.sleep(args.poll)
        write(state_path,dict(status='complete',records=records,at=time.time()))
    finally:
        for job in active.values():job['process'].wait();job['handle'].close()


def run(args):
    lock=(experiment.RUN/f'ready-phase{args.phase}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cells=[c for c in experiment.cells('all') if c['pilot']==(args.phase==1)]
    head=read(experiment.RUN/'measurement-head.json')['head']
    active={};attempts=collections.Counter();records=[]
    state_path=experiment.RUN/f'ready-phase{args.phase}-state.json'
    policy=read(dispatch.POLICY)
    visible=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader,nounits'],text=True)
    allowed=sorted(set(map(int,visible.split()))&set(dispatch.allowed(policy)))
    try:
        while True:
            for identity,job in list(active.items()):
                code=job['process'].poll()
                if code is None:continue
                job['log_handle'].close()
                records.append(dict(cell=identity[0],variant=identity[1],code=code,log=job['log'],at=time.time()))
                del active[identity]
            campaign=read(experiment.RUN/'campaign-state.json')
            if campaign['status']!='running':break
            if experiment.head()!=head:raise ValueError('measurement HEAD changed')
            rows=inventory(cells,args.phase)
            if plans_ready(cells,args.phase) and rows and all(row['status']=='done' for row in rows) and not active:break
            claims=dispatch.live_claims()
            idle=[gpu for gpu in allowed if gpu not in claims and dispatch.idle_card(gpu,policy)]
            candidates=[row for row in rows if key(row) not in active and attempts[key(row)]<2]
            slots=min(len(idle),args.max_workers-len(active))
            for row in choose(candidates,slots,[job['class'] for job in active.values()]):
                identity=key(row);attempts[identity]+=1
                log=experiment.RUN/'overlap-logs'/f'{time.time_ns()}-ready-{identity[0]}-{identity[1]}.log'
                output=log.open('w')
                argv=[PYTHON,str(Path(__file__).resolve()),'one','--phase',str(args.phase),
                      '--cell',identity[0],'--variant',identity[1]]
                process=subprocess.Popen(argv,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                active[identity]=dict(process=process,log_handle=output,log=str(log),
                                      **{'class':row['class']},started=time.time())
            write(state_path,dict(status='running',head=head,poll_s=args.poll,
                counts=dict(collections.Counter(row['status'] for row in rows)),
                idle_gpus_at_poll=idle,claims=claims,
                active=[dict(cell=k[0],variant=k[1],pid=v['process'].pid,**{'class':v['class']},
                             started=v['started']) for k,v in active.items()],records=records,at=time.time()))
            time.sleep(args.poll)
        write(state_path,dict(status='complete',records=records,at=time.time()))
    except Exception as error:
        write(state_path,dict(status='failed',error=str(error),records=records,at=time.time()))
        raise
    finally:
        # In-flight guarded units finish intact even if their submitter stops.
        for job in active.values():
            job['process'].wait()
            job['log_handle'].close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','one','prepare','post','post-one'))
    parser.add_argument('--phase',type=int,choices=(1,2),default=2)
    parser.add_argument('--cell')
    parser.add_argument('--variant')
    parser.add_argument('--max-workers',type=int,default=4)
    parser.add_argument('--poll',type=float,default=5)
    parser.add_argument('--max-attempts',type=int,default=3)
    parser.add_argument('--retry-wait',type=float,default=30)
    parser.add_argument('--defer-cell',action='append',default=[])
    args=parser.parse_args()
    if args.max_workers<1 or args.poll<=0:parser.error('worker count and poll must be positive')
    if args.max_attempts<1 or args.retry_wait<0:parser.error('attempt limit and retry wait are invalid')
    if args.action=='one' and (not args.cell or not args.variant):parser.error('one requires cell and variant')
    if args.action=='post-one' and not args.cell:parser.error('post-one requires cell')
    gate(args.phase)
    (experiment.RUN/'overlap-logs').mkdir(parents=True,exist_ok=True)
    return {'one':one,'run':run,'prepare':prepare,'post':post,'post-one':post_one}[args.action](args)


if __name__=='__main__':
    main()
