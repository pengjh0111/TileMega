"""Watch free C550 cards and run immutable, dependency-ordered experiment jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tilemega.bench import gpu_guard

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);temporary=p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(p)

def submit(root,name,definition):
    p=root/'jobs'/name/'definition.json'
    if p.exists():
        if read(p)!=definition:raise ValueError('queued definition changed: '+name)
    else:save(p,definition)

def initialize(root,builds):
    binary=builds/'hardware_probe';graph=builds/'graph_probe'
    if not binary.exists() or not graph.exists():raise ValueError('compile both probes first')
    for n in range(5):
        submit(root,f'graph_p{n}',dict(kind='graph_probe',command=[str(graph)],inputs=[ref(graph),ref(ROOT/'experiments/mx8/graph_probe.cu')],
            priority=0,needs_mib=2048,timeout_s=300,after=[]))
    for gpu in range(8):
        for item in ('hbm_vector','hbm_lds','bf16_mma'):
            for n in range(5):
                submit(root,f'{item}_g{gpu}_p{n}',dict(kind='hardware',item=item,card=gpu,command=[str(binary),item],
                    inputs=[ref(binary),ref(ROOT/'experiments/mx8/hardware_probe.cu')],priority=1,needs_mib=12288,timeout_s=900,after=[]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs/mx8/queue')
    parser.add_argument('--initialize-probes',action='store_true')
    parser.add_argument('--builds',type=Path,default=ROOT/'runs/mx8/hardware_build')
    parser.add_argument('--max-parallel',type=int,default=4)
    parser.add_argument('--poll-s',type=float,default=10)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if not 1<=args.max_parallel<=4:parser.error('use one to four concurrent single-card jobs')
    import fcntl
    lock=(out/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.initialize_probes:initialize(out,args.builds.resolve())
    policy=read(ROOT/'configs/bench/guard_policy_c550.json');active={};attempts={};retry_at={}
    if (out/'RUNNER.json').exists():
        previous=read(out/'RUNNER.json')
        save(out/f'RUNNER_{previous["PID"]}.json',previous)
    save(out/'RUNNER.json',dict(PID=os.getpid(),source=ref(__file__),started=time.time(),parallel=args.max_parallel,
        scope='hardware peaks and both AB4c variants; main-experiment and external baseline timings frozen',retry_limit=None))
    try:
        while True:
            for name,job in list(active.items()):
                code=job['process'].poll()
                if code is None:continue
                job['stream'].close();folder=job['folder'];definition=job['definition']
                process=dict(PID=job['process'].pid,returncode=code,start=job['start'],end=time.time(),gpu=job['gpu'],command=job['command'])
                save(folder/'PROCESS.json',process)
                guard=folder/'guard/guard_result.json'
                if sha(job['definition_path'])!=job['definition_sha']:raise ValueError('live job definition changed')
                if code==75:
                    save(folder/'REJECTED.json',dict(reason='GPU guard refused/interrupted; original attempt retained',process=ref(folder/'PROCESS.json')))
                    retry_at[name]=time.time()+args.poll_s
                    print(name,'guard interference; requeue without retry cap',flush=True)
                elif code:
                    save(folder.parent/'FAILED.json',dict(process=ref(folder/'PROCESS.json'),reason='non-interference failure; diagnosis required',definition=definition))
                    print(name,'FAILED',code,flush=True)
                else:
                    if not guard.exists() or read(guard)['code']:raise ValueError('success without passing guard')
                    for item in definition['inputs']:
                        if sha(item['path'])!=item['sha256']:raise ValueError('experiment input changed')
                    payload=None
                    for line in (folder/'stdout.log').read_text(errors='replace').splitlines():
                        if line.startswith('{'):
                            try:payload=json.loads(line)
                            except ValueError:pass
                    if definition['kind']=='graph_probe' and (not payload or not payload.get('pass')):raise ValueError('graph probe lacks exact pass')
                    if definition['kind']=='hardware' and (not payload or not payload.get('samples')):raise ValueError('hardware probe has no samples')
                    save(folder.parent/'ACCEPTED.json',dict(folder=str(folder),definition=ref(job['definition_path']),
                        process=ref(folder/'PROCESS.json'),guard=ref(guard),stdout=ref(folder/'stdout.log'),payload=payload))
                    print(name,'accepted',flush=True)
                del active[name]
            definitions={p.parent.name:(p,read(p)) for p in sorted((out/'jobs').glob('*/definition.json'))}
            complete={n for n,(p,d) in definitions.items() if (p.parent/'ACCEPTED.json').exists()}
            failed={n for n,(p,d) in definitions.items() if (p.parent/'FAILED.json').exists()}
            pending=[n for n in definitions if n not in complete|failed|set(active)]
            used={j['gpu'] for j in active.values()}
            available={};errors={}
            if len(active)<args.max_parallel and pending:
                for gpu in (7,6,5,4,3,2,1,0):
                    if gpu in used:continue
                    try:
                        row=gpu_guard.gpu(policy,gpu)
                        if gpu_guard.idle(row,policy,2048,gpu):available[gpu]=row
                    except Exception as error:errors[gpu]=str(error)
                pending.sort(key=lambda n:(definitions[n][1].get('priority',10),attempts.get(n,0),n))
                for name in pending:
                    if len(active)>=args.max_parallel:break
                    path,definition=definitions[name]
                    # A pending definition may be withdrawn before dispatch.
                    if not path.exists():continue
                    if read(path)!=definition:continue
                    if retry_at.get(name,0)>time.time() or not set(definition.get('after',[]))<=complete:continue
                    chosen=next((g for g,row in available.items() if g not in used and
                        ('card' not in definition or definition['card']==g) and row['free_mib']>=definition.get('needs_mib',12288)),None)
                    if chosen is None:continue
                    for item in definition['inputs']:
                        if sha(item['path'])!=item['sha256']:raise ValueError('queued input changed: '+name)
                    count=len(list(path.parent.glob('attempt*')));folder=path.parent/f'attempt{count:03d}';folder.mkdir()
                    command=[str(ROOT/'scripts/mx_gpu_job.sh'),str(chosen),str(definition.get('timeout_s',900)),str(folder/'guard')]
                    command += [str(s).replace('{out}',str(folder)) for s in definition['command']]
                    env=dict(os.environ,PYTHONPATH=str(ROOT/'python'),MX_GPU_WAIT_IDLE='1',MX_GPU_IDLE_WAIT_S='-1',
                        MX_GPU_POLL_S=str(args.poll_s),NEEDS_MIB=str(definition.get('needs_mib',12288)),TILEMEGA_GPU_JOB_LOG=str(out/'gpu_jobs.log'))
                    for key in ('TILEMEGA_DEVICE_GROUP','TILEMEGA_GPU_LOCK_HELD','TILEMEGA_GPU_LOCK'):env.pop(key,None)
                    stream=(folder/'stdout.log').open('x');process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    active[name]=dict(process=process,stream=stream,folder=folder,definition=definition,definition_path=path,
                        definition_sha=sha(path),gpu=chosen,start=time.time(),command=command)
                    attempts[name]=count+1;used.add(chosen)
                    save(folder/'RUNNING.json',dict(PID=process.pid,gpu=chosen,command=command,start=time.time()))
                    print(name,'started GPU',chosen,'attempt',count,flush=True)
            save(out/'STATUS.json',dict(time=time.time(),runner_PID=os.getpid(),accepted=len(complete),failed=len(failed),pending=len([n for n in pending if n not in active]),
                active={n:dict(PID=j['process'].pid,gpu=j['gpu'],folder=str(j['folder'])) for n,j in active.items()},device_errors=errors,
                waiting_for_new_definitions=not pending and not active,closed=(out/'INPUTS_CLOSED').exists()))
            if (out/'INPUTS_CLOSED').exists() and not active and not pending:break
            time.sleep(args.poll_s)
    finally:
        for job in active.values():
            gpu_guard.stop(job['process']);job['stream'].close()
            save(job['folder']/'INTERRUPTED.json',dict(reason='queue stopped; do not promote partial output',time=time.time()))

if __name__=='__main__':main()
