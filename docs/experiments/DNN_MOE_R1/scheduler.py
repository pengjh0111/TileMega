#!/usr/bin/env python3
"""Single dynamic queue; timing runs only under gpu_guard's shared lock."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from gpu_guard import gpu,idle,stop

def acquire_command_lock(command,lock_path):
    """Separate shared-lock queue time from the command's execution timeout."""
    if len(command)<3 or Path(command[0]).name!='flock' or command[1]!=lock_path:
        return command,None
    descriptor=os.open(lock_path,os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(descriptor,fcntl.LOCK_EX)
    except BaseException:
        os.close(descriptor);raise
    # flock on an inherited descriptor uses the same open-file description.
    # Opening the pathname again while the parent holds it would deadlock.
    # Descriptor mode takes no command operands; pass argv through a shell
    # without interpolating any command text or reopening the lock file.
    return ['bash','-c','flock -x "$1" || exit "$?"; shift; exec "$@"',
            'dm-lock',str(descriptor),*command[2:]],descriptor

def main():
    p=argparse.ArgumentParser();p.add_argument('--queue-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--policy',type=Path,required=True);p.add_argument('--deadline-hours',type=float,default=96);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);a.queue_dir.mkdir(parents=True,exist_ok=True)
    singleton=(a.out/'scheduler.pid').open('a+');fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB);singleton.truncate(0);singleton.write(str(os.getpid())+'\n');singleton.flush()
    policy=json.loads(a.policy.read_text());state_path=a.out/'state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {};deadline=time.time()+a.deadline_hours*3600;observed=None
    os.environ.setdefault('TILEMEGA_GPU_LOCK','/root/r14_work/gpu.lock')
    def persist():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    def record(name,r,last):
        with (a.out/'progress.tsv').open('a') as f:f.write('\t'.join(map(str,[name,r['status'],r.get('attempts',0),r.get('wait_s',0),r.get('run_s',0),last.replace('\t',' ')[:500]]))+'\n')
        marker=a.out/(name+'.'+r['status']);marker.write_text(json.dumps(r)+'\n');persist()
    while time.time()<deadline:
        steps={}
        for path in sorted(a.queue_dir.glob('queue_*.json')):
            for step in json.loads(path.read_text()):
                if step['name'] in steps:raise ValueError('duplicate step '+step['name'])
                steps[step['name']]=step;state.setdefault(step['name'],dict(status='pending',attempts=0,enqueued=time.time(),retry_at=0))
        ready=[]
        for name,s in steps.items():
            r=state[name]
            if r['status']!='pending':continue
            if any(state.get(dep,{}).get('status') in ('failed','skipped','not_run') for dep in s.get('after',[])):
                r['status']='skipped';record(name,r,'required dependency failed');continue
            if not all(state.get(dep,{}).get('status')=='done' for dep in s.get('after',[])):continue
            if not all(state.get(dep,{}).get('status') in ('done','failed','skipped','not_run') for dep in s.get('after_any',[])):continue
            if r['retry_at']<=time.time():ready.append(s)
        ready.sort(key=lambda s:(s.get('priority',100),s['name']))
        gpu_steps=[s for s in ready if s.get('gpu')];chosen=None
        if gpu_steps:
            try:row=gpu();available=idle(row,policy,min(s.get('needs_free_mib',0) for s in gpu_steps))
            except Exception as e:row=dict(error=str(e));available=False
            signature=(available,tuple(row.get('owners',{})),row.get('hidden_mib',0)>policy['max_hidden_mib'])
            if signature!=observed:
                with (a.out/'occupancy.jsonl').open('a') as f:f.write(json.dumps(dict(row,available=available))+'\n')
                observed=signature
            if available:chosen=next((s for s in gpu_steps if idle(row,policy,s.get('needs_free_mib',0))),None)
        if chosen is None:chosen=next((s for s in ready if not s.get('gpu')),None)
        if chosen is None:persist();time.sleep(60);continue
        s=chosen;name=s['name'];r=state[name];folder=Path(s.get('out',a.out.parent/'raw'/name));folder.mkdir(parents=True,exist_ok=True)
        cmd=list(s['command']);env=dict(os.environ,**s.get('env',{}));env['TILEMEGA_GPU_LOCK']=os.environ['TILEMEGA_GPU_LOCK']
        if r['attempts']:cmd+=s.get('retry_args',[])
        if s.get('gpu'):
            cmd=[sys.executable,str(Path(__file__).with_name('gpu_guard.py')),'--policy',str(a.policy),'--out',str(folder),'--needs-free-mib',str(s.get('needs_free_mib',12288)),'--timeout-s',str(s['timeout_s']),'--']+cmd
        lock_begin=time.monotonic();lock_fd=None
        if not s.get('gpu'):cmd,lock_fd=acquire_command_lock(cmd,env['TILEMEGA_GPU_LOCK'])
        r['lock_wait_s']=time.monotonic()-lock_begin
        r['attempts']+=1;r['status']='running';r['wait_s']=time.time()-r['enqueued'];persist();begin=time.monotonic()
        log=a.out/(name+'.log')
        try:
            with log.open('a') as f:
                f.write('\nATTEMPT '+str(r['attempts'])+' '+str(time.time())+'\n');f.flush()
                proc=subprocess.Popen(cmd,cwd=s.get('cwd'),env=env,stdout=f,stderr=subprocess.STDOUT,
                    start_new_session=True,pass_fds=(() if lock_fd is None else (lock_fd,)))
                r['pid']=proc.pid;persist()
                try:code=proc.wait(timeout=s['timeout_s']+(600 if s.get('gpu') else 0))
                except subprocess.TimeoutExpired:
                    if s.get('gpu') and (folder/'pgid').exists():
                        try:os.killpg(int((folder/'pgid').read_text()),signal.SIGKILL)
                        except ProcessLookupError:pass
                    # Kill the whole CPU session including GNU timeout's separate groups.
                    subprocess.run(['pkill','-KILL','-s',str(proc.pid)],check=False);stop(proc);code=124
        finally:
            if lock_fd is not None:os.close(lock_fd)
        r['run_s']=time.monotonic()-begin;r['exit_code']=code
        if code==75:
            result=json.loads((folder/'guard_result.json').read_text()) if (folder/'guard_result.json').exists() else {}
            cooldown=300 if result.get('reason','').startswith('preflight') else 600
            r.update(status='pending',retry_at=time.time()+cooldown)
        else:r['status']='done' if code==0 else 'failed'
        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2048));lines=f.read().decode(errors='replace').strip().splitlines()
        record(name,r,lines[-1] if lines else '')
    for name,r in state.items():
        if r['status']=='pending':r['status']='not_run';record(name,r,'deadline reached')
if __name__=='__main__':main()
