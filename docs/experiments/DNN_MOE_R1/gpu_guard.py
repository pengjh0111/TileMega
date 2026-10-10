#!/usr/bin/env python3
"""Hold the shared lock and reject entire steps when another user interferes."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path

def gpu():
    index=os.getenv('TILEMEGA_DEVICE_INDEX','0')
    raw=subprocess.check_output(['nvidia-smi','-i',index,'--query-gpu=memory.used,memory.free,memory.total,power.draw,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]
    used,free,total,power,util=map(float,raw.split(','))
    apps=subprocess.check_output(['nvidia-smi','-i',index,'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    owners={}
    for line in apps.splitlines():
        try:pid,size=map(int,line.split(','));owners[pid]=size
        except ValueError:continue
    return dict(time=time.time(),used_mib=used,free_mib=free,total_mib=total,power_w=power,utilization_pct=util,owners=owners,hidden_mib=max(0,used-sum(owners.values())))

def descendants(pid):
    own={pid};parents={}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:parents[int(path.parent.name)]=int(path.read_text().rsplit(')',1)[1].split()[1])
        except (OSError,ValueError,IndexError):pass
    while True:
        added={c for c,p in parents.items() if p in own}-own
        if not added:return own
        own.update(added)

def external(row,own):
    foreign=set()
    for pid in row['owners']:
        if int(pid) in own:continue
        proc=Path('/proc')/str(pid)
        if not proc.exists():continue
        try:
            exe=str((proc/'exe').resolve())
            if '.resources/' in exe and exe.endswith('/query'):continue
        except OSError:pass
        foreign.add(int(pid))
    return foreign

def idle(row,policy,needs=0):
    return (not external(row,descendants(os.getpid())) and row['utilization_pct']<=policy['max_util_pct']
            and row['power_w']<=policy['idle_power_w']+policy['power_margin_w']
            and row['hidden_mib']<=policy['max_hidden_mib'] and row['free_mib']>=needs)

def session_members(sid):
    members=set()
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            if int(path.read_text().rsplit(')',1)[1].split()[3])==sid:
                members.add(int(path.parent.name))
        except (OSError,ValueError,IndexError):pass
    return members

def process_start(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError):return None

def stop(process,known=None):
    own=descendants(process.pid)|session_members(process.pid)
    starts=dict(known or {})
    for pid in own:starts.setdefault(pid,process_start(pid))
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for pid,started in starts.items():
            # A captured descendant may exit before cleanup; never signal a
            # newly reused PID belonging to a different process.
            if started is None or process_start(pid)!=started:continue
            try:os.kill(pid,sig)
            except ProcessLookupError:pass
        if sig==signal.SIGTERM:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
            # Parent exit does not prove descendant exit. Reparented vLLM
            # workers retain this child's session and can ignore SIGTERM.
            for pid in session_members(process.pid):starts.setdefault(pid,process_start(pid))
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--needs-free-mib',type=int,default=12288);p.add_argument('--timeout-s',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    policy=json.loads(a.policy.read_text());a.out.mkdir(parents=True,exist_ok=True)
    def sample(phase,**extra):
        row=gpu()
        with (a.out/'guard.jsonl').open('a') as f:f.write(json.dumps(dict(row,phase=phase,**extra))+'\n')
        return row
    def finish(code,reason):
        (a.out/'guard_result.json').write_text(json.dumps(dict(code=code,reason=reason))+'\n');print(reason,flush=True);return code
    lock=Path(os.environ['TILEMEGA_GPU_LOCK']);lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        previous=set();rows=[]
        for i in range(policy['samples']):
            row=sample('preflight',sample=i);foreign=external(row,descendants(os.getpid()))
            # Do not start with a newly observed owner either; the two-sample
            # rule identifies interference during execution, not permission to start.
            if not idle(row,policy,a.needs_free_mib):return finish(75,'preflight occupied')
            previous=foreign;rows.append(row)
            if i+1<policy['samples']:time.sleep(policy['interval_s'])
        baseline=max(r['hidden_mib'] for r in rows)
        process=subprocess.Popen(cmd,env=dict(os.environ,TILEMEGA_GPU_LOCK_HELD='1'),start_new_session=True)
        (a.out/'pgid').write_text(str(process.pid)+'\n');begin=time.monotonic();hidden=0;known={process.pid:process_start(process.pid)};own={process.pid}
        while True:
            try:code=process.wait(timeout=policy['interval_s']);break
            except subprocess.TimeoutExpired:pass
            for pid in descendants(process.pid):known[pid]=process_start(pid)
            own={pid for pid,started in known.items() if started is not None and process_start(pid)==started}
            row=sample('running');foreign=external(row,own|{os.getpid()})
            hidden=hidden+1 if row['hidden_mib']>baseline+policy['max_hidden_mib'] else 0
            if foreign & previous or hidden>=policy['hidden_violations']:
                stop(process,known);return finish(75,'running interference')
            previous=foreign
            if time.monotonic()-begin>a.timeout_s:
                stop(process,known);return finish(124,'guard timeout')
        # Clean retained workers after normal/error parent exit as well.
        stop(process,known)
        row=sample('after',exit_code=code);foreign=external(row,own|{os.getpid()})
        if foreign & previous or row['hidden_mib']>baseline+policy['max_hidden_mib'] or (code and foreign):
            return finish(75,'exit interference')
        return finish(code,'child exit '+str(code))
if __name__=='__main__':raise SystemExit(main())
