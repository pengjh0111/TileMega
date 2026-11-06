#!/usr/bin/env python3
"""Hold the shared lock and reject entire steps when another user interferes."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
from contextlib import ExitStack, contextmanager

from tilemega.bench import mx_smi

def gpu(policy,index=None):
    index=os.getenv('TILEMEGA_DEVICE_INDEX','0') if index is None else str(index)
    row=mx_smi.device(index)
    # Other containers' processes are invisible to mx-smi; their memory is
    # what remains after this container's owners and the driver baseline.
    row['hidden_mib']=max(0,row['used_mib']-policy['idle_vram_mib']-sum(row['owners'].values()))
    return row

def device_openers(pids):
    """PIDs among `pids` holding an open MetaX device file."""
    openers=set()
    for pid in pids:
        try:
            for fd in Path('/proc',str(pid),'fd').iterdir():
                try:target=os.readlink(fd)
                except OSError:continue
                if target=='/dev/mxcd' or target.startswith('/dev/dri/'):openers.add(pid);break
        except OSError:pass
    return openers

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
    # Listed PIDs are container-local, so any owner outside this job's tree is
    # foreign; more listed owners than device-opening own processes is too.
    foreign={int(pid) for pid in row['owners'] if int(pid) not in own}
    if len(row['owners'])>len(device_openers(own)):foreign.add(-1)
    return foreign

def idle(row,policy,needs=0,index=None):
    index=os.getenv('TILEMEGA_DEVICE_INDEX','0') if index is None else str(index)
    idle_w=policy.get('idle_power_w_by_gpu',{}).get(index,policy['idle_power_w'])
    return (row.get('availability')=='Available' and not external(row,descendants(os.getpid())) and row['utilization_pct']<=policy['max_util_pct']
            and row['power_w']<=idle_w+policy['power_margin_w']
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

def running_sample(index,sample,process,known):
    for attempt in range(2):
        row=sample(index,'running' if attempt==0 else 'running_recheck')
        for pid in descendants(process.pid)|session_members(process.pid):known[pid]=process_start(pid)
        own={pid for pid,started in known.items() if started is not None and process_start(pid)==started}
        foreign=external(row,own|{os.getpid()})
        if not foreign:break
    return row,foreign,own

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

def device_indices():
    group=os.getenv('TILEMEGA_DEVICE_GROUP')
    if group is None:return [os.getenv('TILEMEGA_DEVICE_INDEX','0')]
    try:indices=sorted(set(int(g) for g in group.split(',')))
    except ValueError:raise ValueError('device group requires integer GPU indices') from None
    if len(indices) not in (2,4,8) or not set(indices)<=set(range(8)):
        raise ValueError('device group requires 2, 4 or 8 GPUs from 0 through 7')
    return list(map(str,indices))

@contextmanager
def device_locks(indices):
    # The global order prevents two overlapping groups from deadlocking and
    # uses exactly the same locks as the existing single-GPU launcher.
    with ExitStack() as stack:
        for index in sorted(indices,key=int):
            name=(f'/tmp/tilemega_mx_gpu{index}.lock' if os.getenv('TILEMEGA_DEVICE_GROUP')
                  else os.environ['TILEMEGA_GPU_LOCK'])
            lock=Path(name);lock.parent.mkdir(parents=True,exist_ok=True)
            handle=stack.enter_context(lock.open('a'))
            fcntl.flock(handle,fcntl.LOCK_EX)
        yield

def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--needs-free-mib',type=int,default=12288);p.add_argument('--timeout-s',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if not cmd:p.error('a child command is required')
    try:indices=device_indices()
    except ValueError as error:p.error(str(error))
    policy=json.loads(a.policy.read_text());a.out.mkdir(parents=True,exist_ok=True)
    def sample(index,phase,**extra):
        row=gpu(policy,index)
        with (a.out/'guard.jsonl').open('a') as f:f.write(json.dumps(dict(row,gpu_index=index,phase=phase,**extra))+'\n')
        return row
    def finish(code,reason):
        (a.out/'guard_result.json').write_text(json.dumps(dict(code=code,reason=reason,gpu_indices=indices))+'\n');print(reason,flush=True);return code
    if os.environ.get('TILEMEGA_GUARD_MAX_HIDDEN_MIB'):
        policy['max_hidden_mib']=float(os.environ['TILEMEGA_GUARD_MAX_HIDDEN_MIB'])
    with device_locks(indices):
        previous={g:set() for g in indices};rows={g:[] for g in indices}
        for i in range(policy['samples']):
            for g in indices:
                try:row=sample(g,'preflight',sample=i)
                except Exception as error:return finish(75,'preflight telemetry unavailable: '+str(error))
                if row.get('availability')!='Available':return finish(75,'preflight unavailable GPU '+g)
                if not idle(row,policy,a.needs_free_mib,g):
                    settle=policy.get('preflight_settle_s',0);waited=0.0
                    while settle and waited<settle:
                        time.sleep(policy['interval_s']);waited+=policy['interval_s']
                        try:row=sample(g,'preflight_settle',sample=i)
                        except Exception as error:return finish(75,'preflight telemetry unavailable: '+str(error))
                        if idle(row,policy,a.needs_free_mib,g):break
                    else:return finish(75,'preflight occupied GPU '+g)
                    if not idle(row,policy,a.needs_free_mib,g):return finish(75,'preflight occupied GPU '+g)
                previous[g]=external(row,descendants(os.getpid()));rows[g].append(row)
            if i+1<policy['samples']:time.sleep(policy['interval_s'])
        baseline={g:max(r['hidden_mib'] for r in rows[g]) for g in indices}
        process=subprocess.Popen(cmd,env=dict(os.environ,TILEMEGA_GPU_LOCK_HELD='1'),start_new_session=True)
        (a.out/'pgid').write_text(str(process.pid)+'\n');begin=time.monotonic()
        hidden={g:0 for g in indices};known={process.pid:process_start(process.pid)};own={process.pid}
        while True:
            try:code=process.wait(timeout=policy['interval_s']);break
            except subprocess.TimeoutExpired:pass
            for pid in descendants(process.pid)|session_members(process.pid):known[pid]=process_start(pid)
            own={pid for pid,started in known.items() if started is not None and process_start(pid)==started}
            for g in indices:
                try:row,foreign,own=running_sample(g,sample,process,known)
                except Exception as error:
                    stop(process,known);return finish(75,'running telemetry unavailable: '+str(error))
                if row.get('availability')!='Available':
                    stop(process,known);return finish(75,'running unavailable GPU '+g)
                hidden[g]=hidden[g]+1 if row['hidden_mib']>baseline[g]+policy['max_hidden_mib'] else 0
                if foreign & previous[g] or hidden[g]>=policy['hidden_violations']:
                    stop(process,known);return finish(75,'running interference GPU '+g)
                previous[g]=foreign
            if time.monotonic()-begin>a.timeout_s:
                stop(process,known);return finish(124,'guard timeout')
        stop(process,known)
        own|=descendants(process.pid)|session_members(process.pid)
        for g in indices:
            try:row=sample(g,'after',exit_code=code)
            except Exception as error:return finish(75,'exit telemetry unavailable: '+str(error))
            if row.get('availability')!='Available':return finish(75,'exit unavailable GPU '+g)
            foreign=external(row,own|{os.getpid()})
            if foreign & previous[g] or row['hidden_mib']>baseline[g]+policy['max_hidden_mib'] or (code and foreign):
                return finish(75,'exit interference GPU '+g)
        return finish(code,'child exit '+str(code))
if __name__=='__main__':raise SystemExit(main())
