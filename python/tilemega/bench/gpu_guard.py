#!/usr/bin/env python3
"""Hold the shared lock and reject entire steps when another user interferes."""
import argparse,contextlib,fcntl,json,os,signal,subprocess,time
from pathlib import Path

def require_allowed_gpu(policy, index, timing=False):
    constraints = [policy]
    if policy.get('gpu_allowlist_file'):
        constraints.append(json.loads(Path(policy['gpu_allowlist_file']).read_text()))
    for constraint in constraints:
        allowed = constraint.get('allowed_gpu_indices')
        if allowed is not None:
            if not isinstance(allowed, list) or not allowed or any(type(i) is not int or i < 0 for i in allowed):
                raise ValueError('allowed_gpu_indices must be a nonempty list of GPU indices')
            if int(index) not in allowed:
                raise ValueError(f'GPU {index} is excluded by the active user GPU restriction')
        timed = constraint.get('allowed_timing_gpu_indices')
        if timed is not None:
            if not isinstance(timed, list) or any(type(i) is not int or i < 0 for i in timed):
                raise ValueError('allowed_timing_gpu_indices must be a list of GPU indices')
            if timing and int(index) not in timed:
                raise ValueError(f'GPU {index} is restricted to untimed functional jobs')

def gpu_step_allowed(policy, index, step):
    """Leave timing work pending on cards authorized only for correctness."""
    try:
        require_allowed_gpu(policy, index, step.get('timing', step.get('gpu', False)))
    except ValueError:
        return False
    return True

def gpu():
    index=os.getenv('TILEMEGA_DEVICE_INDEX','0')
    raw=subprocess.check_output(['nvidia-smi','-i',index,'--query-gpu=memory.used,memory.free,memory.total,power.draw,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]
    used,free,total,power,util=map(float,raw.split(','))
    apps=subprocess.check_output(['nvidia-smi','-i',index,'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    owners={}
    for line in apps.splitlines():
        try:pid,size=map(int,line.split(','));owners[pid]=size
        except ValueError:continue
    detail=subprocess.check_output(['nvidia-smi','-i',index,
        '--query-gpu=clocks.current.sm,clocks.current.memory,temperature.gpu,clocks_event_reasons.active',
        '--format=csv,noheader,nounits'],text=True).splitlines()[0].split(',')
    sm,memory,temp=map(float,detail[:3])
    return dict(time=time.time(),gpu_index=int(index),used_mib=used,free_mib=free,total_mib=total,
                power_w=power,utilization_pct=util,owners=owners,hidden_mib=max(0,used-sum(owners.values())),
                sm_clock_mhz=sm,memory_clock_mhz=memory,temperature_c=temp,
                throttle_mask=int(detail[3].strip(),0))

def throttle_invalid(row):
    # Idle/application clocks are permitted; hardware, thermal and power-brake
    # slowdowns invalidate a round even if its numeric output is correct.
    return bool(row.get('throttle_mask',0)&(0x8|0x20|0x40|0x80))

def needs_numa_lock(env):
    return (env.get('TILEMEGA_TIMING')=='1'
            and env.get('TILEMEGA_NUMA_CONCURRENCY_VERIFIED')!='1'
            and env.get('TILEMEGA_NUMA_CONCURRENCY_TRIAL')!='1')

def idle_power(policy,index=None):
    index=str(os.getenv('TILEMEGA_DEVICE_INDEX','0') if index is None else index)
    return policy.get('idle_power_w_by_gpu',{}).get(index,policy['idle_power_w'])

def descendants(pid):
    own={pid};parents={}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:parents[int(path.parent.name)]=int(path.read_text().rsplit(')',1)[1].split()[1])
        except (OSError,ValueError,IndexError):pass
    while True:
        added={c for c,p in parents.items() if p in own}-own
        if not added:return own
        own.update(added)

def nvml_owners(own):
    host=os.environ.get('TILEMEGA_HOST_PROC')
    if not host:return own
    namespace=Path('/proc/self/ns/pid').stat().st_ino
    result=set()
    for status in Path(host).glob('[0-9]*/status'):
        try:
            if (status.parent/'ns/pid').stat().st_ino!=namespace:continue
            nspid=next(line for line in status.read_text().splitlines() if line.startswith('NSpid:'))
            if int(nspid.split()[-1]) in own:result.add(int(status.parent.name))
        except (OSError,StopIteration,ValueError):continue
    return result

def external(row,own,*,allow_unresolved=False):
    own=nvml_owners(own)
    foreign=set()
    for pid in row['owners']:
        if int(pid) in own:continue
        proc=Path('/proc')/str(pid)
        # NVML may expose host PIDs absent from this container's /proc.
        # An invisible owner is not evidence that the GPU is unoccupied.
        if not proc.exists():
            if not allow_unresolved:foreign.add(int(pid))
            continue
        try:
            exe=str((proc/'exe').resolve())
            if '.resources/' in exe and exe.endswith('/query'):continue
        except OSError:pass
        foreign.add(int(pid))
    return foreign

def idle(row,policy,needs=0):
    return (not external(row,descendants(os.getpid())) and row['utilization_pct']<=policy['max_util_pct']
            and row['power_w']<=idle_power(policy,row.get('gpu_index'))+policy['power_margin_w']
            and row['hidden_mib']<=policy['max_hidden_mib'] and row['free_mib']>=needs
            and not throttle_invalid(row))

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
    if not cmd:p.error('a child command is required')
    index=os.environ.get('TILEMEGA_DEVICE_INDEX')
    if index is None or not index.isdigit() or os.environ.get('CUDA_VISIBLE_DEVICES')!=index:
        p.error('explicit matching TILEMEGA_DEVICE_INDEX and CUDA_VISIBLE_DEVICES are required')
    policy=json.loads(a.policy.read_text())
    require_allowed_gpu(policy,index,os.environ.get('TILEMEGA_TIMING','1') != '0')
    a.out.mkdir(parents=True,exist_ok=True)
    limited=os.environ.get('TILEMEGA_GPU_PID_MODE')=='namespace_limited'
    def sample(phase,**extra):
        row=gpu()
        row['unresolved_host_pids']=[int(pid) for pid in row['owners'] if not Path('/proc',str(pid)).exists()]
        row['ownership_verified']=not limited
        with (a.out/'guard.jsonl').open('a') as f:f.write(json.dumps(dict(row,phase=phase,**extra))+'\n')
        return row
    def finish(code,reason):
        acceptance='provisional_container_pid_namespace' if limited else 'strict'
        (a.out/'guard_result.json').write_text(json.dumps(dict(code=code,reason=reason,
            ownership_verified=not limited,acceptance=acceptance if code==0 else 'rejected',
            numa_concurrency_trial=os.environ.get('TILEMEGA_NUMA_CONCURRENCY_TRIAL')=='1'))+'\n')
        print(reason,flush=True);return code
    lock=Path(os.environ['TILEMEGA_GPU_LOCK']);lock.parent.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        if needs_numa_lock(os.environ):
            node=os.environ.get('TILEMEGA_NUMA_NODE')
            if node is None:p.error('timing jobs require TILEMEGA_NUMA_NODE')
            numa=stack.enter_context(Path(f'/tmp/tilemega_numa{node}.lock').open('a'))
            fcntl.flock(numa,fcntl.LOCK_EX)
        handle=stack.enter_context(lock.open('a'))
        fcntl.flock(handle,fcntl.LOCK_EX)
        previous=set();rows=[]
        for i in range(policy['samples']):
            try:row=sample('preflight',sample=i)
            except Exception as error:return finish(75,'preflight telemetry unavailable: '+str(error))
            foreign=external(row,descendants(os.getpid()))
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
            try:row=sample('running')
            except Exception as error:
                stop(process,known);return finish(75,'running telemetry unavailable: '+str(error))
            if throttle_invalid(row):
                stop(process,known);return finish(75,'running throttle')
            foreign=external(row,own|{os.getpid()},allow_unresolved=limited)
            hidden=hidden+1 if row['hidden_mib']>baseline+policy['max_hidden_mib'] else 0
            if foreign & previous or hidden>=policy['hidden_violations']:
                stop(process,known);return finish(75,'running interference')
            previous=foreign
            if time.monotonic()-begin>a.timeout_s:
                stop(process,known);return finish(124,'guard timeout')
        # Clean retained workers after normal/error parent exit as well.
        stop(process,known)
        try:row=sample('after',exit_code=code)
        except Exception as error:return finish(75,'exit telemetry unavailable: '+str(error))
        if throttle_invalid(row):return finish(75,'exit throttle')
        foreign=external(row,own|{os.getpid()})
        if limited and foreign:return finish(75,'exit unresolved GPU owner')
        if foreign & previous or row['hidden_mib']>baseline+policy['max_hidden_mib'] or (code and foreign):
            return finish(75,'exit interference')
        return finish(code,'child exit '+str(code))
if __name__=='__main__':raise SystemExit(main())
