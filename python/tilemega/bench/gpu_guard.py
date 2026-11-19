#!/usr/bin/env python3
"""Hold the shared lock and reject entire steps when another user interferes."""
import argparse,contextlib,fcntl,json,os,signal,subprocess,time,uuid
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

def single_card_mask(policy, index):
    return str(policy.get('cuda_visible_devices', index))


def smi(*args):
    return subprocess.check_output(['nvidia-smi', *args], text=True, timeout=5)


def gpu(index=None, *, strict=False):
    index=str(os.getenv('TILEMEGA_DEVICE_INDEX','0') if index is None else index)
    raw=smi('-i',index,'--query-gpu=memory.used,memory.free,memory.total,power.draw,utilization.gpu','--format=csv,noheader,nounits').splitlines()[0]
    used,free,total,power,util=map(float,raw.split(','))
    apps=smi('-i',index,'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits')
    owners={}
    for line in apps.splitlines():
        try:pid,size=map(int,line.split(','));owners[pid]=size
        except ValueError:
            if strict and line.strip():raise
    detail=smi('-i',index,
        '--query-gpu=clocks.current.sm,clocks.current.memory,temperature.gpu,clocks_event_reasons.active'
        + (',clocks_event_reasons_counters.hw_thermal_slowdown,clocks_event_reasons_counters.sw_thermal_slowdown'
           if strict else ''),
        '--format=csv,noheader,nounits').splitlines()[0].split(',')
    sm,memory,temp=map(float,detail[:3])
    row = dict(time=time.time(),gpu_index=int(index),used_mib=used,free_mib=free,total_mib=total,
                power_w=power,utilization_pct=util,owners=owners,hidden_mib=max(0,used-sum(owners.values())),
                sm_clock_mhz=sm,memory_clock_mhz=memory,temperature_c=temp,
                throttle_mask=int(detail[3].strip(),0))
    if strict:row['thermal_counters_us'] = [int(value) for value in detail[4:6]]
    return row

def verify_device(policy, row):
    if policy.get('gpu_uuid'):
        index = str(row['gpu_index'])
        observed = smi('-i',index,
            '--query-gpu=uuid','--format=csv,noheader').strip()
        if observed != policy['gpu_uuid']:
            raise ValueError(f'NVML index {index} maps to {observed}, expected {policy["gpu_uuid"]}')
        row['gpu_uuid'] = observed
    return row

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

def proc_files(root,name):
    # Glob checks each leaf before yielding it, outside the callers' race guard.
    return (path/name for path in Path(root).iterdir() if path.name.isdecimal())

def descendant_starts(pid, root='/proc'):
    own={pid};records={}
    for path in proc_files(root,'stat'):
        try:
            fields = path.read_text().rsplit(')',1)[1].split()
            records[int(path.parent.name)] = (int(fields[1]), fields[19])
        except (OSError,ValueError,IndexError):pass
    while True:
        added={child for child,(parent,_) in records.items() if parent in own}-own
        if not added:return {child:records[child][1] for child in own if child in records}
        own.update(added)

def descendants(pid):
    return {pid} | set(descendant_starts(pid))

def nvml_owners(own):
    host=os.environ.get('TILEMEGA_HOST_PROC')
    if not host:return own
    namespace=Path('/proc/self/ns/pid').stat().st_ino
    result=set()
    for status in proc_files(host,'status'):
        try:
            if (status.parent/'ns/pid').stat().st_ino!=namespace:continue
            nspid=next(line for line in status.read_text().splitlines() if line.startswith('NSpid:'))
            if int(nspid.split()[-1]) in own:result.add(int(status.parent.name))
        except (OSError,StopIteration,ValueError):continue
    return result

def nonce_owners(nonce):
    # A multi-device engine may place workers outside this process tree (an MPI
    # pool is not a descendant), so a job stamps its children's environment and
    # claims every process carrying the stamp.
    if not nonce:return set()
    result=set()
    for environ in proc_files('/proc','environ'):
        try:
            if f'TILEMEGA_RUN_NONCE={nonce}\0'.encode() in environ.read_bytes():
                result.add(int(environ.parent.name))
        except OSError:continue
    return result

def numa_node_of(index):
    bus=smi('-i',str(index),'--query-gpu=pci.bus_id',
        '--format=csv,noheader').strip().lower()
    domain,rest=bus.split(':',1)
    node=Path('/sys/bus/pci/devices',f'{domain[-4:]}:{rest}','numa_node').read_text().strip()
    if not node.lstrip('-').isdigit() or int(node)<0:
        raise ValueError(f'GPU {index} has no NUMA affinity')
    return node

def external(row,own,*,allow_unresolved=False,nonce=None):
    own=nvml_owners(own)|nonce_owners(nonce)
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

def idle(row,policy,needs=0,nonce=None):
    occupied = (bool(row['owners']) if policy.get('strict_host_owners') else
                bool(external(row,descendants(os.getpid()),nonce=nonce)))
    return (not occupied and row['utilization_pct']<=policy['max_util_pct']
            and ('idle_power_w' not in policy or
                 row['power_w']<=idle_power(policy,row.get('gpu_index'))+policy['power_margin_w'])
            and row['hidden_mib']<=policy['max_hidden_mib'] and row['free_mib']>=needs
            and not throttle_invalid(row))

def session_members(sid):
    members=set()
    for path in proc_files('/proc','stat'):
        try:
            if int(path.read_text().rsplit(')',1)[1].split()[3])==sid:
                members.add(int(path.parent.name))
        except (OSError,ValueError,IndexError):pass
    return members

def process_start(pid, root='/proc'):
    try:return Path(root,str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError):return None

def host_proc():
    default = '/host/proc' if Path('/host/proc/self/status').is_file() else '/proc'
    root = Path(os.environ.get('TILEMEGA_HOST_PROC', default)).resolve()
    # Linux reserves this inode for the initial PID namespace.
    initial_pid_ns = os.readlink('/proc/self/ns/pid') == 'pid:[4026531836]'
    try:
        pids = next(line.split()[1:] for line in (root/'self/status').read_text().splitlines()
                    if line.startswith('NSpid:'))
        local_start = process_start(os.getpid())
        if (int(pids[-1]) == os.getpid() and local_start is not None
                and (initial_pid_ns or len(pids) > 1)
                and process_start(int(pids[0]), root) == local_start):
            return root, int(pids[0])
    except (OSError, ValueError, IndexError, StopIteration):
        pass
    raise RuntimeError('strict GPU ownership requires readable host /proc with host-to-container PIDs')

def strict_foreign(row, guard_pid, known):
    root, _ = host_proc()
    known.update(descendant_starts(guard_pid, root))
    # Use host PIDs throughout. A coincidentally equal local PID does not
    # establish ownership and all vanished or reused PIDs fail closed.
    return {pid for pid in row['owners']
            if pid not in known or process_start(pid, root) != known[pid]}

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
    p.add_argument('--needs-free-mib',type=int,default=12288);p.add_argument('--timeout-s',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER)
    p.add_argument('--heartbeat-timeout-s', type=float)
    a=p.parse_args()
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if not cmd:p.error('a child command is required')
    if a.heartbeat_timeout_s is not None and not 0 < a.heartbeat_timeout_s <= a.timeout_s:
        p.error('heartbeat timeout must be positive and no greater than the child timeout')
    index=os.environ.get('TILEMEGA_DEVICE_INDEX')
    group=os.environ.get('TILEMEGA_DEVICE_GROUP')
    policy=json.loads(a.policy.read_text())
    if group is not None:
        # Group mode (R5 B0): one unit owns every card it spans, so the whole
        # group is locked at once and any foreign process on any of them voids
        # the round.
        indices=group.split(',')
        if (not all(i.isdigit() for i in indices) or len(set(indices))!=len(indices) or
                os.environ.get('CUDA_VISIBLE_DEVICES')!=group):
            p.error('explicit matching TILEMEGA_DEVICE_GROUP and CUDA_VISIBLE_DEVICES are required')
    elif index is None or not index.isdigit() or os.environ.get('CUDA_VISIBLE_DEVICES')!=single_card_mask(policy,index):
        p.error('explicit matching TILEMEGA_DEVICE_INDEX and CUDA_VISIBLE_DEVICES are required')
    else:
        indices=[index]
    timing=os.environ.get('TILEMEGA_TIMING','1') != '0'
    for one in indices:require_allowed_gpu(policy,one,timing)
    nonce=str(uuid.uuid4())
    a.out.mkdir(parents=True,exist_ok=True)
    limited=os.environ.get('TILEMEGA_GPU_PID_MODE')=='namespace_limited'
    strict=policy.get('strict_host_owners',False)
    if strict and (limited or group is not None):p.error('strict host ownership requires single-card strict mode')
    host_known={}
    def sample(phase,**extra):
        rows=[verify_device(policy,gpu(one,strict=True) if strict else gpu(one)) for one in indices]
        for row in rows:
            if strict:
                row['foreign_or_unresolved_pids']=sorted(
                    strict_foreign(row,guard_host_pid,host_known) if phase!='preflight' else row['owners'])
            else:
                row['unresolved_host_pids']=[int(pid) for pid in row['owners'] if not Path('/proc',str(pid)).exists()]
            row['ownership_verified']=not limited
            with (a.out/'guard.jsonl').open('a') as f:f.write(json.dumps(dict(row,phase=phase,**extra))+'\n')
        return rows
    def finish(code,reason):
        acceptance='provisional_container_pid_namespace' if limited else 'strict'
        (a.out/'guard_result.json').write_text(json.dumps(dict(code=code,reason=reason,
            ownership_verified=not limited,acceptance=acceptance if code==0 else 'rejected',
            numa_concurrency_trial=os.environ.get('TILEMEGA_NUMA_CONCURRENCY_TRIAL')=='1'))+'\n')
        print(reason,flush=True);return code
    locks=[Path(os.environ['TILEMEGA_GPU_LOCK'])] if group is None else [
        Path(f'/tmp/tilemega_gpu{one}.lock') for one in sorted(indices,key=int)]
    for lock in locks:lock.parent.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        if strict:
            try:_,guard_host_pid=host_proc()
            except Exception as error:return finish(75,'preflight ownership unavailable: '+str(error))
        if needs_numa_lock(os.environ):
            if group is None:
                node=os.environ.get('TILEMEGA_NUMA_NODE')
                if node is None:p.error('timing jobs require TILEMEGA_NUMA_NODE')
                nodes=[node]
            else:
                nodes=sorted({numa_node_of(one) for one in indices},key=int)
            # Ascending order over both lock families keeps concurrent units
            # from taking them in opposite orders.
            for node in nodes:
                numa=stack.enter_context(Path(f'/tmp/tilemega_numa{node}.lock').open('a'))
                try:fcntl.flock(numa,fcntl.LOCK_EX|(fcntl.LOCK_NB if strict else 0))
                except BlockingIOError:return finish(75,'preflight NUMA lock occupied')
        for lock in locks:
            handle=stack.enter_context(lock.open('a'))
            try:fcntl.flock(handle,fcntl.LOCK_EX|(fcntl.LOCK_NB if strict else 0))
            except BlockingIOError:return finish(75,'preflight GPU lock occupied')
        previous=set();rows=[]
        for i in range(policy['samples']):
            try:current=sample('preflight',sample=i)
            except Exception as error:return finish(75,'preflight telemetry unavailable: '+str(error))
            # Do not start with a newly observed owner either; the two-sample
            # rule identifies interference during execution, not permission to start.
            if any(not idle(row,policy,a.needs_free_mib,nonce=nonce) for row in current):
                return finish(75,'preflight occupied')
            rows.extend(current)
            if i+1<policy['samples']:time.sleep(policy['interval_s'])
        baseline=max(r['hidden_mib'] for r in rows)
        thermal=rows[0].get('thermal_counters_us')
        def throttled(current):
            return any(throttle_invalid(row) or (strict and row['thermal_counters_us']!=thermal)
                       for row in current)
        if throttled(rows):return finish(75,'preflight thermal throttle')
        def checked_sample(phase, **extra):
            current = sample(phase, **extra)
            if (strict and policy.get('confirm_hidden_memory', False)
                    and not throttled(current)
                    and not any(row['foreign_or_unresolved_pids'] for row in current)
                    and any(row['hidden_mib'] > baseline + policy['max_hidden_mib'] for row in current)):
                # NVML fetches memory and owners separately; an owned worker's
                # exit between those calls can leave a one-snapshot mismatch.
                return sample(phase + '_confirm', **extra)
            return current
        env=dict(os.environ,TILEMEGA_GPU_LOCK_HELD='1',TILEMEGA_RUN_NONCE=nonce)
        if group is None:
            env['TILEMEGA_CUDA_VISIBLE_DEVICES'] = single_card_mask(policy, index)
        if policy.get('keep_host_pid_visible', False):
            env['TILEMEGA_KEEP_HOST_PID_VISIBLE'] = '1'
        heartbeat = a.out.resolve()/'heartbeat.jsonl'
        if a.heartbeat_timeout_s is not None:
            env['TILEMEGA_HEARTBEAT'] = str(heartbeat)
        if strict:env['TILEMEGA_STRICT_HOST_GUARD_PID']=str(guard_host_pid)
        process=subprocess.Popen(cmd,env=env,start_new_session=True)
        begin=time.monotonic();hidden=0;known={process.pid:process_start(process.pid)};own={process.pid}
        (a.out/'pgid').write_text(str(process.pid)+'\n')
        (a.out/'pgid_start').write_text(str(known[process.pid])+'\n')
        last_progress, heartbeat_size = begin, 0
        while True:
            try:code=process.wait(timeout=policy['interval_s']);break
            except subprocess.TimeoutExpired:pass
            for pid in descendants(process.pid):known[pid]=process_start(pid)
            own={pid for pid,started in known.items() if started is not None and process_start(pid)==started}
            try:current=checked_sample('running')
            except Exception as error:
                stop(process,known);return finish(75,'running telemetry unavailable: '+str(error))
            if throttled(current):
                stop(process,known);return finish(75,'running throttle')
            foreign=set()
            for row in current:
                if strict:foreign.update(row['foreign_or_unresolved_pids'])
                else:foreign|=external(row,own|{os.getpid()},allow_unresolved=limited,nonce=nonce)
            hidden=hidden+1 if any(row['hidden_mib']>baseline+policy['max_hidden_mib']
                                   for row in current) else 0
            if (foreign if strict else foreign & previous) or hidden>=policy['hidden_violations']:
                stop(process,known);return finish(75,'running interference')
            previous=foreign
            now = time.monotonic()
            if a.heartbeat_timeout_s is not None:
                try:size = heartbeat.stat().st_size
                except FileNotFoundError:size = 0
                if size > heartbeat_size:
                    heartbeat_size, last_progress = size, now
                if now-last_progress > a.heartbeat_timeout_s:
                    stop(process,known);return finish(124,'benchmark heartbeat timed out')
            if now-begin>a.timeout_s:
                stop(process,known);return finish(124,'guard timeout')
        # Clean retained workers after normal/error parent exit as well.
        stop(process,known)
        try:current=checked_sample('after',exit_code=code)
        except Exception as error:return finish(75,'exit telemetry unavailable: '+str(error))
        if throttled(current):return finish(75,'exit throttle')
        foreign=set()
        for row in current:
            if strict:foreign.update(row['foreign_or_unresolved_pids'])
            else:foreign|=external(row,own|{os.getpid()},nonce=nonce)
        if limited and foreign:return finish(75,'exit unresolved GPU owner')
        if (foreign if strict else foreign & previous) or any(row['hidden_mib']>baseline+policy['max_hidden_mib']
                                     for row in current) or (code and foreign):
            return finish(75,'exit interference')
        return finish(code,'child exit '+str(code))
if __name__=='__main__':raise SystemExit(main())
