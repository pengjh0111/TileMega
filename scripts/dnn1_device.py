"""DNN1 user policy: exclude GPU6; gate only on processes and extra memory."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
POLICY='user-2026-11-04-processes-and-memory-only'


def gpu(index=None):
    index=str(os.environ.get('TILEMEGA_DEVICE_INDEX','0') if index is None else index)
    if int(index) not in (0,1,2,3,4,5,7):
        raise ValueError('DNN1 excludes GPU6')
    def query(flag):
        return subprocess.check_output(['nvidia-smi','-i',index,flag,
            '--format=csv,noheader,nounits'],text=True,timeout=15)
    # Clocks are result metadata, never an availability criterion. No power,
    # temperature, utilization or throttle query is made by this provider.
    raw=query('--query-gpu=memory.used,memory.free,memory.total,clocks.current.sm,clocks.current.memory')
    used,free,total,sm,mem=map(float,raw.splitlines()[0].split(','))
    owners={}
    for line in query('--query-compute-apps=pid,used_memory').splitlines():
        if line.strip():
            pid,size=map(int,line.split(','));owners[pid]=size
    return dict(time=time.time(),gpu_index=int(index),used_mib=used,free_mib=free,total_mib=total,
        owners=owners,hidden_mib=max(0,used-sum(owners.values())),sm_clock_mhz=sm,
        memory_clock_mhz=mem,resource_policy=POLICY,power_monitoring=False)


def idle(row,policy,needs=0,nonce=None):
    return (row['gpu_index']!=6 and not row['owners'] and row['used_mib']==0
            and row['free_mib']>=needs)


def observe_one(path,label,index):
    import baseline_common as common
    row=gpu(index)
    pids=set(row['owners']);not_own=pids-common._own_pids()
    unresolved=sorted(pid for pid in not_own if not Path(f'/proc/{pid}').exists())
    limited=os.environ.get('TILEMEGA_GPU_PID_MODE')=='namespace_limited'
    foreign=sorted(not_own-set(unresolved)) if limited else sorted(not_own)
    sample=dict(time=row['time'],label=label,gpu_index=row['gpu_index'],
        telemetry={'memory.used':str(row['used_mib']),'clocks.sm':str(row['sm_clock_mhz']),
                   'clocks.mem':str(row['memory_clock_mhz'])},
        pids=sorted(pids),foreign_pids=foreign,hidden_mib=row['hidden_mib'],
        accepted=not foreign and row['hidden_mib']<=1024,unresolved_host_pids=unresolved,
        ownership_verified=not limited,acceptance='provisional_container_pid_namespace' if limited else 'strict',
        resource_policy=POLICY,power_monitoring=False)
    with Path(path).open('a') as output:output.write(json.dumps(sample)+'\n')
    return sample


def install():
    """Use the DNN provider within this process; shared files stay unchanged."""
    sys.path.insert(0,str(ROOT/'python'))
    import tilemega.bench.gpu_guard as guard
    guard.gpu=gpu
    guard.idle=idle
    if 'baseline_common' not in sys.modules:
        spec=importlib.util.spec_from_file_location('baseline_common',ROOT/'python/tilemega/serving/baseline_common.py')
        common=importlib.util.module_from_spec(spec);sys.modules['baseline_common']=common
        spec.loader.exec_module(common)
    common=sys.modules['baseline_common'];common._observe_one=observe_one
    import tilemega.serving as serving
    sys.modules['tilemega.serving.baseline_common']=common
    serving.baseline_common=common
    os.environ['TILEMEGA_DNN1_RESOURCE_POLICY']=POLICY
