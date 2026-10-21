#!/usr/bin/env python3
"""Prepare indivisible guarded development pairs; retries repeat both arms."""
import argparse
from pathlib import Path
from sm90d1_common import ROOT,PYTHON,read,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,required=True,choices=(0,1,2,3,4,7))
    p.add_argument('--name',required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--baseline-tag',default='reference')
    p.add_argument('--tag',required=True)
    p.add_argument('--rounds',type=int,required=True)
    p.add_argument('--after',nargs='*',default=[])
    p.add_argument('--queue-dir',type=Path,required=True)
    a=p.parse_args();node=0 if a.gpu<4 else 1
    policy=ROOT/'configs/bench/sm90r2_guard_policy.json'
    uuid=read(policy)['idle_power_mapping_provenance']['uuid_by_gpu'][str(a.gpu)]
    pci={0:'0f',1:'10',2:'41',3:'44',4:'86',7:'b7'}[a.gpu]
    previous=list(a.after);jobs=[]
    for fresh in range(a.rounds):
        name=f'{a.name}-pair-f{fresh}'
        command=['numactl',f'--cpunodebind={node}',PYTHON,
            str(ROOT/'scripts/run_r16_with_device_identity.py'),'--guard-policy',str(policy),
            '--out','{attempt_dir}/pair','--expected-uuid',uuid,
            '--expected-pci',f'0000:{pci}:00.0','--numa-node',str(node),
            '--allow-no-measurements','--',PYTHON,str(ROOT/'scripts/run_sm90r2_pair.py'),
            '--config',str(a.config.resolve()),'--baseline-tag',a.baseline_tag,'--tag',a.tag,
            '--round',str(fresh),'--out','{attempt_dir}/pair']
        jobs.append(dict(name=name,gpu=True,timing=True,priority=10,timeout_s=3600,
            needs_free_mib=32768,cwd=str(ROOT),after=previous,
            env=dict(CUDA_VISIBLE_DEVICES=str(a.gpu),TILEMEGA_DEVICE_INDEX=str(a.gpu),
                TILEMEGA_NUMA_NODE=str(node),TILEMEGA_GPU_PID_MODE='namespace_limited',
                PYTHONPATH=str(ROOT/'python'),OMP_NUM_THREADS='8'),command=command))
        previous=[name]
    write(a.queue_dir/f'queue_{a.name}.json',jobs)


if __name__=='__main__':
    main()
