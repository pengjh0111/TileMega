#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare guarded Hopper diagnostics and compare five fresh processes per GPU."""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)


def prepare(args):
    root=args.root.absolute();out=args.out.absolute();receipt=read(args.receipt)
    if not receipt.get('build_completed') or not receipt.get('snapshot_source_unchanged'):
        raise ValueError('completed immutable CLI receipt required')
    if out.exists():raise FileExistsError('new diagnostic run directory required')
    source=Path(receipt['source_dir']);files=['lib/Target/HopperCalibration.cu','include/tilemega/Target/HopperCalibration.h','tools/commands/calibrate.cpp']
    source_sha={name:sha(source/name) for name in files}
    if sha(receipt['cli'])!=receipt['cli_sha256']:raise ValueError('CLI differs from completed receipt')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'driver.py');shutil.copy2(args.base,out/'base.json')
    by_gpu={}
    for gpu in args.gpus:
        node=0 if gpu<4 else 1
        config=dict(cli=receipt['cli'],cli_sha256=receipt['cli_sha256'],source=str(source),source_head=receipt['snapshot_head'],source_sha256=source_sha,base=str(out/'base.json'),base_sha256=sha(out/'base.json'),gpu=gpu,repeats=args.repeats,run_root=str(root),scope='diagnostic calibration; not model end-to-end acceptance',pid_mode=args.pid_mode)
        path=out/f'gpu{gpu}.json';write(path,config);jobs=[]
        for fresh in range(5):
            j=dict(name=f'b2-hopper-{out.name}-gpu{gpu}-f{fresh}',gpu=True,timing=True,cwd=str(root),needs_free_mib=4096,timeout_s=600,priority=args.priority,env={'CUDA_VISIBLE_DEVICES':str(gpu),'TILEMEGA_DEVICE_INDEX':str(gpu),'TILEMEGA_NUMA_NODE':str(node),'TILEMEGA_GPU_PID_MODE':args.pid_mode},command=['numactl',f'--cpunodebind={node}',sys.executable,str(out/'driver.py'),'run','--config',str(path),'--fresh',str(fresh),'--attempt','{attempt_dir}'])
            if gpu==args.gpus[-1] and args.second_after_queue:j['after_any']=[x['name'] for x in read(args.second_after_queue)]
            jobs.append(j)
        write(out/f'queue-gpu{gpu}.json',jobs);by_gpu[str(gpu)]=str(out/f'queue-gpu{gpu}.json')
    write(out/'preparation.json',dict(source_receipt=str(args.receipt.absolute()),queues=by_gpu,driver_sha256=sha(out/'driver.py'),enqueued=False,independent_fragment=True,numa_memory_binding='CPU affinity only; container membind denied',fresh_processes_per_gpu=5))
    print(json.dumps(by_gpu))


def run(args):
    c=read(args.config);gpu=str(c['gpu'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or os.environ.get('TILEMEGA_DEVICE_INDEX')!=gpu or os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1':raise ValueError('explicit GPU variables and guarded lease required')
    if sha(c['cli'])!=c['cli_sha256'] or sha(c['base'])!=c['base_sha256']:raise ValueError('CLI or independent frozen base changed')
    if any(sha(Path(c['source'])/name)!=wanted for name,wanted in c['source_sha256'].items()):raise ValueError('diagnostic source identity changed')
    out=args.attempt.absolute();out.mkdir(parents=True,exist_ok=True);fragment=out/'fragment.json'
    cmd=[c['cli'],'calibrate','--hopper-diagnostics','--device','0','--base',c['base'],'--repeats',str(c['repeats']),'--out',str(fragment)]
    begin=time.time()
    with (out/'calibrate.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=c['run_root'],stdout=log,stderr=subprocess.STDOUT);code=p.wait()
    value=dict(config=c,config_sha256=sha(args.config),fresh_index=args.fresh,child_pid=p.pid,supervisor_pid=os.getpid(),command=cmd,started=begin,ended=time.time(),exit_code=code,fragment=str(fragment),raw=str(fragment)+'.tsv',passed=False)
    failure=None
    try:
        if code==0:
            report=read(fragment);records=report['records'];raw=Path(str(fragment)+'.tsv')
            if len(records)!=18 or not all(math.isfinite(x['median']) for x in records):raise ValueError('missing or nonfinite diagnostic records')
            with raw.open() as f:
                if any(row['checksum_pass']!='1' for row in csv.DictReader(f,delimiter='\t')):raise ValueError('transport or cluster correctness failed')
            value.update(passed=True,fragment_sha256=sha(fragment),raw_sha256=sha(raw),records=records)
    except (ValueError,OSError,KeyError) as error:
        value['error']=str(error);failure=error
    write(out/'process.json',value)
    if failure is not None:raise failure
    if code:raise ValueError('official Hopper diagnostic command failed; see calibrate.log')


def analyze_receipts(paths):
    groups=collections.defaultdict(list);rejected=[]
    for path in paths:
        p=Path(path)
        try:
            v=read(p);guard=read(p.parent/'guard_result.json')
            if not v.get('passed') or v['exit_code']!=0 or guard['code']!=0:raise ValueError('failed command or guard')
            if sha(v['fragment'])!=v['fragment_sha256'] or sha(v['raw'])!=v['raw_sha256']:raise ValueError('measurement evidence changed')
            if read(v['fragment'])['records']!=v['records']:raise ValueError('receipt coefficients differ from raw fragment')
            with Path(v['raw']).open() as stream:
                raw_rows=list(csv.DictReader(stream,delimiter='\t'))
            if not raw_rows or any(x.get('checksum_pass')!='1' for x in raw_rows):raise ValueError('raw correctness checks absent or failed')
            v['ownership_verified']=guard.get('ownership_verified',False);v['acceptance']=guard.get('acceptance','unknown')
            groups[str(v['config']['gpu'])].append(v)
        except (OSError,ValueError,KeyError) as e:rejected.append({'path':str(p),'reason':str(e)})
    result=dict(evidence='verified_analysis_of_raw_receipts',passed=False,rejected=rejected,processes_by_gpu={g:len(a) for g,a in groups.items()},ownership_verified=all(v['ownership_verified'] for a in groups.values() for v in a) if groups else False,rows=[],gate='five fresh processes per GPU, matching binary/source/base, all coefficient differences <=3%; no threshold changes')
    if rejected or len(groups)!=2 or any(len(a)!=5 for a in groups.values()):result['reason']='requires exactly two GPUs and five accepted fresh processes each';return result
    identities={(v['config']['cli_sha256'],v['config']['source_head'],v['config']['base_sha256']) for a in groups.values() for v in a}
    if len(identities)!=1:result['reason']='cross-card binary/source/base identity mismatch';return result
    for a in groups.values():
        if len({v['child_pid'] for v in a})!=5 or {v['fresh_index'] for v in a}!=set(range(5)):result['reason']='distinct child PIDs and fresh indices 0..4 required';return result
    def medians(a):
        values=collections.defaultdict(list)
        for v in a:
            seen=set()
            for row in v['records']:
                key=(row['name'],row['cluster_dim'],row['peer_rank'],row['unit'])
                if key in seen:raise ValueError('duplicate coefficient in one process')
                seen.add(key);values[key].append(row['median'])
        if len(values)!=18 or any(len(x)!=5 for x in values.values()):raise ValueError('all 18 coefficient keys must exist in all five processes')
        return {k:statistics.median(x) for k,x in values.items()}
    cards=sorted(groups);a,b=(medians(groups[g]) for g in cards)
    if a.keys()!=b.keys():result['reason']='cross-card coefficient keys differ';return result
    for key in sorted(a):
        relative=(b[key]/a[key]-1) if a[key]>0 and b[key]>0 else None
        result['rows'].append(dict(name=key[0],cluster_dim=key[1],peer_rank=key[2],unit=key[3],gpus=cards,medians=[a[key],b[key]],relative_difference=relative,passed=relative is not None and abs(relative)<=.03+1e-12))
    result['acceptance']='verified_ownership' if result['ownership_verified'] else 'provisional_container_pid_namespace'
    result['passed']=all(x['passed'] for x in result['rows']);result['reason']='cross-card coefficients pass' if result['passed'] else 'cross-card coefficient disagreement or nonpositive differential retained'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare');a.add_argument('--root',type=Path,default=Path('/root/TileMega'));a.add_argument('--receipt',type=Path,required=True);a.add_argument('--base',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--gpus',type=int,nargs=2,default=[1,2]);a.add_argument('--repeats',type=int,default=21);a.add_argument('--priority',type=float,default=9);a.add_argument('--pid-mode',choices=('strict','namespace_limited'),default='strict');a.add_argument('--second-after-queue',type=Path)
    a=sub.add_parser('run');a.add_argument('--config',type=Path,required=True);a.add_argument('--fresh',type=int,required=True);a.add_argument('--attempt',type=Path,required=True)
    a=sub.add_parser('analyze');a.add_argument('receipts',type=Path,nargs='+');a.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='run':run(args)
    else:
        v=analyze_receipts(args.receipts);write(args.out,v);print(json.dumps(v));return 0 if v['passed'] else 3
    return 0
if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,KeyError) as e:print(str(e),file=sys.stderr);sys.exit(3)
