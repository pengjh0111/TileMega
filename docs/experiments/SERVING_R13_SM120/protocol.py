#!/usr/bin/env python3
"""R13 fresh-process protocol, with fixed L1 prefill and unambiguous outcomes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from bootstrap import FRAME, write, sha


def child(case, output):
    import torch
    from tilemega.serving.engine import ServingEngine
    from tilemega.serving.measure import _exclusive, _preflight_external_memory
    output.mkdir(parents=True,exist_ok=True)
    prompts=torch.tensor(json.loads(Path(case['prompt_ids']).read_text())[:case['batch']],dtype=torch.int32)
    records=[]
    reference=None
    with open(os.environ['TILEMEGA_GPU_LOCK'],'a') as handle:
        if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1':
            fcntl.flock(handle,fcntl.LOCK_EX)
        torch.cuda.set_device(0)
        probe=torch.empty(1,device='cuda')
        if not _exclusive(output/'guard.jsonl','before',True):
            return 75
        _preflight_external_memory(output)
        for arm in case['arms']:
            for phase in ('prefill','decode'):
                if sha(arm[phase])!=arm['sha256'][phase]:
                    raise RuntimeError('protocol binary changed: '+arm[phase])
            old={key:os.environ.get(key) for key in arm.get('env',{})}
            os.environ.update(arm.get('env',{}))
            try:
                with ServingEngine(case['model'],arm['prefill'],arm['decode'],case['batch'],
                                   max_new_tokens=case['steps'],mode=arm['mode'],
                                   prefill_mode='L1',decode_loop=arm['decode_loop'],step_events=False) as engine:
                    generated=engine.generate(prompts,case['steps'])
                    if arm['decode_loop'] and not generated.decode_loop_used:
                        raise RuntimeError('requested protocol loop was not used')
                    if reference is None:
                        reference=generated.tokens.clone()
                    mismatch=int((generated.tokens!=reference).sum().item())
                    records.append(dict(label=arm['label'],mode=arm['mode'],
                          prefill_mode='L1',step_events=0,decode_loop=arm['decode_loop'],
                          decode_loop_used=generated.decode_loop_used,mismatches=mismatch))
                    if mismatch:
                        write(output/(arm['label']+'_mismatch_tokens.json'),generated.tokens.tolist())
            finally:
                for key,value in old.items():
                    if value is None:os.environ.pop(key,None)
                    else:os.environ[key]=value
            torch.cuda.empty_cache()
        if not _exclusive(output/'guard.jsonl','after',False):
            return 75
    passed=all(row['mismatches']==0 for row in records)
    write(output/'result.json',dict(pid=os.getpid(),case=case,records=records,passed=passed,
          execution_complete=True,performance_measurement=False))
    del probe
    return 0 if passed else 1


def aggregate(rows, requested=50):
    completed=[row for row in rows if row.get('execution_complete')]
    pids=[row['pid'] for row in completed]
    fresh=len(pids)==len(set(pids))
    return dict(requested=requested,attempted=len(rows),completed=len(completed),
                unique_pids=len(set(pids)),fresh_pids=fresh,
                passed=sum(row.get('passed',False) for row in completed),
                failed=sum(not row.get('passed',False) for row in completed),
                complete=len(completed)==requested and fresh,
                pass_rate=sum(row.get('passed',False) for row in completed)/requested)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args()
    case=json.loads(args.case.read_text())
    if args.child:
        return child(case,args.out)
    if (args.out/'processes.json').exists():
        previous=args.out.with_name(args.out.name+'_previous_'+str(time.time_ns()))
        args.out.rename(previous)
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    sys.path.insert(0,str(FRAME))
    from gpu_guard import stop
    for number in range(50):
        output=args.out/f'{number:03d}'
        output.mkdir(exist_ok=True)
        command=[sys.executable,__file__,'--case',str(args.case.resolve()),'--out',str(output),'--child']
        start=time.monotonic()
        with (output/'run.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                stop(process)
                code=124
        result=output/'result.json'
        row=dict(run=number,exit_code=code,seconds=time.monotonic()-start,execution_complete=False)
        if result.exists():
            row.update(json.loads(result.read_text()))
        rows.append(row)
        write(args.out/'processes.json',rows)
        summary=aggregate(rows)
        write(args.out/'summary.json',summary)
        if code==75:
            return 75
        # Numerical mismatches complete the registered 50; crashes/hangs do not.
        if not row['execution_complete']:
            return code or 3
        from device_health import check
        check()
    print(json.dumps(summary),flush=True)
    return 0 if summary['complete'] else 3


if __name__=='__main__':
    raise SystemExit(main())
