#!/usr/bin/env python3
"""Fresh-process token comparison; timing here is never a performance result."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def child(case, output):
    import torch
    from tilemega.serving.engine import ServingEngine
    from tilemega.serving.measure import _exclusive
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.cuda.init()
    context_probe = torch.empty(1,device="cuda")
    prompts = torch.tensor(json.loads(Path(case['prompt_ids']).read_text())[:case['batch']], dtype=torch.int32)
    common = dict(model_dir=case['model'], batch=case['batch'])
    records = []
    reference = None
    with open(os.environ.get('TILEMEGA_GPU_LOCK', '/root/r10_work/serving_gpu.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not _exclusive(output/'guard.jsonl', 'before', True):
            raise RuntimeError('GPU guard rejected protocol check')
        for label, prefill, decode, modes in [
            ('reference', case['reference_prefill'], case['reference_decode'], ['L1']),
            ('candidate', case['prefill'], case['decode'], ['L1', 'L2', 'L1', 'L2']),
        ]:
            with ServingEngine(prefill_so=prefill, decode_so=decode, mode='L1', **common) as engine:
                for mode in modes:
                    engine.prefill_mode = engine.decode_mode = {'L1':1, 'L2':2}[mode]
                    print(label,mode,"begin",flush=True)
                    tokens = engine.generate(prompts, case.get('steps',64)).tokens
                    if reference is None:
                        reference = tokens.clone()
                    mismatches = int((tokens != reference).sum().item())
                    values = tokens.tolist()
                    records.append(dict(arm=label, mode=mode, mismatches=mismatches,
                        token_sha256=hashlib.sha256(json.dumps(values).encode()).hexdigest()))
                    if mismatches:
                        (output/'mismatch_tokens.json').write_text(json.dumps(values))
            torch.cuda.empty_cache()
        if not _exclusive(output/'guard.jsonl', 'after', False):
            raise RuntimeError('GPU guard rejected completed protocol check')
    result = dict(pid=os.getpid(), case=case, records=records,
                  passed=all(r['mismatches']==0 for r in records), performance_measurement=False)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['passed'] else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--processes',type=int,default=50)
    parser.add_argument('--child',type=int)
    a=parser.parse_args();cases=json.loads(a.cases.read_text())
    if a.child is not None:
        return child(cases[a.child],a.out)
    a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index,case in enumerate(cases):
        for run in range(a.processes):
            output=a.out/f"{index}_{run:03d}";output.mkdir(exist_ok=True)
            command=[sys.executable,__file__,'--cases',str(a.cases.resolve()),'--child',str(index),'--out',str(output)]
            started=time.monotonic()
            with (output/'run.log').open('w') as log:
                try:
                    status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
                except subprocess.TimeoutExpired:
                    status=124
            rows.append(dict(case=index,run=run,exit_code=status,seconds=time.monotonic()-started))
            (a.out/'processes.json').write_text(json.dumps(rows,indent=2)+'\n')
            if status:
                return status
    (a.out/'summary.json').write_text(json.dumps(dict(cases=cases,processes_per_case=a.processes,
        passed=len(rows),failed=0,complete=True),indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
