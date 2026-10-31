#!/usr/bin/env python3
"""Capture one actual greedy request's native stage stamps; never E2E evidence."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path

from tilemega.build.identity import sha, verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('model','prefill-so','decode-so','prompt-ids','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--variant',required=True)
    a=p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1':p.error('requires mx_gpu_job.sh')
    a.out.mkdir(parents=True,exist_ok=False)
    identity=verify(a.decode_so);m=identity['plan']
    if not identity['trace']or m['diag']or m['watchdog']:p.error('requires trace1/DIAG0/WD0')
    os.environ['TILEMEGA_TRACE_LAUNCHES']='577'
    import torch
    from tilemega.serving.engine import ServingEngine
    prompt=json.loads(a.prompt_ids.read_text())[:a.batch]
    with ServingEngine(a.model,a.prefill_so,a.decode_so,a.batch,prompt_len=64,
            max_new_tokens=578,mode='L1',decode_loop=True,prefill_mode='L1',step_events=False)as engine:
        result=engine.generate(torch.tensor(prompt,dtype=torch.int32),578)
        torch.cuda.synchronize()
        dump=engine.decode_lib.lib.tm_plan_dump_serving_trace
        dump.argtypes=[C.c_void_p,C.c_char_p];dump.restype=C.c_int
        if dump(engine.decode.handle,os.fsencode(a.out.resolve())):raise RuntimeError('trace read failed')
        execution=engine.decode.execution_identity(1,loop=True)
        tokens=result.tokens.tolist()
    (a.out/'generated.json').write_text(json.dumps(tokens)+'\n')
    scope=dict(model=str(a.model),batch=a.batch,variant=a.variant,executor='L1_loop',contract='bf16',
        trace=1,diag=0,watchdog=0,decode_steps=577,discard_steps=65,accepted_steps=512,
        PID=os.getpid(),timing_eligible=False,execution_identity=execution,
        decode_binary_sha256=sha(a.decode_so),prefill_binary_sha256=sha(a.prefill_so),
        source_head=identity['source']['head'],source_sha256=identity['source']['source_digest'])
    (a.out/'trace_scope.json').write_text(json.dumps(scope,indent=2)+'\n')
    print('ACTUAL_GREEDY_TRACE_CAPTURED',a.variant,a.batch,flush=True)


if __name__=='__main__':main()
