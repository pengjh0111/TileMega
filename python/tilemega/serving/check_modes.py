"""Check four serving execution arms against the same real-weight prompts."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import torch
from .engine import ServingEngine
from .measure import _exclusive, _preflight_external_memory


def main() -> None:
    p=argparse.ArgumentParser()
    for name in ('model','prefill-so','decode-so','prompt-ids','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--steps',type=int,default=1024)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.cuda.init()
    guard_allocation=torch.empty(1,device='cuda')
    _preflight_external_memory(a.out)
    prompts=torch.tensor(json.loads(a.prompt_ids.read_text())[:a.batch],dtype=torch.int32)
    metadata=json.loads(Path(str(a.decode_so)+".plan.json").read_text())
    arms=(('L1_separate','L1',False,None),('L2_separate','L2',False,None),
          ('L2_loop','L2',True,None),('L2_loop_no_phase','L2',True,'0'))
    if metadata.get("pg")!="pages":
        arms=(('L1_separate','L1',False,None),('L2_separate','L2',False,None),
              ('L1_loop','L1',True,None),('L1_loop_no_phase','L1',True,'0'))
    results={};mismatches={}
    for label,mode,loop,mask in arms:
        if not _exclusive(a.out/'guard.jsonl',label+'-before',True):
            raise SystemExit(75)
        previous=os.environ.get('TILEMEGA_KPHASE_MASK')
        if mask is not None:os.environ['TILEMEGA_KPHASE_MASK']=mask
        try:
            with ServingEngine(a.model,a.prefill_so,a.decode_so,a.batch,
                               max_new_tokens=a.steps,mode=mode,decode_loop=loop,prefill_mode="L1") as engine:
                try:
                    generation=engine.generate(prompts,a.steps)
                    if loop and not generation.decode_loop_used:
                        raise RuntimeError("correctness arm did not use its requested loop")
                    tokens=generation.tokens
                except BaseException:
                    (a.out/f'watchdog_{label}.json').write_text(json.dumps(engine.decode.watchdog(),indent=2)+'\n')
                    raise
                results[label]=tokens
                (a.out/f'tokens_{label}.json').write_text(json.dumps(tokens.tolist(),separators=(',',':'))+'\n')
        finally:
            if previous is None:os.environ.pop('TILEMEGA_KPHASE_MASK',None)
            else:os.environ['TILEMEGA_KPHASE_MASK']=previous
        if not _exclusive(a.out/'guard.jsonl',label+'-after',False):
            raise SystemExit(75)
        mismatches[label]=int((results[label]!=results[arms[0][0]]).sum().item())
    report=dict(batch=a.batch,steps=a.steps,mismatches=mismatches,
                **{'pass':all(n==0 for n in mismatches.values())})
    (a.out/'mode_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    if not report['pass']:raise SystemExit(1)
    del guard_allocation

if __name__=='__main__':main()
