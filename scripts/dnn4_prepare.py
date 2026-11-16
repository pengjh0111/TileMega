#!/usr/bin/env python3
"""Prepare explicit DNN4 correctness/sentinel cells with deterministic seed-v2."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('cells',nargs='+',help='model:batch; no measurements are run')
    args=p.parse_args();args.compiler=args.compiler.resolve();args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=False);rows=[]
    for cell in args.cells:
        model,batch=cell.split(':');batch=int(batch)
        if model not in ('resnet18','resnet50','vgg16','bert') or batch not in (1,16):
            raise ValueError('before Phase Z prepare only B1/B16 correctness/sentinel cells')
        directory=args.out/(model+'_b'+str(batch));directory.mkdir()
        bridge=ROOT/f'runs/dnn3/takeover_20261107/plan_gate/{model}_b{batch}.bridge.json'
        plan=directory/'plan.json';library=directory/'model.so'
        common=['--serving','forward','--frontend','dnn','--batch',str(batch),'--abi','f3']
        commands=[
            [str(args.compiler),'compile',str(bridge),str(directory/'unused.cu'),*common,'--dump-plan',str(plan)],
            [sys.executable,str(ROOT/'scripts/dnn4_seed.py'),'--binary',str(args.compiler),'--plan',str(plan),'--bridge',str(bridge),'--out',str(directory/'seed')],
            [str(args.compiler),'compile',str(bridge),str(library),'--emit','serving',*common,
             '--tmexec-config',str(directory/'seed/seed.tmexec.json'),'--runtime-target',str(ROOT/'configs/targets/sm_90_h800.json')]]
        for index,command in enumerate(commands):
            with (directory/f'step{index}.log').open('w') as stream:
                done=subprocess.run(['taskset','-c','136-143',*command],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT)
            if done.returncode:
                rows.append(dict(cell=cell,status='failed',step=index));break
        else:rows.append(dict(cell=cell,status='compiled',library=str(library)))
        (args.out/'result.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
        print(json.dumps(rows[-1]),flush=True)

if __name__=='__main__':main()
