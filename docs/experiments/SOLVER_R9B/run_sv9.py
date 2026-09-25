#!/usr/bin/env python3
"""Build the two archived single points and four legacy trace controls.

All GPU execution shares the R9b control lock. No outer search is launched.
Each build records its exact command and a hard NEED_MIB disk check.
"""
import argparse,fcntl,json,os,pathlib,shutil,sys
E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2];OLD=E.parent/'SOLVER_V2'
sys.path.insert(0,str(OLD))
from measure import run,parse
from gpu_admission import wait_for_device
from run_controls import audit
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--only",choices=["early","trace"]);args=parser.parse_args()
    jobs=[]
    for seq in (1,4):jobs.append((f'llama_s{seq}',E/'early_resident'/f'llama_s{seq}.cu',E/'early_resident'/f'llama_s{seq}.measurement',False))
    for model in ('llama','qwen3'):
        for seq in (1,64):
            cell=f'{model}_s{seq}';jobs.append((cell,OLD/'legacy_r8_domain'/cell/'selected.cu',E/'trace'/cell,True))
    if args.only:jobs=[job for job in jobs if job[3]==(args.only=="trace")]
    env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
    for cell,source,out,trace in jobs:
        out.mkdir(parents=True,exist_ok=True);free=shutil.disk_usage(out).free//2**20;need=8192
        (out/'disk.json').write_text(json.dumps(dict(NEED_MIB=need,FREE_MIB=free))+'\n')
        if free<need:raise RuntimeError('insufficient compile disk')
        original=OLD/'legacy_r8_domain'/cell/'selected.cu.measurement'
        fixture=pathlib.Path(json.loads((original/'process_00.command.json').read_text())['command'][1])
        binary=out/'kernel';cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-arch=sm_89','-lineinfo','-Xptxas=-v','-DTILEMEGA_MIDPOINT_REFINE=0']
        if trace:cmd+=['-DTILEMEGA_TRACE_V2=1']
        for sub in ('include','third_party/cutlass/include','third_party/cutlass/tools/util/include','third_party/cutlass/test'):cmd+=['-I'+str(ROOT/sub)]
        cmd += [source,ROOT/'build-portable/libtilemega.a','-L/usr/local/cuda/lib64','-lcudart','-o',binary]
        if run(cmd,out/'build.log',env):raise RuntimeError('SV9 build failed '+cell)
        audit(binary,out)
        runenv=dict(env,TILEMEGA_WARMUP='1' if trace else '2',TILEMEGA_REPEAT='1' if trace else '10')
        if trace:runenv['TILEMEGA_TRACE_V2_OUT']=str(out/'dump')
        with open('/tmp/tilemega-r9-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for i in range(1 if trace else 10):
                wait_for_device(fixture,out/'gpu_admission.jsonl');log=out/f'process_{i:02}.log'
                code=run([binary,fixture],log,runenv,timeout=600);good,times=parse(log.read_text())
                print(f'SV9 cell={cell} trace={int(trace)} round={i} internal={int(good)} exit={code}',flush=True)
                if not good:raise RuntimeError('internal equality failure '+str(log))
        (out/'status.json').write_text(json.dumps(dict(state='complete',processes=1 if trace else 10,source=str(source),fixture=str(fixture)))+'\n')
if __name__=='__main__':main()
