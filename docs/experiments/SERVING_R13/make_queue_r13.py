#!/usr/bin/env python3
"""Prepare every Phase A job and arm before the scheduler starts."""
import argparse,copy,json,math,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PY='/root/venvs/tilemega-torch213-cu126/bin/python'
LOCK='/root/r13_work/gpu.lock'
BASE='48c013d1863d5a5af249401be217d4aa4c15d710'
CELLS=('llama_B1','qwen3_B1','llama_B16','qwen3_B16')

def main():
    p=argparse.ArgumentParser();p.add_argument('--diag',type=Path,required=True)
    p.add_argument('--queue-dir',type=Path,required=True);a=p.parse_args()
    arms=json.loads((HERE/'arms.json').read_text());jobs=[]
    for cell in CELLS:
        model,b=cell.split('_B');batch=int(b);rows={r['label']:r for r in arms[cell]}
        cache=json.loads((ROOT/f'runs/r12b-{model}/cache.json').read_text())
        exports={phase:Path.home()/'.cache/tilemega/exports'/next(
            r['key'] for r in cache if r.get('layer')=='export' and r.get('phase')==phase)/'bridge.json'
            for phase in ('prefill','decode')}
        def add(label,phase,source,projection='identity'):
            so=Path(rows[source][phase]);folder=HERE/'raw/A0'/cell/label
            pages=projection=='paged'
            overrides=dict(watchdog=0,handoff='off',pg='pages' if pages else 'l2',
                           weight_layout='tiled' if pages else 'row',sync='calibrated',
                           lookahead_bytes=0,pdl='auto' if pages else 'off')
            if pages:overrides['page_bytes']=16384
            jobs.append(dict(cell=cell,label=label,phase=phase,batch=batch,
                manifest=str(so)+'.plan.json',classes=str(so)+'.classes.tsv',
                export=str(exports[phase]),projection=projection,overrides=overrides,
                out=str(folder)))
        add('N-R12b-noWD','decode','R12bN_L1')
        if cell!='qwen3_B16':add('N-R12bh-noWD','decode','R12bN_L1','head')
        add('P-R12bN-noWD','decode','R12bN_L1','paged')
        add('PF-R12b-noWD','prefill','R12bN_L1')
        add('PF-R10-noWD','prefill','R10C')
    missing=[]
    for job in jobs:
        for key in ('manifest','classes','export'):
            if not Path(job[key]).is_file():missing.append({'cell':job['cell'],'label':job['label'],'key':key,'path':job[key]})
    (HERE/'jobs_a.json').write_text(json.dumps(jobs,indent=2)+'\n')
    (HERE/'input_check.json').write_text(json.dumps({'jobs':len(jobs),'missing':missing},indent=2)+'\n')
    script=a.diag/'docs/experiments/SERVING_R13'
    binary=a.diag/'build-phase13/tools/tilemega';steps=[]
    env=dict(PYTHONPATH=str(a.diag/'python'),TILEMEGA_BIN=str(binary),
             TILEMEGA_R13_ROOT=str(ROOT),TILEMEGA_R13_DATA=str(HERE),
             TILEMEGA_R13_COMPILER_COMMIT=BASE,TILEMEGA_GPU_LOCK=LOCK)
    def add(name,command,gpu=False,priority=50,after=(),after_any=(),needs=12288,timeout=14400):
        if not gpu:command=['flock',LOCK,'env','TILEMEGA_GPU_LOCK_HELD=1']+command
        steps.append(dict(name=name,command=command,cwd=str(a.diag),env=env,
            gpu=gpu,priority=priority,after=list(after),after_any=list(after_any),
            timeout_s=timeout,needs_free_mib=needs,out=str(HERE/'raw'/name)))
    configure=['cmake','-S',str(a.diag),'-B',str(a.diag/'build-phase13'),
               '-DCMAKE_BUILD_TYPE=Release','-DMLIR_DIR=/root/toolchains/mlir-23a60f15/lib/cmake/mlir',
               '-DLLVM_DIR=/root/toolchains/mlir-23a60f15/lib/cmake/llvm',
               '-DTILEMEGA_ISL_BUILD_DIR=/root/TileMega/build-isl',
               '-DTILEMEGA_POLYLIB_BUILD_DIR=/root/TileMega/build-polylib',
               '-DTILEMEGA_BARVINOK_BUILD_DIR=/root/TileMega/build-barvinok',
               '-DTILEMEGA_ISL_GENERATED_INCLUDE_DIR=/root/TileMega/build-isl/include',
               '-DTILEMEGA_ISL_LIBRARY=/root/TileMega/build-isl/.libs/libisl.a']
    import shlex
    shell=shlex.join(configure)+' && '+shlex.join(['cmake','--build',str(a.diag/'build-phase13'),'--target','tilemega','-j','6'])+' && '+shlex.join(['python3',str(a.diag/'python/tilemega/fingerprint.py'),'--check',str(binary)])
    add('Apre',['bash','-c',shell],priority=0,timeout=7200)
    add('MB-1a',[PY,str(script/'dram_ceiling.py'),'--binary','/root/r13_work/tilemega-loadbench',
        '--out',str(HERE/'raw/MB-1a')],gpu=True,priority=3,timeout=7200)
    add('A0',[PY,str(script/'builds_r13.py'),'--jobs',str(HERE/'jobs_a.json'),
        '--out',str(HERE/'builds_a.json')],priority=4,after=['Apre'],timeout=21600)
    add('A0s',[PY,str(script/'phase_a.py'),'smoke','--out',str(HERE/'raw/A0s')],
        gpu=True,priority=5,after=['A0'],timeout=3600)
    total=float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    matrices={}
    for matrix,priority in [('A1',10),('A2',20)]:
        matrices[matrix]=[]
        for cell in CELLS:
            for round_ in range(3):
                name=f'{matrix}_{cell}_r{round_}';matrices[matrix].append(name)
                add(name,[PY,str(script/'phase_a.py'),'anchor','--matrix',matrix,
                    '--cell',cell,'--round',str(round_),'--out',str(HERE/'raw'/name)],
                    gpu=True,priority=priority,after=['A0s'],
                    needs=math.ceil(.85*total)+1024 if matrix=='A1' else 12288)
    for suite in ('b','c','d','e','f'):
        out=HERE/'raw'/f'MB-1{suite}';out.mkdir(parents=True,exist_ok=True)
        add(f'MB-1{suite}',['/root/r13_work/tilemega-loadbench','--suite',suite,'--out',str(out/'loadbench.json')],
            gpu=True,priority=15,after=['MB-1a'],timeout=7200)
    add('Achoose',[PY,str(script/'phase_a.py'),'prefill','--out',str(HERE/'raw/Achoose')],
        priority=21,after_any=matrices['A1'],timeout=600)
    a.queue_dir.mkdir(parents=True,exist_ok=True)
    (a.queue_dir/'queue_a.json').write_text(json.dumps(steps,indent=2)+'\n')
    (HERE/'queue_a.json').write_text(json.dumps(steps,indent=2)+'\n')
    print(json.dumps({'jobs':len(jobs),'steps':len(steps),'missing_inputs':len(missing)}))
if __name__=='__main__':main()
