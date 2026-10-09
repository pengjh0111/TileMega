#!/usr/bin/env python3
"""Prepare immutable repair inputs; publish only after reviewing their commit."""
import copy,json
from make_phase0 import HERE,ROOT,PYTHON,write

def definitions():
    baselines=json.loads((HERE/'phase_d_baseline_builds_v2.json').read_text())
    arms=json.loads((HERE/'phase_d_baseline_arms_v2.json').read_text())
    for j in baselines:
        j['out']=str(HERE/'raw/R_complete_build'/j['cell']/'baseline')
        arms[j['cell']]['decode']=str(j['out'])+'/plan.so'
    jobs=list(baselines)
    narrow=[]
    donors=json.loads((HERE/'gemv_builds.json').read_text())
    for model in ('llama','qwen3'):
        donor=next(j for j in donors if j['cell']==model+'_B1')
        for tn,layout in ((8,'row'),(16,'tiled')):
            j=copy.deepcopy(donor);j['label']=f'{model}_tn{tn}_{layout}'
            j['out']=str(HERE/'raw/R_complete_build'/j['cell']/j['label'])
            j['overrides']['nonpaged_weight_layout']=layout
            for g in j['gemm_overrides']:g['values']['tile_n']=tn
            jobs.append(j);narrow.append(j['label'])
    for kind in ('stage','task'):
        jobs.append(dict(label=kind,cell='llama_B1',base_so=arms['llama_B1']['decode'],
            out=str(HERE/'raw/R_complete_build/llama_B1'/kind),defines={
                'TILEMEGA_TRACE_STAGE':int(kind=='stage'),'TILEMEGA_TRACE_TASK':int(kind=='task'),'TILEMEGA_TRACE_STEP':0}))
    trace_arms=[]
    for kind in ('baseline','stage','task'):
        arm=copy.deepcopy(arms['llama_B1']);arm['label']=kind
        arm['decode']=str(HERE/f'raw/R_complete_build/llama_B1/{kind}/plan.so');trace_arms.append(arm)
    write(HERE/'completion_builds.json',jobs);write(HERE/'phase_d_baseline_arms_v3.json',arms)
    write(HERE/'completion_trace_arms.json',dict(llama_B1=trace_arms))
    env=dict(PYTHONPATH=str(ROOT/'python'),TILEMEGA_BIN=str(ROOT/'build-phase12/tools/tilemega'))
    steps=[];driver=HERE/'completion_repair.py';d_driver=HERE/'phase_d_r14.py'
    def add(name,cmd,after=(),gpu=False,priority=0,timeout=3600):
        own_env=dict(env)
        if not gpu:
            cmd=['flock','/root/r14_work/gpu.lock']+cmd;own_env['TILEMEGA_GPU_LOCK_HELD']='1'
        steps.append(dict(name=name,command=list(map(str,cmd)),cwd=str(ROOT),env=own_env,after=list(after),
            gpu=gpu,priority=priority,timeout_s=timeout,needs_free_mib=12288))
    def action(name,which,after,**kw):add(name,[PYTHON,driver,which],after,**kw)
    action('R_complete_prepare','prepare',[],timeout=7200)
    action('R_complete_numeric','numeric',['R_complete_prepare'],gpu=True,priority=1,timeout=600)
    add('R_complete_build',[PYTHON,HERE/'builds_r14.py','--jobs',HERE/'completion_builds.json',
        '--out',HERE/'raw/R_complete_build/results.json'],['R_complete_numeric'],timeout=14400)
    action('R_complete_compare','compare',['R_complete_build'],timeout=3000)
    action('R_complete_smoke','smoke',['R_complete_build'],gpu=True,priority=2,timeout=5400)
    for label in narrow:
        add('R_complete_c1_'+label,[PYTHON,driver,'correctness','--label',label],['R_complete_smoke'],gpu=True,priority=3,timeout=5400)
    required=['R_complete_compare','R_complete_smoke']+['R_complete_c1_'+label for label in narrow]
    action('R_complete_trace','trace',required,gpu=True,priority=4,timeout=3600)
    for r in range(3):
        add(f'R_complete_overhead_r{r}',[PYTHON,HERE/'anchor.py','--arms',HERE/'completion_trace_arms.json',
            '--cell','llama_B1','--round',r,'--out',HERE/f'raw/R_complete_overhead_r{r}'],required,gpu=True,priority=5,timeout=1800)
    action('R_complete_overhead','overhead',[f'R_complete_overhead_r{r}' for r in range(3)],timeout=300)
    add('D0_v3',[PYTHON,d_driver,'calibrate','--tag','_v3'],required+['R_complete_trace','R_complete_overhead'],gpu=True,priority=10,timeout=10800)
    for model in ('llama','qwen3'):
        add(f'D1_{model}_v3',[PYTHON,d_driver,'build','--model',model,'--tag','_v3'],['D0_v3'],gpu=True,priority=11,timeout=46000)
        add(f'D1_smoke_{model}_v3',[PYTHON,d_driver,'smoke','--model',model,'--tag','_v3'],[f'D1_{model}_v3'],gpu=True,priority=12,timeout=1800)
        add(f'D1_family_{model}_v3',[PYTHON,d_driver,'family','--model',model,'--tag','_v3'],[f'D1_smoke_{model}_v3'],timeout=300)
    return steps

if __name__=='__main__':
    steps=definitions();write(HERE/'queue_completion_repair.pending.json',steps)
    print(f'{len(steps)} repair steps prepared; no old D1 winner or final queue is reused')
