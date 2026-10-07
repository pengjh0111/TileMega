#!/usr/bin/env python3
"""Prepare D0/D1 and immutable controls before publishing GPU work."""
import copy,json,math
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write
from tilemega.cli import read_config

def main():
    retention=json.loads((HERE/'phase_c_retention.json').read_text())
    if any(r['retain'] for r in retention.values()):raise ValueError('configure retained flags explicitly before publishing')
    baseline_jobs=[];baseline_arms={}
    source=json.loads((HERE/'phase_b_builds.json').read_text());arms=json.loads((HERE/'phase_b_arms.json').read_text())
    for model in ('llama','qwen3'):
        cfg=read_config(ROOT/f'configs/e2e/{model}_r13.json')
        cfg['output']['dir']=str(ROOT/f'runs/r14-{model}')
        cfg['solver'].update(measure_top=6,top_m=8,jobs=3,time_budget_s=1800,candidate_guard_wait_s=1800,mode='L1',candidate_loop=0,resident2_batches=[16] if model=='llama' else [])
        cfg['features'].update(pg='measure',decode_executor='measure',decode_loop='measure',prefill_executor='L1',watchdog=0,mma_reg_pipe=0,parallel_argmax=0,attention_noinline=0,attention_impl='mma16',attention_buffers=2,attention_frontier=0,ep_direct=0,nonpaged_la=1)
        cfg.pop('features_by_batch',None)
        cfg['r14_decisions']=dict(conditional_retention=retention,nonpaged_la='first-stage fill candidates use LA; second-stage compares LA=0/1',resident2='C-RW1 fixed trial lost; SL-6 still includes its required structural family for Llama B16 alongside ordinary double-buffer plans; no default enabled before final acceptance')
        write(ROOT/f'configs/e2e/{model}_r14.json',cfg)
        read_config(ROOT/f'configs/e2e/{model}_r14.json')
        for b in (1,16):
            cell=f'{model}_B{b}';j=copy.deepcopy(next(x for x in source if x['cell']==cell and x['label']=='baseline'))
            j.update(label='baseline',out=str(HERE/'raw/D_baseline_build'/cell))
            baseline_jobs.append(j)
            a=copy.deepcopy(next(x for x in arms[cell] if x['label']=='baseline'));a['decode']=j['out']+'/plan.so';baseline_arms[cell]=a
    write(HERE/'phase_d_baseline_builds.json',baseline_jobs);write(HERE/'phase_d_baseline_arms.json',baseline_arms)
    q=[];env=dict(PYTHONPATH=str(ROOT/'python'),TILEMEGA_BIN=str(ROOT/'build-phase12/tools/tilemega'),CUDACXX='/usr/local/cuda/bin/nvcc')
    def step(name,args,gpu=False,after=(),priority=80,timeout=3600,free=12288):
        q.append(dict(name=name,command=list(map(str,args)),gpu=gpu,after=list(after),priority=priority,timeout_s=timeout,needs_free_mib=free,cwd=str(ROOT),env=env))
    driver=HERE/'phase_d_r14.py'
    step('C_arch_v2',['flock','/root/r14_work/gpu.lock',PYTHON,HERE/'phase_c_r14.py','architecture'],after=['C_smoke_qwen3_B1'],priority=79,timeout=9000)
    step('Dpre',['flock','/root/r14_work/gpu.lock',PYTHON,driver,'prepare'],after=['C_arch_v2'],priority=80,timeout=7200)
    step('D0',[PYTHON,driver,'calibrate'],True,['Dpre'],81,7200,4096)
    step('D_baseline_build',['flock','/root/r14_work/gpu.lock',PYTHON,HERE/'builds_r14.py','--jobs',HERE/'phase_d_baseline_builds.json','--out',HERE/'raw/D_baseline_build/results.json'],after=['Dpre'],priority=82,timeout=14400)
    step('D_baseline_smoke',[PYTHON,driver,'baseline_smoke'],True,['D_baseline_build'],83,2400)
    for model in ('llama','qwen3'):
        # Guard the entire compile/measure command so hidden external users
        # invalidate the attempt. Children inherit LOCK_HELD and cannot self-lock.
        step('D1_'+model,[PYTHON,driver,'build','--model',model],True,['D0','D_baseline_smoke'],83,28800)
        step('D1_smoke_'+model,[PYTHON,driver,'smoke','--model',model],True,['D1_'+model],84,1800)
        step('D1_family_'+model,[PYTHON,driver,'family','--model',model],after=['D1_smoke_'+model],priority=85,timeout=600)
    write(HERE/'queue_phase_d1.pending.json',q)
    print(len(q),'D0/D1 and architecture repair steps prepared; D2 awaits PlanFamily gate')
if __name__=='__main__':main()
