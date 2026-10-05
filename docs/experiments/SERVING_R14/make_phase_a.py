#!/usr/bin/env python3
"""Freeze all Phase-A jobs, references and commands before GPU execution."""
import copy,json,math,os
from pathlib import Path
from make_phase0 import job_for,write,PYTHON,HERE,ROOT
from tilemega.build.identity import sha

def main():
    jobs=[];arms={};traces={};references={}
    for model,b in [('llama',1),('qwen3',1),('llama',16),('qwen3',16)]:
        cell=f'{model}_B{b}';base=HERE/'raw/A0'/cell/'R13D_prime'
        job,selected,config=job_for(model,b,'R13D_prime',base);jobs.append(job)
        common=dict(kind='tm',root=str(ROOT),python=PYTHON,model=model,model_path=config['model']['path'],batch=b,
            prefill=selected['prefill'],mode='L1',decode_loop=0,prefill_mode='L1',env={},binaries={'prefill':{'sha256':sha(selected['prefill'])}})
        old=dict(common,label='R13D',decode=selected['decode']);old['binaries']=dict(common['binaries'],decode={'sha256':sha(selected['decode'])})
        new=dict(common,label='R13D_prime',decode=str(base/'plan.so'))
        arms[cell]=[dict(common,label='vllm',kind='vllm',python='/root/venv_vllm/bin/python'),old,new]
        trace_bases=[new];references[cell]=[old,new]
        if cell=='llama_B1':
            donor=next(j for j in json.loads((ROOT/'docs/experiments/SERVING_R13/builds_b.json').read_text())
                       if j['cell']==cell and j['label']=='N-R12b-noWD')
            oldso=donor['so'];data=json.loads(Path(oldso+'.plan.json').read_text())
            n=copy.deepcopy(job);n.update(label='N1_prime',manifest=oldso+'.plan.json',classes=oldso+'.classes.tsv',out=str(base.parent/'N1_prime'))
            for key in ('sync','pdl','watchdog','l2_slim','page_loop_split','nonpaged_weight_layout','evict_first','evict_last'):
                n['overrides'][key]=data.get(key,{'pdl':'auto','sync':'calibrated','nonpaged_weight_layout':'row'}.get(key,0))
            n['overrides'].update(paged_la=0,paged_la_splitk=0,lookahead_bytes=0)
            jobs.append(n);narm=dict(common,label='N1_prime',decode=n['out']+'/plan.so');nold=dict(common,label='N1',decode=oldso)
            nold['binaries']=dict(common['binaries'],decode={'sha256':sha(oldso)})
            arms[cell]+=[nold,narm];references[cell]+=[nold,narm];trace_bases.append(narm)
            head=copy.deepcopy(n);head.update(label='B0h_prime',out=str(base.parent/'B0h_prime'),gemm_overrides=[dict(index=data['gemms'][-1]['index'],values=dict(tile_n=128,tile_k=64,stages=4,split_k=1))]);jobs.append(head)
            references[cell].append(dict(common,label='B0h_prime',decode=head['out']+'/plan.so'))
        traces[cell]=[]
        for arm in trace_bases:
            traces[cell].append(arm)
            for suffix,defines in [('stage',{'TILEMEGA_TRACE_STAGE':1,'TILEMEGA_TRACE_STAGE_SAMPLE':8}),
                                   ('task',{'TILEMEGA_TRACE_TASK':1})]:
                out=Path(arm['decode']).parent.parent/(arm['label']+'_'+suffix)
                jobs.append(dict(label=arm['label']+'_'+suffix,cell=cell,base_so=arm['decode'],out=str(out),defines=defines))
                traces[cell].append(dict(arm,label=arm['label']+'_'+suffix,decode=str(out/'plan.so')))
    write(HERE/'phase_a_builds.json',jobs);write(HERE/'phase_a_arms.json',arms);write(HERE/'phase_a_trace_arms.json',traces);write(HERE/'phase_a_references.json',references)
    env=dict(PYTHONPATH=str(ROOT/'python'),TILEMEGA_BIN=str(ROOT/'build-phase12/tools/tilemega'))
    steps=[]
    def step(name,command,gpu=False,after=(),priority=10,timeout=3600,free=12288):
        steps.append(dict(name=name,command=list(map(str,command)),gpu=gpu,after=list(after),priority=priority,timeout_s=timeout,needs_free_mib=free,cwd=str(ROOT),env=env))
    step('Apre',['flock','/root/r14_work/gpu.lock',PYTHON,HERE/'phase_a.py','prepare'],after=['TR4_compile_fixed'],priority=0,timeout=7200)
    step('A0',['flock','/root/r14_work/gpu.lock',PYTHON,HERE/'builds_r14.py','--jobs',HERE/'phase_a_builds.json','--out',HERE/'raw/A0/results.json'],after=['Apre'],priority=1,timeout=14400)
    step('A0s',[PYTHON,HERE/'phase_a.py','smoke'],True,['A0'],5,3600)
    for cell in arms:
        for r in range(3):
            # 24 GiB board: reserve vLLM's 85% plus 1 GiB from actual device total.
            step(f'A1_{cell}_r{r}',[PYTHON,HERE/'anchor.py','--arms',HERE/'phase_a_arms.json','--cell',cell,'--round',r,'--out',HERE/f'raw/A1_{cell}_r{r}'],True,['A0s'],10,5400,math.ceil(.85*gpu_total_mib())+1024)
            step(f'A2_overhead_{cell}_r{r}',[PYTHON,HERE/'anchor.py','--arms',HERE/'phase_a_trace_arms.json','--cell',cell,'--round',r,'--out',HERE/f'raw/A2_overhead_{cell}_r{r}'],True,['A0s'],20,5400)
        step(f'A2_trace_{cell}',[PYTHON,HERE/'phase_a.py','trace','--cell',cell],True,['A0s'],21,1800)
        step(f'A3_{cell}',[PYTHON,HERE/'phase_a.py','correctness','--cell',cell],True,[f'A1_{cell}_r{r}' for r in range(3)],25,7200)
    deps=[s['name'] for s in steps if s['name'].startswith(('A1_','A2_','A3_'))]
    step('A_analyze',[PYTHON,HERE/'phase_a.py','analyze'],after=deps,priority=26,timeout=600)
    write(HERE/'queue_phase_a.pending.json',steps)

def gpu_total_mib():
    import subprocess
    return int(subprocess.check_output(['nvidia-smi','-i',os.getenv('TILEMEGA_DEVICE_INDEX','0'),'--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).splitlines()[0])
if __name__=='__main__':main()
