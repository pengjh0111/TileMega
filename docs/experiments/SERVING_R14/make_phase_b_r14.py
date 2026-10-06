#!/usr/bin/env python3
"""Prepare the complete fixed Phase-B matrix before any timing begins."""
import copy,json
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write
CELLS=('llama_B1','llama_B16','qwen3_B1','qwen3_B16')
def main():
    donors=json.loads((HERE/'phase_a_builds.json').read_text());oldarms=json.loads((HERE/'phase_a_arms.json').read_text())
    choices=json.loads((HERE/'baselines_r14.json').read_text());la=json.loads((HERE/'la_builds.json').read_text())
    jobs=[];arms={};matrices={f'B{i}':{} for i in range(1,6)}
    for cell in CELLS:
        source_label='N1_prime' if choices[cell]=="N1'" else 'R13D_prime'
        donor=next(j for j in donors if j['cell']==cell and j['label']==source_label)
        old=next(a for a in oldarms[cell] if a['label']==source_label)
        base=copy.deepcopy(donor);base.update(manifest=old['decode']+'.plan.json',classes=old['decode']+'.classes.tsv')
        base.pop('gemm_overrides',None);manifest=json.loads(Path(base['manifest']).read_text());paged=manifest['pg']=='pages'
        arms[cell]=[]
        def add(label,overrides=None,geometry=None,source=None,matrix=None):
            j=copy.deepcopy(source or base);j.update(cell=cell,batch=int(cell.split('_B')[1]),label=label,out=str(HERE/'raw/B0b'/cell/label))
            j['overrides'].update(overrides or {})
            if geometry:j['gemm_overrides']=geometry
            jobs.append(j)
            arm=dict(old,label=label,decode=j['out']+'/plan.so',root=str(ROOT),python=PYTHON,mode='L1',decode_loop=0,prefill_mode='L1',env={})
            arm.pop('binaries',None);arms[cell].append(arm)
            if matrix:matrices[matrix].setdefault(cell,[]).append(arm)
            return j,arm
        _,baseline=add('baseline')
        for matrix in matrices.values():matrix[cell]=[baseline]
        for ec in (32,64,128,256):add(f'AT_ec{ec}',{'serve_kv_block':ec},matrix='B1')
        add('AT_pv',{'attention_impl':'pvswap'},matrix='B2')
        nonpaged=not paged or cell.endswith('_B1')
        if nonpaged:
            nd=base if not paged else next(j for j in la if j['cell']==cell)
            if paged:
                _,nbase=add('AT_la_ref',{'nonpaged_la':0},source=nd)
                matrices['B3'][cell].append(nbase) # Same nonpaged geometry for the GEMV comparison.
            add('AT_la',{'nonpaged_la':1},source=nd,matrix='B2')
            add('AT_la_pv',{'nonpaged_la':1,'attention_impl':'pvswap'},source=nd,matrix='B2')
        fill=[]
        for g in manifest['gemms'][:-1]:
            if g['index']%4 in (1,3):
                n=json.loads((Path(old['model_path'])/'config.json').read_text())['hidden_size']
                tiles=((base['batch']+g['tile_m']-1)//g['tile_m'])*((n+g['tile_n']-1)//g['tile_n'])
                split=min(4,max(1,(manifest['grid']+tiles-1)//tiles))
                fill.append(dict(index=g['index'],values={'split_k':split}))
        add('SK_fill',{'paged_la_splitk':1} if paged else {'nonpaged_la':1},fill,matrix='B3')
        if cell=='llama_B16':add('SK_fill_ref',{'nonpaged_la':0},fill)
        if cell.endswith('_B1'):
            nd=base if not paged else next(j for j in la if j['cell']==cell)
            gm=json.loads(Path(nd['manifest']).read_text())['gemms']
            for cls,indices in [('o',[g['index'] for g in gm[:-1] if g['index']%4==1]),('down',[g['index'] for g in gm[:-1] if g['index']%4==3]),('all',[g['index'] for g in gm])]:
                geom=[dict(index=i,values=dict(tile_m=16,tile_n=32,tile_k=128,stages=2,split_k=1,impl='gemv')) for i in indices]
                add('GV_'+cls,{'nonpaged_la':0},geom,nd,'B3')
        if paged:add('RW_pipe',{'mma_reg_pipe':1},matrix='B4')
        if cell.startswith('qwen3'):add('RA_noinline',{'attention_noinline':1},matrix='B5')
        add('EP_arg',{'parallel_argmax':1},matrix='B5')
    # All trace variants are built from their exact nontrace parent artifacts.
    for cell,items in arms.items():
        for arm in items:
            jobs.append(dict(cell=cell,label=arm['label']+'_task',base_so=arm['decode'],out=str(Path(arm['decode']).parent)+'_task',defines={'TILEMEGA_TRACE_TASK':1}))
    write(HERE/'phase_b_builds.json',jobs);write(HERE/'phase_b_arms.json',arms)
    for name,matrix in matrices.items():write(HERE/(name+'_arms.declared.json'),matrix)
    env=dict(PYTHONPATH=str(ROOT/'python'),TILEMEGA_BIN=str(ROOT/'build-phase12/tools/tilemega'),CUDACXX='/usr/local/cuda/bin/nvcc')
    q=[]
    def step(name,args,gpu=False,after=(),priority=40,timeout=3600):
        q.append(dict(name=name,command=list(map(str,args)),gpu=gpu,after=list(after),priority=priority,timeout_s=timeout,needs_free_mib=12288,cwd=str(ROOT),env=env))
    driver=HERE/'phase_b_r14.py'
    step('Bpre',['flock','/root/r14_work/gpu.lock',PYTHON,driver,'prepare'],after=['GV_model_smoke'],priority=30,timeout=7200)
    step('Bpre_numeric',[PYTHON,driver,'numerical'],True,['Bpre'],31,3600)
    step('B0b',['flock','/root/r14_work/gpu.lock',PYTHON,HERE/'builds_r14.py','--jobs',HERE/'phase_b_builds.json','--out',HERE/'raw/B0b/results.json','--keep-going'],after=['Bpre_numeric'],priority=31,timeout=28800)
    for cell in CELLS:
        step('B0c_'+cell,[PYTHON,driver,'smoke','--cell',cell],True,['B0b'],32,5400)
        for name,matrix in matrices.items():
            if len(matrix[cell])<2:continue
            for r in range(3):
                step(f'{name}_{cell}_r{r}',[PYTHON,HERE/'anchor.py','--arms',HERE/(name+'_'+cell+'_arms.json'),'--cell',cell,'--round',r,'--out',HERE/f'raw/{name}_{cell}_r{r}'],True,['B0c_'+cell],40+int(name[1:]),7200)
        step('B_trace_'+cell,[PYTHON,driver,'trace','--cell',cell],True,['B0c_'+cell],46,5400)
        step('B6_'+cell,[PYTHON,driver,'correctness','--cell',cell],True,['B0c_'+cell],60,14400)
    for cell,label in [('llama_B1','AT_la'),('qwen3_B1','AT_la'),('llama_B16','SK_fill')]:
        step('B6_protocol_'+cell,[PYTHON,driver,'protocol','--cell',cell,'--label',label],True,['B0c_'+cell],65,14400)
    write(HERE/'queue_phase_b.pending.json',q)
    print(f'Prepared {len(jobs)} fixed/trace artifacts and {len(q)} steps; queue not published')
if __name__=='__main__':main()
