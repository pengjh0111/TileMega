#!/usr/bin/env python3
"""Prepare only registered Phase-C arms; publish after all inputs are committed."""
import copy,json
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write
from choose_r14 import phase_c

def main():
    decision=json.loads((HERE/'phase_c_decision.json').read_text());inputs=json.loads((HERE/'results/phase_c_inputs.json').read_text())
    if phase_c(inputs)!=decision:raise ValueError('decision differs from registered rules')
    sources=json.loads((HERE/'phase_b_builds.json').read_text());oldarms=json.loads((HERE/'phase_b_arms.json').read_text());jobs=[];arms={}
    for cell,rule,label in [('llama_B16','C-RW1','C_RW1'),('qwen3_B16','C-EP2','C_EP2'),('qwen3_B1','C-AT4','C_AT4')]:
        if cell not in decision[rule]['cells']:continue
        donor=next(j for j in sources if j['cell']==cell and j['label']=='baseline');arm=next(a for a in oldarms[cell] if a['label']=='baseline')
        arms[cell]=[]
        for name in ('baseline',label):
            j=copy.deepcopy(donor);j.update(label=name,out=str(HERE/'raw/C_build'/cell/name),manifest=arm['decode']+'.plan.json',classes=arm['decode']+'.classes.tsv');j.pop('gemm_overrides',None)
            if name=='C_RW1':
                m=json.loads(Path(j['manifest']).read_text());j['manifest_overrides']={'residency':2,'grid':2*m['grid']};j['overrides']['attention_buffers']=1
                j['gemm_overrides']=[dict(index=g['index'],values=dict(tile_n=128 if g['index']==len(m['gemms'])-1 else g['tile_n'],stages=min(g['stages'],4 if g['tile_n']==32 else 2))) for g in m['gemms']]
            elif name=='C_EP2':j['overrides']['ep_direct']=1
            elif name=='C_AT4':j['overrides']['attention_frontier']=1
            jobs.append(j);arms[cell].append(dict(arm,label=name,decode=j['out']+'/plan.so'))
    write(HERE/'phase_c_builds.json',jobs);write(HERE/'phase_c_arms.json',arms)
    q=[];driver=HERE/'phase_c_r14.py';env=dict(PYTHONPATH=str(ROOT/'python'),TILEMEGA_BIN=str(ROOT/'build-phase12/tools/tilemega'),CUDACXX='/usr/local/cuda/bin/nvcc')
    def step(name,args,gpu=False,after=(),priority=70,timeout=3600):q.append(dict(name=name,command=list(map(str,args)),gpu=gpu,after=list(after),priority=priority,timeout_s=timeout,needs_free_mib=12288,cwd=str(ROOT),env=env))
    done=[s['name'] for s in json.loads((HERE/'queue/queue_phase_b.json').read_text())]
    step('C_archive',[PYTHON,HERE/'accept_phase_b.py','--archive'],after=done,priority=69,timeout=3600)
    step('Cpre',['flock','/root/r14_work/gpu.lock',PYTHON,driver,'prepare'],after=['C_archive'],priority=70,timeout=10800)
    step('C_numeric',[PYTHON,driver,'numerical'],True,['Cpre'],71,1800)
    step('C_build',['flock','/root/r14_work/gpu.lock',PYTHON,HERE/'builds_r14.py','--jobs',HERE/'phase_c_builds.json','--out',HERE/'raw/C_build/results.json','--keep-going'],after=['C_numeric'],priority=72,timeout=14400)
    for cell in arms:
        step('C_smoke_'+cell,[PYTHON,driver,'smoke','--cell',cell],True,['C_build'],73,1200)
        for r in range(3):step(f'C_matrix_{cell}_r{r}',[PYTHON,HERE/'anchor.py','--arms',HERE/f'C_{cell}_arms.json','--cell',cell,'--round',r,'--out',HERE/f'raw/C_matrix_{cell}_r{r}'],True,['C_smoke_'+cell],75,1800)
        step('C_correct_'+cell,[PYTHON,driver,'correctness','--cell',cell],True,['C_smoke_'+cell],76,3600)
    step('C_arch',['flock','/root/r14_work/gpu.lock',PYTHON,driver,'architecture'],after=['C_smoke_qwen3_B1'],priority=74,timeout=9000)
    for cell,label in [('llama_B16','C_RW1'),('qwen3_B1','C_AT4')]:step('C_protocol_'+cell,[PYTHON,driver,'protocol','--cell',cell,'--label',label],True,['C_correct_'+cell],77,14400)
    write(HERE/'queue_phase_c.pending.json',q);print(len(jobs),'artifacts',len(q),'steps prepared, unpublished')
if __name__=='__main__':main()
