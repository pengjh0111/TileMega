#!/usr/bin/env python3
"""Freeze-time Phase B definitions; no measurements execute in this generator."""
import argparse,copy,json,math,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PY='/root/venvs/tilemega-torch213-cu126/bin/python';LOCK='/root/r13_work/gpu.lock';BIN=str(ROOT/'build-phase12/tools/tilemega')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();arms=json.loads((HERE/'arms.json').read_text());jobs=[]
    def arm(cell,label):return next(r for r in arms[cell] if r['label']==label)
    def job(cell,label,source,donor,overrides):
        model,batch=cell.split('_B');events=json.loads((ROOT/f'runs/r12b-{model}/cache.json').read_text());event=next(e for e in events if e.get('layer')=='export' and e.get('phase')=='decode')
        src=Path(source['decode']);target=Path(donor['decode'])
        jobs.append(dict(cell=cell,label=label,batch=int(batch),manifest=str(src)+'.plan.json',classes=str(src)+'.classes.tsv',target_classes=str(target)+'.classes.tsv',export=str(Path.home()/'.cache/tilemega/exports'/event['key']/'bridge.json'),overrides=overrides))
    for b in (1,16):
        cell=f'llama_B{b}';paged=arm(cell,'S1P');non=arm(cell,'S1N');old=arm(cell,'R10C')
        choices={'P-base':{},'P-D64K':{'lookahead_bytes':65536},'P-D128K':{'lookahead_bytes':131072},'P-noWD':{'watchdog':0},'P-noDN':{'deferred_norm':0},'P-noLA':{'paged_la':0},'P-noDN-noLA':{'deferred_norm':0,'paged_la':0}}
        for label,extra in choices.items():
            overrides=dict(pg='pages',weight_layout='tiled',sync='calibrated',lookahead_bytes=0,deferred_norm=1,paged_la=1,watchdog=1);overrides.update(extra)
            job(cell,label,paged,old if overrides['deferred_norm']==0 else paged,overrides)
        for label,dn in [('N-base',1),('N-noDN',0)]:job(cell,label,non,non if dn else old,dict(pg='l2',weight_layout='row',sync='calibrated',deferred_norm=dn,lookahead_bytes=0))
    for model in ('llama','qwen3'):
        cell=model+'_B1';old=arm(cell,'R10C');new=arm(cell,'R12bN_L2')
        for i in range(4):job(cell,f'R10G-{i}',old,new if i==3 else old,dict(pg='off' if i<2 else 'l2',sync='legacy' if i==0 else 'calibrated',deferred_norm=int(i==3),weight_layout='row',lookahead_bytes=0))
        job(cell,'N-R12b',new,new,dict(pg='l2',sync='calibrated',deferred_norm=1,weight_layout='row',lookahead_bytes=0))
    (HERE/'fixed_jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    steps=[]
    def add(name,command,gpu=False,priority=50,after=(),after_any=(),timeout=14400,needs=12288,retry=()):
        env=dict(PYTHONPATH=str(ROOT/'python'),TILEMEGA_BIN=BIN,TILEMEGA_GPU_LOCK=LOCK)
        if not gpu:
            # Local builds may create a resource-query context; serialize all
            # CPU builds/checks with the same lock used by guarded timings.
            command=['flock',LOCK,'env','TILEMEGA_GPU_LOCK_HELD=1']+list(command)
        steps.append(dict(name=name,command=list(command),cwd=str(ROOT),env=env,gpu=gpu,priority=priority,after=list(after),after_any=list(after_any),timeout_s=timeout,needs_free_mib=needs,retry_args=list(retry),out=str(HERE/'raw'/name)))
    add('Bpre',['bash','-c','cmake --build build-phase12 --target tilemega tilemega-unit serving_epilogue_test paged_gemm_test serving_attention_cases_test -j 6 && python3 python/tilemega/fingerprint.py --check build-phase12/tools/tilemega'],priority=49,timeout=3600)
    add('B0a',[PY,'-m','tilemega','calibrate','--config','configs/e2e/llama_r12.json','--run-dir','runs/r13-llama'],gpu=True,priority=50,after=['Bpre'],timeout=7200,needs=4096)
    add('B0b',[PY,str(HERE/'fixed_builds.py'),'--jobs',str(HERE/'fixed_jobs.json')],priority=51,after=['Bpre'],timeout=21600)
    add('Bunit',['ctest','--test-dir','build-phase12','-R','^(serving_epilogue|paged_gemm|serving_attention_cases)$','--output-on-failure'],gpu=True,priority=51,after=['Bpre'],timeout=600,needs=4096)
    add('Barch',[BIN,'audit','arch','--cu',str(HERE/'raw/B0b/llama_B1/P-base/plan.so.cu'),'--arch','sm_80','--arch','sm_89','--arch','sm_90','--arch','sm_100','--arch','sm_120','--out',str(HERE/'raw/Barch')],priority=76,after=['B0b'],timeout=7200)
    add('B0c',[PY,str(HERE/'phase_b.py'),'smoke','--out',str(HERE/'raw/B0c')],gpu=True,priority=52,after=['B0b'],timeout=3600)
    matrices={}
    for matrix,cells,priority,dep in [('B1',['llama_B1','llama_B16'],55,['B0c']),('B8',['llama_B1','qwen3_B1'],57,['B0c']),('B3',['llama_B1','llama_B16'],60,['B0c'])]:
        matrices[matrix]=[]
        for cell in cells:
            for rnd in range(3):
                name=f'{matrix}_{cell}_r{rnd}';matrices[matrix].append(name)
                add(name,[PY,str(HERE/'phase_b.py'),'anchor','--matrix',matrix,'--cell',cell,'--round',str(rnd),'--out',str(HERE/'raw'/name)],gpu=True,priority=priority,after=dep,timeout=7200)
    add('B1c',[PY,str(HERE/'choose_defaults_r12c.py')],priority=61,after=matrices['B1'],timeout=600)
    for model in ('llama','qwen3'):
        add('B2_'+model,[PY,'-m','tilemega','build','--config',f'configs/e2e/{model}_r12c.json','--run-dir',f'runs/r13-{model}'],priority=62 if model=='llama' else 63,after=['B0a','B1c'],timeout=28800)
    import subprocess
    total=float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).splitlines()[0]);needs=math.ceil(.85*total)+1024
    finals=[];checks=[]
    for cell in ('llama_B1','qwen3_B1','llama_B16','qwen3_B16'):
        model=cell.split('_B')[0]
        for rnd in range(3):
            name=f'B5_{cell}_r{rnd}';finals.append(name)
            add(name,[PY,str(HERE/'phase_b.py'),'anchor','--matrix','B5','--cell',cell,'--round',str(rnd),'--out',str(HERE/'raw'/name)],gpu=True,priority=70,after=['B2_'+model],timeout=14400,needs=needs)
        name='B7a_'+cell;checks.append(name)
        add(name,[PY,str(HERE/'phase_b.py'),'check','--cell',cell,'--out',str(HERE/'raw'/name)],gpu=True,priority=80,after=['B2_'+model],timeout=10800)
    add('B6',[PY,str(HERE/'phase_b.py'),'trace','--out',str(HERE/'raw/B6')],gpu=True,priority=75,after=['B0b'],timeout=7200)
    add('Bsass',[PY,str(HERE/'phase_b.py'),'audit','--out',str(HERE/'raw/Bsass')],priority=83,after_any=['B0b','B2_llama','B2_qwen3'],timeout=1800)
    add('B7b',[PY,str(HERE/'phase_b.py'),'hf','--out',str(HERE/'raw/B7b')],gpu=True,priority=82,after=finals,timeout=14400)
    add('B7c',[PY,str(HERE/'phase_b.py'),'protocol','--out',str(HERE/'raw/B7c')],gpu=True,priority=95,after=['B2_llama'],timeout=10800,retry=['--resume'])
    add('Breport',[PY,str(HERE/'analyze.py')],priority=99,after_any=finals+checks+['B7b','B7c','B6']+matrices['B8']+matrices['B3'],timeout=1800)
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'queue_b.json').write_text(json.dumps(steps,indent=2)+'\n');print(f'Phase B: {len(steps)} steps; {len(jobs)} builds plus two trace builds')
if __name__=='__main__':main()
