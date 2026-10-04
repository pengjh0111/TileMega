#!/usr/bin/env python3
"""Publish predeclared C controls and dependent D stages to the live scheduler."""
import argparse,copy,json,os,shlex,subprocess
from pathlib import Path
from phase_cd_r13 import HERE,ROOT,CELLS,MATRICES,read,write
PY='/root/venvs/tilemega-torch213-cu126/bin/python';LOCK='/root/r13_work/gpu.lock'
def main():
    p=argparse.ArgumentParser();p.add_argument('--queue-dir',type=Path,required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
    decision=read(HERE/'phase_c_decision.json');source=read(HERE/'jobs_b.json');chosen=read(HERE/'defaults_r13.json')['prefill'];jobs=[]
    def job(cell,donor,label,overrides):
        row=copy.deepcopy(next(r for r in source if r['cell']==cell and r['label']==donor))
        row['model']=cell.split('_B')[0];row['label']=label;row['out']=str(HERE/'raw/Cbuild'/cell/label)
        row['overrides'].update(l2_slim=0,page_loop_split=0,nonpaged_weight_layout='row',evict_first=0,evict_last=1,watchdog=0)
        row['overrides'].update(overrides);jobs.append(row);return row
    for cell in CELLS:
        job(cell,'N-R12b-noWD','N-control',{});job(cell,'P-R12bN-noWD','P-control',{})
        job(cell,chosen[cell],chosen[cell],{})
        if 'C-L2b' in decision['selected']:job(cell,'N-R12b-noWD','N-slim',{'l2_slim':1})
        if 'C-WL' in decision['selected']:job(cell,'N-R12b-noWD','N-tiled',{'nonpaged_weight_layout':'tiled'})
        if 'C-LP2' in decision['selected']:job(cell,'P-R12bN-noWD','P-loop-split',{'page_loop_split':1})
        if 'C-PG3' in decision['selected']:
            for label,first,last in [('P-D64-last',0,1),('P-D64-normal',0,0),('P-D64-first',1,0)]:
                job(cell,'P-R12bN-noWD',label,dict(lookahead_bytes=65536,evict_first=first,evict_last=last))
        if 'C-AT' in decision['selected'] and cell.startswith('qwen3'):
            donor=next(r for r in source if r['cell']==cell and r['label']=='N-R12b-noWD')
            rq=read(donor['manifest'])['attention_query_rows'];alt=max(1,rq//2)
            for ec in (128,256,512):
                for q in sorted({rq,alt}):job(cell,'N-R12b-noWD',f'N-AT-{ec}-{q}',dict(serve_kv_block=ec,serve_query_rows=q))
    write(HERE/'jobs_c.json',jobs)
    env=dict(PYTHONPATH=str(ROOT/'python'),TILEMEGA_BIN=str(ROOT/'build-phase12/tools/tilemega'),
      TILEMEGA_R13_ROOT=str(ROOT),TILEMEGA_R13_DATA=str(HERE),
      TILEMEGA_R13_COMPILER_COMMIT=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      TILEMEGA_GPU_LOCK=LOCK,PATH='/usr/local/cuda/bin:'+os.environ['PATH'])
    steps=[]
    def add(name,cmd,gpu=False,priority=87,after=(),after_any=(),timeout=14400,free=12288,retry=()):
        if not gpu:cmd=['flock',LOCK,'env','TILEMEGA_GPU_LOCK_HELD=1']+list(map(str,cmd))
        steps.append(dict(name=name,command=list(map(str,cmd)),cwd=str(ROOT),env=env,gpu=gpu,priority=priority,
          after=list(after),after_any=list(after_any),timeout_s=timeout,needs_free_mib=free,retry_args=list(retry),out=str(HERE/'raw'/name)))
    def action(name,act,**kw):
        extra=kw.pop('extra',());add(name,[PY,HERE/'phase_cd_r13.py',act,'--out',HERE/'raw'/name,*extra],**kw)
    prep=['cmake','--build','build-phase12','--target','tilemega','tilemega-unit','-j','6']
    check=['python3','python/tilemega/fingerprint.py','--check',env['TILEMEGA_BIN']]
    add('Cfreeze',['bash','-c',shlex.join(prep)+' && '+shlex.join(check)],after_any=['Cpre'],timeout=7200)
    action('Csmall','small',gpu=True,after=['Cpre'],priority=87,timeout=1800)
    action('Cbuild','build-c',after=['Cfreeze'],after_any=['Csmall'],timeout=43200)
    action('Csmoke','smoke-c',gpu=True,after=['Cbuild'],priority=88,timeout=10800)
    add('Carch',[PY,HERE/'arch_checks.py','--nonpaged',HERE/'raw/Cbuild/llama_B1/N-tiled/plan.so',
        '--paged',HERE/'raw/Cbuild/llama_B1/P-loop-split/plan.so','--out',HERE/'raw/Carch'],
        after=['Cbuild'],priority=94,timeout=14400)
    for label,reference,donor in [('nonpaged','N-R12b-noWD','N-control'),('paged','P-R12bN-noWD','P-control')]:
        add('Cidentity_'+label,[PY,HERE/'compare_kernels.py','--reference',HERE/'raw/A0/llama_B1'/reference/'plan.so',
            '--candidate',HERE/'raw/Cbuild/llama_B1'/donor/'plan.so','--ignore-pdl-macro','--out',HERE/'raw'/('Cidentity_'+label)],
            after=['Cbuild'],priority=95,timeout=3600)
    matrices=[]
    for matrix in decision['selected']:
        if matrix not in MATRICES and matrix!='C-AT':continue
        for cell in CELLS:
            if matrix=='C-AT' and not cell.startswith('qwen3'):continue
            for rnd in range(3):
                name=f'C_{matrix}_{cell}_r{rnd}';matrices.append(name)
                action(name,'anchor-c',gpu=True,after=['Csmoke'],priority=89,timeout=5400,
                    extra=['--matrix',matrix,'--cell',cell,'--round',str(rnd)])
    protocols=[]
    for matrix in ('C-L2b','C-WL','C-LP2'):
        if matrix not in decision['selected']:continue
        name='Cprotocol_'+matrix;protocols.append(name)
        action(name,'protocol-c',gpu=True,after=['Csmoke'],priority=90,timeout=7200,retry=['--resume'],extra=['--matrix',matrix])
    action('Ccheck','check-c',gpu=True,after=['Csmoke'],after_any=matrices,priority=91,timeout=3600)
    action('Cretain','retain',after_any=matrices+protocols+['Ccheck','Csmoke','Cbuild'],priority=92,timeout=1200)
    # D0 changes only invalid calibration sections and the documented DRAM
    # value. D1 always runs after retention, including a no-change outcome.
    action('D0','d0',gpu=True,after=['Cretain','Cfreeze'],priority=100,timeout=14400,free=4096)
    for model in ('llama','qwen3'):
        action('D1_'+model,'d1',after=['D0'],priority=102,timeout=43200,extra=['--model',model])
    rounds=[]
    for cell in CELLS:
        model=cell.split('_B')[0]
        for rnd in range(3):
            name=f'D2_{cell}_r{rnd}';rounds.append(name)
            action(name,'anchor-d',gpu=True,after=['D1_'+model],priority=105,timeout=10800,free=42793,
                extra=['--cell',cell,'--round',str(rnd)])
        action('D3_'+cell,'check-d',gpu=True,after=['D1_'+model],after_any=[f'D2_{cell}_r{r}' for r in range(3)],
               priority=110,timeout=10800,extra=['--cell',cell])
    action('D3_protocol','protocol-d',gpu=True,after=['D1_llama'],priority=111,timeout=10800,retry=['--resume'])
    action('Dfinal','final',after_any=rounds+['D3_'+c for c in CELLS]+['D3_protocol'],priority=112,timeout=1800)
    add('CDanalyze',[PY,HERE/'analyze.py'],after_any=['Dfinal'],priority=113,timeout=3600)
    a.queue_dir.mkdir(parents=True,exist_ok=True);path=HERE/'queue_cd.json';write(path,steps)
    if a.publish:
        dest=a.queue_dir/path.name;tmp=dest.with_suffix('.tmp');tmp.write_bytes(path.read_bytes());tmp.replace(dest)
    print(json.dumps(dict(fixed_jobs=len(jobs),steps=len(steps),published=a.publish)))
if __name__=='__main__':main()
