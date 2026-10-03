#!/usr/bin/env python3
"""Prepare frozen Phase B inputs without reading or polling running jobs."""
import argparse,copy,json,os,shlex,subprocess
from pathlib import Path
from make_queue_r13 import CELLS,PY,LOCK,HERE,ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--queue-dir',type=Path,required=True);a=p.parse_args()
    compiler=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    jobs=copy.deepcopy(json.loads((HERE/'jobs_a.json').read_text()))
    for row in jobs:
        row['out']=str(HERE/'raw/B0b'/row['cell']/row['label']);row['overrides']['pdl']='auto'
    off=copy.deepcopy(next(r for r in jobs if r['cell']=='llama_B1' and r['label']=='N-R12b-noWD'))
    off['label']='N-R12b-noWD-pdl-off';off['out']=str(HERE/'raw/B0b/llama_B1'/off['label']);off['overrides']['pdl']='off';jobs.append(off)
    for cell in CELLS:
        model,b=cell.split('_B');so=json.loads((ROOT/f'runs/r12c-{model}/plans.json').read_text())[b]['decode']
        base=next(r for r in jobs if r['cell']==cell and r['label']=='P-R12bN-noWD')
        for label,enabled in [('PS',0),('PSA',1)]:
            job=copy.deepcopy(base);job.update(label=label,manifest=so+'.plan.json',classes=so+'.classes.tsv',projection='identity',out=str(HERE/'raw/B0b'/cell/label))
            job['overrides'].update(paged_la_splitk=enabled);jobs.append(job)
    (HERE/'jobs_b.json').write_text(json.dumps(jobs,indent=2)+'\n')
    env=dict(PYTHONPATH=str(ROOT/'python'),TILEMEGA_BIN=str(ROOT/'build-phase12/tools/tilemega'),
             TILEMEGA_R13_ROOT=str(ROOT),TILEMEGA_R13_DATA=str(HERE),
             TILEMEGA_R13_COMPILER_COMMIT=compiler,TILEMEGA_GPU_LOCK=LOCK)
    steps=[]
    def add(name,cmd,gpu=False,priority=50,after=(),after_any=(),timeout=14400,retry=()):
        if not gpu:cmd=['flock',LOCK,'env','TILEMEGA_GPU_LOCK_HELD=1']+cmd
        steps.append(dict(name=name,command=cmd,cwd=str(ROOT),env=env,gpu=gpu,
            priority=priority,after=list(after),after_any=list(after_any),timeout_s=timeout,
            needs_free_mib=12288,retry_args=list(retry),out=str(HERE/'raw'/name)))
    cmd=' && '.join(shlex.join(c) for c in [
        ['cmake','--build','build-phase12','--target','tilemega','tilemega-unit','-j','6'],
        ['python3','python/tilemega/fingerprint.py','--check',env['TILEMEGA_BIN']],
        ['ctest','--test-dir','build-phase12','-R','^(handoff_|serving_lag|serving_page_layout|serving_task_index)','--output-on-failure']])
    add('Bpre',['bash','-c',cmd],priority=40,timeout=7200)
    add('B0b',[PY,str(HERE/'phase_b_r13.py'),'build','--out',str(HERE/'raw/B0b')],priority=41,after=['Bpre'],timeout=43200)
    add('B0c',[PY,str(HERE/'phase_b_r13.py'),'smoke','--out',str(HERE/'raw/B0c')],gpu=True,priority=50,after=['B0b','Achoose'],timeout=7200)
    for cell in CELLS:
        add('B1_'+cell,[PY,str(HERE/'phase_b_r13.py'),'B1','--cell',cell,'--out',str(HERE/'raw'/('B1_'+cell))],gpu=True,priority=52,after=['B0c'],timeout=3600)
        for matrix,priority in [('B2',55),('B3',57),('B4',60)]:
            for rnd in range(3):
                name=f'{matrix}_{cell}_r{rnd}'
                add(name,[PY,str(HERE/'phase_b_r13.py'),'anchor','--matrix',matrix,'--cell',cell,
                    '--round',str(rnd),'--out',str(HERE/'raw'/name)],gpu=True,priority=priority,after=['B0c'],timeout=10800)
        for action,priority in [('B2trace',56),('B3trace',58),('B4trace',61)]:
            name=action+'_'+cell
            add(name,[PY,str(HERE/'phase_b_r13.py'),action,'--cell',cell,'--out',str(HERE/'raw'/name)],gpu=True,priority=priority,after=['B0c'],timeout=3600)
        add('B5_'+cell,[PY,str(HERE/'check_r13.py'),'--cell',cell,'--out',str(HERE/'raw'/('B5_'+cell))],gpu=True,priority=80,after=['B0c'],after_any=[f'B4_{cell}_r{r}' for r in range(3)],timeout=10800)
    add('B5_protocol',[PY,str(HERE/'phase_b_r13.py'),'protocol','--out',str(HERE/'raw/B5_protocol')],gpu=True,priority=81,after=['B0c'],timeout=14400,retry=['--resume'])
    add('Barch',[PY,str(HERE/'arch_checks.py'),'--nonpaged',str(HERE/'raw/B0b/llama_B1/N-R12b-noWD/plan.so'),
        '--paged',str(HERE/'raw/B0b/llama_B1/P-R12bN-noWD/plan.so'),'--out',str(HERE/'raw/Barch')],priority=82,after=['B0b'],timeout=14400)
    add('Bidentity',[PY,str(HERE/'identity_r13.py'),'--out',str(HERE/'raw/Bidentity')],priority=83,after=['B0b','A0'],timeout=3600)
    matrices=[s['name'] for s in steps if s.get('gpu')]
    add('Banalyze',[PY,str(HERE/'analyze.py')],priority=85,after_any=matrices+['Barch','Bidentity'],timeout=3600)
    a.queue_dir.mkdir(parents=True,exist_ok=True)
    data=json.dumps(steps,indent=2)+'\n';(HERE/'queue_b.json').write_text(data)
    (a.queue_dir/'queue_b.json').write_text(data)
    (HERE/'freeze.json').write_text(json.dumps(dict(compiler_commit=compiler,jobs=len(jobs),steps=len(steps),
        phase='B',phase_c='not submitted; requires registered decision after B',queue_dir=str(a.queue_dir)),indent=2)+'\n')
    print(json.dumps(dict(jobs=len(jobs),steps=len(steps),compiler_commit=compiler)))
if __name__=='__main__':main()
