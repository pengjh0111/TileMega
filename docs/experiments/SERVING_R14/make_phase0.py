#!/usr/bin/env python3
"""Prepare all Phase-0 trace arms before any timing process can start."""
import json,shutil,sys
from pathlib import Path
from tilemega.build.identity import sha
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PYTHON='/root/venvs/tilemega-torch213-cu126/bin/python'
def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n')
def job_for(model,batch,label,out):
    config=json.loads((ROOT/f'configs/e2e/{model}_r13_final.json').read_text())
    selected=config['selected_plans'][str(batch)];so=Path(selected['decode'])
    manifest=Path(str(so)+'.plan.json');data=json.loads(manifest.read_text())
    jobs=json.loads((ROOT/'docs/experiments/SERVING_R13/builds_a.json').read_text())
    donor=next(j for j in jobs if j['cell']==f'{model}_B{batch}' and j['label']=='N-R12b-noWD')
    target=Path(data['runtime_target']);saved=HERE/'inputs'/f'target_{sha(target)}.json'
    saved.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(target,saved)
    overrides={key:data[key] for key in ('watchdog','pdl','sync','l2_slim','page_loop_split','nonpaged_weight_layout','evict_first','evict_last')}
    overrides.update(deferred_norm=int(data['deferred_norm']),paged_la=int(data['paged_la']),paged_la_splitk=int(data['paged_la_splitk']),kphase_mask=31,v3_poll_ns=0)
    original={phase:{'path':selected[phase],'sha256':sha(selected[phase]),
              'generated_source_sha256':sha(selected[phase]+'.cu') if Path(selected[phase]+'.cu').exists() else None}
              for phase in ('prefill','decode')}
    write(HERE/'inputs'/f'{model}_B{batch}_R13D.json',dict(binaries=original,manifest=data,
        source_provenance='R13 final configuration; historical HEAD not retroactively invented'))
    job=dict(cell=f'{model}_B{batch}',batch=batch,label=label,manifest=str(manifest),classes=str(so)+'.classes.tsv',
             export=donor['export'],target=str(saved),overrides=overrides,out=str(out))
    return job,selected,config

def main():
    base=HERE/'raw/P0_build/qwen3_B16/base'
    job,selected,config=job_for('qwen3',16,'base',base);jobs=[job];arms=[]
    for label,diagnostic in [('base',0),('timer_only',1),('stores_only',2),('full_trace',3)]:
        folder=base if label=='base' else base.parent/label
        if diagnostic:jobs.append(dict(label=label,cell='qwen3_B16',base_so=str(base/'plan.so'),out=str(folder),defines={
            'TILEMEGA_TRACE_STAGE':1,'TILEMEGA_TRACE_STEP':0,'TILEMEGA_TRACE_STAGE_DIAGNOSTIC':diagnostic}))
        arms.append(dict(label=label,kind='tm',root=str(ROOT),python=PYTHON,model='qwen3',model_path=config['model']['path'],batch=16,
             prefill=selected['prefill'],decode=str(folder/'plan.so'),mode='L1',decode_loop=0,prefill_mode='L1',
             env={},binaries={'prefill':{'sha256':sha(selected['prefill'])}}))
    write(HERE/'phase0_builds.json',jobs);write(HERE/'phase0_arms.json',{'qwen3_B16':arms})
    env={'PYTHONPATH':str(ROOT/'python'),'TILEMEGA_BIN':str(ROOT/'build-phase12/tools/tilemega')}
    steps=[dict(name='P0_build',gpu=False,priority=1,cwd=str(ROOT),env=env,timeout_s=7200,
        command=['flock','/root/r14_work/gpu.lock',PYTHON,str(HERE/'builds_r14.py'),'--jobs',str(HERE/'phase0_builds.json'),
                 '--out',str(HERE/'raw/P0_build/results.json')])]
    steps.append(dict(name='P0_smoke',gpu=True,priority=2,cwd=str(ROOT),env=env,timeout_s=900,needs_free_mib=12288,after=['P0_build'],
        command=[PYTHON,'-m','tilemega.serving.smoke','--so',str(base/'plan.so'),'--model',config['model']['path'],'--batch','16',
                 '--steps','64','--out',str(HERE/'raw/P0_smoke')]))
    for round in range(3):
        steps.append(dict(name=f'P0_trace_round{round}',gpu=True,priority=3,cwd=str(ROOT),env=env,timeout_s=3600,needs_free_mib=12288,
            after=['P0_smoke'],command=[PYTHON,str(HERE/'anchor.py'),'--arms',str(HERE/'phase0_arms.json'),'--cell','qwen3_B16',
                       '--round',str(round),'--out',str(HERE/f'raw/P0_trace_round{round}')]))
    steps.append(dict(name='P0_trace_analyze',gpu=False,priority=4,cwd=str(ROOT),env=env,timeout_s=300,
        after=[f'P0_trace_round{r}' for r in range(3)],command=[PYTHON,str(HERE/'ledger_r14.py'),
            '--diagnostic-root',str(HERE),'--out',str(HERE/'results/T2_diagnostic.json')]))
    write(HERE/'queue/queue_phase0.json',steps)
if __name__=='__main__':main()
