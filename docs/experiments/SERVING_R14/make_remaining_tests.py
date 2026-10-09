#!/usr/bin/env python3
"""Publish an isolated R14 continuation without reusing retired choices."""
import copy,json,math,os
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write
from choose_r14 import CELLS
from tilemega.build.identity import sha,verify

INPUTS=HERE/'inputs/remaining'
QUEUE=HERE/'queue_remaining/queue_r14_remaining.json'
STATE=HERE/'scheduler_remaining'
LOCK='/root/r14_work/gpu.lock'
PREFIX='R14_remaining_'

def definitions(total_mib):
    rows=[];driver=HERE/'remaining_tests.py';d_driver=HERE/'phase_d_r14.py'
    env=dict(PYTHONPATH=str(ROOT/'python'),TILEMEGA_BIN=str(ROOT/'build-phase12/tools/tilemega'),
             PATH='/usr/local/cuda/bin:'+os.environ['PATH'])
    def name(s):return PREFIX+s
    def add(s,command,after=(),gpu=False,priority=0,timeout=3600,free=12288,after_any=(),retry_args=(),lock=False):
        own=dict(env)
        if lock:
            command=['flock',LOCK]+command;own['TILEMEGA_GPU_LOCK_HELD']='1'
        rows.append(dict(name=name(s),command=list(map(str,command)),cwd=str(ROOT),env=own,
            after=[name(x) for x in after],after_any=[name(x) for x in after_any],gpu=gpu,
            priority=priority,timeout_s=timeout,needs_free_mib=free,retry_args=list(retry_args)))
    def action(s,which,after=(),args=(),**kw):add(s,[PYTHON,driver,which,*args],after,**kw)
    def phase(s,which,after=(),model=None,**kw):
        cmd=[PYTHON,d_driver,which,'--tag','_remaining','--run-prefix','r14-remaining','--config-dir',INPUTS]
        if model:cmd+=['--model',model]
        add(s,cmd,after,**kw)
    action('preflight','preflight',timeout=600)
    phase('prepare','prepare',['preflight'],lock=True,timeout=7200)
    required=['preflight']
    for model in ('llama','qwen3'):
        for suffix in ('tn8_row','tn16_tiled'):
            label=f'{model}_{suffix}';s='narrow_'+label
            action(s,'narrow',['preflight'],args=['--label',label],gpu=True,priority=1,timeout=9000)
            required.append(s)
    action('trace','trace',['preflight'],gpu=True,priority=2,timeout=3600)
    required.append('trace')
    for r in range(3):
        s=f'overhead_r{r}'
        add(s,[PYTHON,HERE/'anchor.py','--arms',INPUTS/'trace_arms.json','--cell','llama_B1','--round',r,
            '--out',HERE/f'raw/{name(s)}'],['preflight'],gpu=True,priority=3,timeout=2400)
    action('overhead','overhead',[f'overhead_r{r}' for r in range(3)],timeout=300)
    required.append('overhead')
    phase('calibrate','calibrate',['prepare'],gpu=True,priority=5,timeout=10800,free=4096)
    terminals=list(required)
    # Model dependencies are separate: a failed build cannot skip the other model.
    for index,model in enumerate(('llama','qwen3')):
        phase('build_'+model,'build',['calibrate'],model=model,gpu=True,priority=10+index,timeout=47000)
        phase('smoke_'+model,'smoke',['build_'+model],model=model,gpu=True,priority=12,timeout=1800)
        phase('family_'+model,'family',['smoke_'+model],model=model,timeout=600)
        action('register_'+model,'register',['smoke_'+model,'family_'+model],args=['--model',model],timeout=600)
        terminals.append('register_'+model)
    for cell in CELLS:
        model=cell.split('_')[0];registered='register_'+model;checks=[]
        for r in range(3):
            s=f'd2_{cell}_r{r}';v=f'validate_{cell}_r{r}'
            add(s,[PYTHON,HERE/'anchor.py','--arms',INPUTS/f'arms_{cell}.json','--cell',cell,'--round',r,
                '--out',HERE/f'raw/{name(s)}'],[registered],gpu=True,priority=20+(5 if cell.endswith('B16') else 0),
                timeout=3600,free=math.ceil(.85*total_mib)+1024)
            action(v,'validate',[s],args=['--cell',cell,'--round',r],timeout=300)
            checks.append(v)
        canary='canary_'+cell;replay='replay_'+cell;c1='c1_'+cell;c2='c2_'+cell;protocol='protocol_'+cell
        action(canary,'canary',checks,args=['--cell',cell],timeout=300)
        action(replay,'replay',[canary],args=['--cell',cell],gpu=True,priority=30,timeout=10800,
               free=math.ceil(.85*total_mib)+1024)
        action(c1,'c1',[replay],args=['--cell',cell],gpu=True,priority=35,timeout=12000)
        action(c2,'c2',[registered],args=['--cell',cell],gpu=True,priority=36,timeout=9000)
        action(protocol,'protocol',[c2],args=['--cell',cell],gpu=True,priority=40,timeout=12000,retry_args=['--resume'])
        terminals += [c1,c2,protocol,replay]
    action('analyze','analyze',after_any=terminals,timeout=600,priority=100)
    return rows

def main():
    if (STATE/'state.json').exists():raise SystemExit('continuation already initialized; do not overwrite its inputs')
    baselines=json.loads((HERE/'logic_completion_arms.json').read_text())
    paths=[HERE/'guard_policy.json',HERE/'logic_completion_arms.json',HERE/'logic_completion_builds.json',
           HERE/'raw/logic_completion/reviewed_acceptance.json',ROOT/'build-phase12/tools/tilemega']
    paths += [HERE/name for name in ('make_remaining_tests.py','remaining_tests.py','phase_d_r14.py',
              'anchor.py','scheduler.py','gpu_guard.py','choose_r14.py','test_remaining_tests.py')]
    for model in ('llama','qwen3'):
        cfg=json.loads((ROOT/f'configs/e2e/{model}_r14.json').read_text())
        cfg['output']['dir']=str(ROOT/f'runs/r14-remaining-{model}')
        path=INPUTS/f'{model}_r14.json';write(path,cfg);paths.append(path)
    trace=[]
    for label in ('baseline','stage','task'):
        arm=copy.deepcopy(baselines['llama_B1']);arm['label']=label
        arm['decode']=str(HERE/f'raw/logic_completion/build/llama_B1/{label}/plan.so')
        arm.update(mode='L1',decode_loop=False,prefill_mode='L1')
        arm['binaries']={k:dict(sha256=sha(arm[k])) for k in ('prefill','decode')}
        verify(arm['decode']);trace.append(arm)
    path=INPUTS/'trace_arms.json';write(path,dict(llama_B1=trace));paths.append(path)
    legacy_prefill=[]
    for arm in baselines.values():
        verify(arm['decode'])
        for key in ('decode','prefill'):
            paths.append(Path(arm[key]));identity=Path(arm[key]+'.identity.json')
            if identity.exists():paths.append(identity)
            elif key=='prefill':
                paths.append(Path(arm[key]+'.plan.json'));legacy_prefill.append(arm[key])
            else:raise ValueError('new control lacks artifact identity')
    for job in json.loads((HERE/'logic_completion_builds.json').read_text()):
        so=Path(job['out'])/'plan.so';verify(so)
        paths += [so,Path(str(so)+'.identity.json')]
    path=INPUTS/'final_template.json'
    write(path,dict(cells=list(CELLS),labels=['vllm','R13D','baseline','R14F'],rounds=3,new_tokens=1024,
        prefill='same fresh D1 prefill per cell; L1',winner='fresh r14-remaining plans.json and serving sidecar',
        correctness=dict(c1='unchanged HF teacher-forced thresholds',c2='four-arm 1024-step check',fresh_processes=50),
        canary=dict(threshold=.02,replays_per_marked_round=1),guard='unchanged guard_policy.json'))
    paths.append(path)
    recorded=[json.loads(line) for line in (HERE/'raw/D0/guard.jsonl').read_text().splitlines()]
    totals={r['total_mib'] for r in recorded if r.get('phase')=='preflight'}
    if len(totals)!=1:raise ValueError('recorded device memory is ambiguous')
    total=totals.pop();rows=definitions(total);write(QUEUE,rows)
    paths.append(QUEUE)
    write(INPUTS/'definition.json',dict(sha256={str(p):sha(p) for p in sorted(set(paths))},
        steps=len(rows),recorded_total_mib=total,queue=str(QUEUE),state=str(STATE),lock=LOCK,
        legacy_prefill_sha_bound=sorted(set(legacy_prefill)),
        retains='completed Phase 0/A/B/C and accepted 20-check logic review',
        generated_after_build='only binary paths, identities and protocol cases; arm templates/rules are frozen above'))
    print(f'{len(rows)} isolated remaining steps prepared; scheduler not started')

if __name__=='__main__':main()
