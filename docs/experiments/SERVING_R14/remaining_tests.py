#!/usr/bin/env python3
"""Resume frozen R14 measurements using the existing guarded tools."""
import argparse,copy,json,os,statistics
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write
from phase_a import run,timed_tokens
from phase_d_final import outliers
from choose_r14 import CELLS,final
from tilemega.build.identity import verify,sha

PREFIX='R14_remaining'
INPUTS=HERE/'inputs/remaining'
def folder(name):return HERE/'raw'/(PREFIX+'_'+name)
def load(path):return json.loads(Path(path).read_text())
def arms(cell):return load(INPUTS/f'arms_{cell}.json')[cell]
def baseline_arms():return load(HERE/'logic_completion_arms.json')
def run_dir(model):return ROOT/f'runs/r14-remaining-{model}'

def preflight():
    record=load(INPUTS/'definition.json')
    for path,expected in record['sha256'].items():
        if sha(path)!=expected:raise ValueError('frozen input changed: '+path)
    checked=load(HERE/'raw/logic_completion/reviewed_acceptance.json')
    if not checked['pass_']:raise ValueError('logic completion is not accepted')
    for arm in baseline_arms().values():verify(arm['decode'])
    run([PYTHON,'python/tilemega/fingerprint.py','--check',os.environ['TILEMEGA_BIN']],folder('preflight')/'fingerprint.log',300)
    write(folder('preflight')/'result.json',dict(pass_=True,prior_logic_checks=len(checked['checks'])))

def narrow(label):
    job=next(j for j in load(HERE/'logic_completion_builds.json') if j['label']==label)
    arm=baseline_arms()[job['cell']];out=folder('narrow_'+label)
    so=str(Path(job['out'])/'plan.so');verify(so)
    prompts=ROOT/f'docs/experiments/SERVING_R10/prompts/{arm["model"]}_ids.json'
    run([PYTHON,'-m','tilemega.serving.check_modes','--model',arm['model_path'],'--prefill-so',arm['prefill'],
         '--decode-so',so,'--prompt-ids',prompts,'--batch','1','--steps','1024','--out',out/'modes'],out/'modes.log',3600)
    run([PYTHON,'-m','tilemega.serving.hf_check','--model',arm['model_path'],'--prompt-ids',prompts,
         '--generated',out/'modes/tokens_L1_separate.json','--vllm-metrics',HERE/f'raw/A3_{job["cell"]}/vllm_hf.json',
         '--skip-free-greedy','--out',out/'hf.json'],out/'hf.log',3600)
    write(out/'identity.json',verify(so))

def trace():
    from ledger_r14 import tasks
    arm=baseline_arms()['llama_B1']
    for kind in ('stage','task'):
        so=str(HERE/f'raw/logic_completion/build/llama_B1/{kind}/plan.so');out=folder('trace')/kind
        run([PYTHON,'-m','tilemega.serving.trace','--model',arm['model_path'],'--prefill-so',arm['prefill'],
             '--decode-so',so,'--batch','1','--mode','L1','--decode-loop','0','--'+kind,
             '--past-list','64,575,1000','--launches','16','--out',out],out/'run.log',1800)
        if kind=='task':
            for p in out.glob('past*'):
                values=tasks(p/'task_profile.tsv')
                if not values.get('gemm') or any(not r['operand_ready_observed'] for r in values['gemm']['samples'] if r['bytes']):
                    raise ValueError('missing first-ready trace observation')
                write(p/'summary.json',values)

def overhead():
    rows={}
    for r in range(3):
        result=load(folder('overhead_r'+str(r))/f'llama_B1/round{r}.json')
        for label,row in result['arms'].items():
            if row.get('exit_code')!=0:raise ValueError('trace overhead arm failed: '+label)
            identity=row['execution_identity']
            if identity['trace']!=(label!='baseline'):raise ValueError('trace role mismatch')
            sample=rows.setdefault(label,dict(execution_id=identity['execution_id'],samples_ms=[]))
            if sample['execution_id']!=identity['execution_id']:raise ValueError('trace artifact changed')
            sample['samples_ms'].append((row['e2e_seconds']-row['ttft_seconds'])*1000/1023)
    base=statistics.median(rows['baseline']['samples_ms'])
    for row in rows.values():
        row['median_ms']=statistics.median(row['samples_ms']);row['relative']=row['median_ms']/base-1
    write(folder('overhead')/'result.json',dict(arms=rows,within_two_percent={k:abs(v['relative'])<=.02 for k,v in rows.items()},
          semantics='trace timing is diagnostic; failure of the overhead target does not invalidate independent performance collection'))

def register(model):
    plans=load(run_dir(model)/'plans.json');reference=load(HERE/'phase_a_arms.json')
    for batch,plan in plans.items():
        cell=f'{model}_B{batch}';selected=plan['decode_pg_choice']['selected']
        candidate=verify(plan['decode']);prefill=verify(plan['prefill'])
        base=copy.deepcopy(baseline_arms()[cell]);base.update(prefill=plan['prefill'],label='baseline')
        control=verify(base['decode'])
        states={(i['source']['source_digest'],i['source']['compiler_sha256']) for i in (candidate,prefill,control)}
        if len(states)!=1:raise ValueError('control/candidate source/compiler state differs')
        old=copy.deepcopy(next(a for a in reference[cell] if a['label']=='R13D'));old['prefill']=plan['prefill']
        vm=copy.deepcopy(next(a for a in reference[cell] if a['label']=='vllm'));vm.pop('binaries',None)
        winner=dict(base,label='R14F',decode=plan['decode'],mode='auto',decode_loop='auto',prefill_mode='auto',
                    expected_mode=selected['mode'],expected_loop=bool(selected['loop']))
        for arm in (old,base,winner):
            arm['binaries']={key:dict(sha256=sha(arm[key])) for key in ('prefill','decode')}
            arm['prefill_identity']=prefill['artifact_id']
            if arm['label']!='R13D':arm['identity']=verify(arm['decode'])['artifact_id']
        write(INPUTS/f'arms_{cell}.json',{cell:[vm,old,base,winner]})
        common=dict(prefill=winner['prefill'],decode=winner['decode'])
        protocol=[dict(label='L1_separate',modes=['L1'],decode_loop=False,**common),
                  dict(label='L2_separate',modes=['L2'],decode_loop=False,**common)]
        mode='L2' if candidate['plan']['pg']=='pages' else 'L1'
        protocol.append(dict(label=mode+'_loop',modes=[mode],decode_loop=True,require_loop=True,**common))
        if mode=='L2':protocol.append(dict(label='L2_loop_no_phase',modes=['L2'],decode_loop=True,require_loop=True,env={'TILEMEGA_KPHASE_MASK':'0'},**common))
        case=dict(model=winner['model_path'],batch=int(batch),steps=64,
                  prompt_ids=str(ROOT/f'docs/experiments/SERVING_R10/prompts/{model}_ids.json'),
                  reference_prefill=winner['prefill'],reference_decode=winner['decode'],
                  binary_sha256={k:sha(v) for k,v in common.items()},arms=protocol,**common)
        write(INPUTS/f'protocol_{cell}.json',[case])
        write(folder('register_'+model)/(cell+'_identity.json'),candidate)
    write(folder('register_'+model)/'result.json',dict(cells=[f'{model}_B{b}' for b in plans],pass_=True))

def round_record(cell,r,replay=False):
    return folder(('replay_' if replay else 'd2_')+cell+(('' if replay else '_r'+str(r))))/cell/f'round{r}.json'

def validate(cell,r,replay=False):
    path=round_record(cell,r,replay);result=load(path)
    for arm in arms(cell):
        row=result['arms'][arm['label']]
        if row.get('exit_code')!=0:raise ValueError(f'{cell}/{arm["label"]}: failed arm')
        if arm['label'] in ('baseline','R14F'):
            identity=row['execution_identity'];verify(arm['decode'])
            mode=arm.get('expected_mode',arm['mode']);loop=arm.get('expected_loop',bool(arm['decode_loop']))
            if identity['artifact_id']!=arm['identity'] or identity['trace']:raise ValueError('performance identity differs')
            if identity['executor']!=mode or identity['loop']!=loop or row['decode_loop_used']!=loop:raise ValueError('execution differs from requested selection')
            pre=row['prefill_execution_identity']
            if pre['artifact_id']!=arm['prefill_identity'] or pre['executor']!='L1':raise ValueError('prefill identity/mode differs')
    write(path.parent/('validated_replay.json' if replay else 'validated.json'),dict(pass_=True))

def canary(cell):
    records=[load(round_record(cell,r)) for r in range(3)]
    samples=[r['arms']['vllm']['tpot_seconds'] for r in records]
    write(folder('canary_'+cell)/'result.json',dict(samples=samples,marked_rounds=outliers(samples),replay_limit=1))

def replay(cell):
    marked=load(folder('canary_'+cell)/'result.json')['marked_rounds']
    for r in marked:
        out=folder('replay_'+cell)
        run([PYTHON,HERE/'anchor.py','--arms',INPUTS/f'arms_{cell}.json','--cell',cell,'--round',r,'--out',out],out/f'round{r}.log',3600)
        validate(cell,r,True)
    records=[load(round_record(cell,r,r in marked)) for r in range(3)]
    remaining=outliers([v['arms']['vllm']['tpot_seconds'] for v in records])
    write(folder('replay_'+cell)/'result.json',dict(replayed=marked,remaining_canary_rounds=remaining,
          round_paths=[str(round_record(cell,r,r in marked)) for r in range(3)]))

def observations(cell):return [load(p) for p in load(folder('replay_'+cell)/'result.json')['round_paths']]

def c1(cell):
    records=observations(cell);out=folder('c1_'+cell);vm=records[0]['arms']['vllm']
    generated=[r for r in vm['generation_runs'] if not r['warmup'] and r['N']==1024]
    if len(generated)!=1:raise ValueError('missing timed vLLM tokens')
    tokens=Path(vm['out'])/f'B{vm["batch"]}'/generated[0]['tokens_file']
    arm=next(a for a in arms(cell) if a['label']=='R14F')
    cmd=[PYTHON,'-m','tilemega.serving.hf_check','--model',arm['model_path'],'--prompt-ids',
         ROOT/f'docs/experiments/SERVING_R10/prompts/{arm["model"]}_ids.json','--skip-free-greedy']
    run(cmd+['--generated',tokens,'--out',out/'vllm_hf.json'],out/'vllm.log',3600,allowed=(0,1))
    rows={}
    for label in ('baseline','R14F'):
        values=[timed_tokens(r['arms'][label]) for r in records]
        if any(v!=values[0] for v in values):raise ValueError('timed tokens differ between rounds')
        path=out/(label+'_tokens.json');write(path,values[0])
        run(cmd+['--generated',path,'--vllm-metrics',out/'vllm_hf.json','--out',out/(label+'_hf.json')],out/(label+'.log'),3600)
        rows[label]=dict(c1=True,timed_tokens_identical=True)
    write(out/'result.json',rows)

def c2(cell):
    for arm in arms(cell):
        if arm['label'] not in ('baseline','R14F'):continue
        out=folder('c2_'+cell)/arm['label']
        run([PYTHON,'-m','tilemega.serving.check_modes','--model',arm['model_path'],'--prefill-so',arm['prefill'],
             '--decode-so',arm['decode'],'--prompt-ids',ROOT/f'docs/experiments/SERVING_R10/prompts/{arm["model"]}_ids.json',
             '--batch',arm['batch'],'--steps','1024','--out',out],out/'run.log',3600)

def protocol(cell,resume):
    out=folder('protocol_'+cell)
    cmd=[PYTHON,ROOT/'docs/experiments/SERVING_R11/check_protocol.py','--cases',INPUTS/f'protocol_{cell}.json',
         '--processes','50','--out',out]
    run(cmd+(['--resume'] if resume else []),out/'driver.log',10800)

def analyze():
    cells={};missing=[];canaries=[];families=[]
    state=load(HERE/'scheduler_remaining/state.json')
    failures={k:v['status'] for k,v in state.items() if v['status'] in ('failed','skipped','not_run')}
    for cell in CELLS:
        try:
            records=observations(cell);rows={}
            for label in ('baseline','R14F'):
                correct=load(folder('c1_'+cell)/'result.json')[label]
                modes=load(folder('c2_'+cell)/label/'mode_check.json')
                rows[label]=dict(c1=correct['c1'],c2=modes['pass'],
                    tpot=[(r['arms'][label]['e2e_seconds']-r['arms'][label]['ttft_seconds'])/1023 for r in records],
                    paired_vllm_ratios=[r['arms']['vllm']['e2e_seconds']/r['arms'][label]['e2e_seconds'] for r in records])
            protocol=load(folder('protocol_'+cell)/'summary.json')
            if not protocol['complete'] or protocol['passed']!=50 or protocol['failed']:raise ValueError('protocol not 50/50')
            marked=load(folder('replay_'+cell)/'result.json')['remaining_canary_rounds']
            if marked:canaries.append(dict(cell=cell,rounds=marked))
            cells[cell]=rows
        except (OSError,KeyError,ValueError) as error:missing.append(dict(cell=cell,error=str(error)))
    for model in ('llama','qwen3'):
        path=HERE/f'results/D1_planfamily_{model}_remaining.json'
        if path.exists():families+=load(path)['triggered']
        else:missing.append(dict(model=model,error='missing PlanFamily audit'))
    decision=final(cells) if not missing and not canaries and not families and not failures else None
    write(HERE/'results/remaining_acceptance.json',dict(cells=cells,missing=missing,pending_canaries=canaries,
          pending_planfamily=families,failed_steps=failures,selection=decision,status='collected' if decision else 'requires_review'))
    if decision:
        for model in ('llama','qwen3'):
            cfg=load(INPUTS/f'{model}_r14.json');cfg['selected_plans']={}
            for b in (1,16):
                cell=f'{model}_B{b}';arm=next(a for a in arms(cell) if a['label']==decision[cell]['selected'])
                cfg['selected_plans'][str(b)]={k:arm[k] for k in ('prefill','decode','mode','decode_loop','prefill_mode')}
            write(ROOT/f'configs/e2e/{model}_r14_final.json',cfg)
    print('R14 remaining collection: '+('complete; review hard gates' if decision else 'requires review'))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('preflight','narrow','trace','overhead','register','validate','canary','replay','c1','c2','protocol','analyze'))
    p.add_argument('--model',choices=('llama','qwen3'));p.add_argument('--cell',choices=CELLS);p.add_argument('--round',type=int);p.add_argument('--label');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='narrow':narrow(a.label)
    elif a.action=='register':register(a.model)
    elif a.action=='validate':validate(a.cell,a.round)
    elif a.action=='protocol':protocol(a.cell,a.resume)
    elif a.action in ('canary','replay','c1','c2'):globals()[a.action](a.cell)
    else:globals()[a.action]()
if __name__=='__main__':main()
