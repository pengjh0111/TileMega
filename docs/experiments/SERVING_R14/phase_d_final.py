#!/usr/bin/env python3
"""Final R14 identity checks, correctness and preregistered default selection."""
import argparse,copy,json,shutil,statistics
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write
from phase_a import run,timed_tokens
from choose_r14 import CELLS,final
from tilemega.build.identity import verify,sha
from tilemega.serving.execution import write_execution

def arms():return json.loads((HERE/'phase_d_final_arms.json').read_text())
def observations(cell):
    return [json.loads((HERE/f'raw/D2_{cell}_r{r}/{cell}/round{r}.json').read_text()) for r in range(3)]
def outliers(samples):
    if len(samples)!=3 or any(value<=0 for value in samples):raise ValueError('canary requires three positive samples')
    median=statistics.median(samples)
    return [r for r,value in enumerate(samples) if abs(value/median-1)>.02]

def reviewed_choice(choice,recheck):
    if not recheck.get('accepted') or recheck.get('canary_rounds'):raise ValueError('recollection is not accepted')
    result=copy.deepcopy(choice);matches=0
    for candidate in result['candidates']:
        matching=[r for r in recheck['candidates'] if (r['library'],r['mode'],r['loop'])==
                  (candidate.get('library'),candidate['mode'],candidate['loop'])]
        if not matching:continue
        if len(matching)!=1:raise ValueError('ambiguous finalist recollection')
        row=matching[0]
        if len(row['samples_ms'])!=3 or outliers(row['samples_ms']):raise ValueError('unstable finalist recollection')
        matches+=1;candidate['original_confirmation_samples_ms']=candidate['samples_ms']
        candidate['original_confirmation_measurements']=candidate.get('measurements',[])
        candidate['samples_ms']=row['samples_ms'];candidate['median_ms']=statistics.median(row['samples_ms'])
        candidate['measurements']=[dict(o,round=f'reconfirm{o["round"]}',measurement_path=o['path'],spill=o['execution_identity']['spill']) for o in row['measurements']]
    if matches!=3 or len(recheck['candidates'])!=3:raise ValueError('recollection must cover all three finalists')
    confirmed=[r for r in result['candidates'] if len(r.get('samples_ms',[]))==3 and not r.get('error')]
    result['selected']=min(confirmed,key=lambda r:statistics.median(r['samples_ms']))
    result['confirmation_recheck']=dict(accepted=True,recollection_count=1,source=str(HERE/'raw/D1_reconfirm_llama_B16_v2/result.json'))
    return result

def prepare():
    destination=HERE/'raw/D_final_definition';destination.mkdir(parents=True,exist_ok=True)
    if (destination/'llama_plans_before.json').exists():raise ValueError('final definitions already prepared; preserve the original snapshot')
    reference=json.loads((HERE/'phase_a_arms.json').read_text());baselines=json.loads((HERE/'phase_d_baseline_arms_v2.json').read_text());registered={};states=set();coverage=[]
    for model in ('llama','qwen3'):
        path=ROOT/f'runs/r14-{model}/plans.json';plans=json.loads(path.read_text())
        shutil.copyfile(path,destination/f'{model}_plans_before.json')
        if model=='llama':
            plans['16']['decode_pg_choice']=reviewed_choice(plans['16']['decode_pg_choice'],json.loads((HERE/'raw/D1_reconfirm_llama_B16_v2/result.json').read_text()))
        for batch,plan in plans.items():
            cell=f'{model}_B{batch}';selected=plan['decode_pg_choice']['selected'];plan['decode']=selected['library']
            plan['serving']=write_execution(plan['decode'],selected['mode'],selected['loop'],'L1',selection=selected,candidates=plan['decode_pg_choice']['candidates'],stage_one=dict(mode='L1',loop=0))
            # Manifest/identity/binary bytes remain immutable; only serving metadata changes.
            base=dict(baselines[cell],prefill=plan['prefill'],label='baseline')
            old=copy.deepcopy(next(a for a in reference[cell] if a['label']=='R13D'));old['prefill']=plan['prefill']
            candidate=dict(base,label='R14F',decode=plan['decode'],mode='auto',decode_loop='auto',prefill_mode='auto',expected_mode=selected['mode'],expected_loop=bool(selected['loop']))
            vllm=copy.deepcopy(next(a for a in reference[cell] if a['label']=='vllm'));vllm.pop('binaries',None)
            for arm in (old,base,candidate):
                arm['binaries']={p:dict(sha256=sha(arm[p])) for p in ('prefill','decode')}
                prefill=verify(arm['prefill']);arm['prefill_identity']=prefill['artifact_id']
                source=prefill['source'];states.add((source['source_digest'],source['compiler_sha256']))
                if arm['label']!='R13D':
                    arm['identity']=verify(arm['decode'])['artifact_id']
                    source=verify(arm['decode'])['source'];states.add((source['source_digest'],source['compiler_sha256']))
            registered[cell]=[vllm,old,base,candidate]
            for label,arm in (('R14F',candidate),):
                if verify(arm['decode'])['plan']['pg']!='pages':raise ValueError('this final protocol requires a paged plan')
                case=dict(model=arm['model_path'],batch=int(batch),steps=64,prompt_ids=str(ROOT/f'docs/experiments/SERVING_R10/prompts/{model}_ids.json'),prefill=arm['prefill'],decode=arm['decode'],reference_prefill=arm['prefill'],reference_decode=arm['decode'],binary_sha256={p:sha(arm[p]) for p in ('prefill','decode')},arms=[
                    dict(label='L1_separate',prefill=arm['prefill'],decode=arm['decode'],modes=['L1'],decode_loop=False),
                    dict(label='L2_separate',prefill=arm['prefill'],decode=arm['decode'],modes=['L2'],decode_loop=False)])
                if label=='R14F':
                    case['arms'] += [dict(label='L2_loop',prefill=arm['prefill'],decode=arm['decode'],modes=['L2'],decode_loop=True,require_loop=True),dict(label='L2_loop_no_phase',prefill=arm['prefill'],decode=arm['decode'],modes=['L2'],decode_loop=True,require_loop=True,env={'TILEMEGA_KPHASE_MASK':'0'})]
                    write(HERE/f'D3_protocol_{cell}_cases.json',[case]);coverage.append(dict(cell=cell,artifact_id=arm['identity'],fresh_processes=50,reason='Final paged execution and phase paths not covered by nonpaged B6 protocols'))
            for label,arm in (('R14F',candidate),('baseline',base)):
                write(destination/f'{cell}_{label}_identity.json',verify(arm['decode']))
            write(destination/f'{cell}_serving.json',plan['serving'])
        write(path,plans);write(destination/f'{model}_plans_reviewed.json',plans)
    if len(states)!=1:raise ValueError('rebuilt control/candidate compiler-source states differ')
    write(HERE/'phase_d_final_arms.json',registered);write(destination/'protocol_coverage.json',coverage)
    print('Four final comparison cells and explicit 50-process protocol cases pinned')

def validate_round(cell,round_index):
    path=HERE/f'raw/D2_{cell}_r{round_index}/{cell}/round{round_index}.json';result=json.loads(path.read_text())
    for arm in arms()[cell]:
        row=result['arms'][arm['label']]
        if row.get('exit_code')!=0:raise ValueError(f'{cell}/{arm["label"]} failed: '+str(row.get('exit_code')))
        if arm['label'] in ('baseline','R14F'):
            identity=row['execution_identity'];current=verify(arm['decode'])
            if identity['artifact_id']!=current['artifact_id'] or identity['trace']:raise ValueError('performance identity mismatch')
            mode=arm.get('expected_mode',arm['mode']);loop=arm.get('expected_loop',bool(arm['decode_loop']))
            if identity['executor']!=mode or identity['loop']!=loop or row['decode_loop_used']!=loop:raise ValueError('actual execution differs from registered selection')
            prefill=row['prefill_execution_identity']
            if prefill['executor']!='L1' or prefill['artifact_id']!=arm['prefill_identity']:raise ValueError('prefill executor or artifact changed')
    write(path.parent/'validated.json',dict(pass_=True,round=round_index,cell=cell))

def canary(cell):
    records=observations(cell);values=[r['arms']['vllm']['tpot_seconds'] for r in records]
    marked=outliers(values)
    write(HERE/f'raw/D2_canary_{cell}/result.json',dict(cell=cell,metric='tpot_seconds',samples_seconds=values,marked_rounds=marked,repeat_limit=1,status='requires_one_bounded_replay' if marked else 'accepted'))
    print(cell+' canary '+('marked '+str(marked) if marked else 'accepted'))

def c2(cell):
    for arm in arms()[cell]:
        if arm['label'] not in ('baseline','R14F'):continue
        out=HERE/f'raw/D3_c2_{cell}'/arm['label']
        run([PYTHON,'-m','tilemega.serving.check_modes','--model',arm['model_path'],'--prefill-so',arm['prefill'],'--decode-so',arm['decode'],'--prompt-ids',ROOT/f'docs/experiments/SERVING_R10/prompts/{arm["model"]}_ids.json','--batch',arm['batch'],'--steps','1024','--out',out],out/'run.log',2400)
        if not json.loads((out/'mode_check.json').read_text())['pass']:raise ValueError('C-2 failed')

def c1(cell):
    records=observations(cell);out=HERE/f'raw/D3_c1_{cell}';rows={}
    vm=records[0]['arms']['vllm'];vroot=Path(vm['out'])/f'B{vm["batch"]}'
    generated=[r for r in vm['generation_runs'] if r['N']==1024 and not r['warmup']]
    if len(generated)!=1:raise ValueError('missing timed vLLM tokens')
    vtokens=vroot/generated[0]['tokens_file'];model=next(a['model_path'] for a in arms()[cell] if a['label']=='R14F')
    common=[PYTHON,'-m','tilemega.serving.hf_check','--model',model,'--prompt-ids',ROOT/f'docs/experiments/SERVING_R10/prompts/{cell.split("_B")[0]}_ids.json','--skip-free-greedy']
    run(common+['--generated',vtokens,'--out',out/'vllm_hf.json'],out/'vllm_hf.log',2400,allowed=(0,1))
    for label in ('baseline','R14F'):
        tokens=[timed_tokens(r['arms'][label]) for r in records]
        repeated=all(t==tokens[0] for t in tokens)
        if not repeated:raise ValueError('timed tokens differ between rounds')
        write(out/(label+'_tokens.json'),tokens[0])
        run(common+['--generated',out/(label+'_tokens.json'),'--vllm-metrics',out/'vllm_hf.json','--out',out/(label+'_hf.json')],out/(label+'_hf.log'),2400)
        rows[label]=dict(timed_tokens_identical=True,c1=True)
    write(out/'result.json',rows)

def analyze():
    pending=[];cells={}
    for cell in CELLS:
        marked=json.loads((HERE/f'raw/D2_canary_{cell}/result.json').read_text())['marked_rounds']
        if marked:pending.append(dict(cell=cell,rounds=marked))
        records=observations(cell);row={}
        for label in ('baseline','R14F'):
            c1_record=json.loads((HERE/f'raw/D3_c1_{cell}/result.json').read_text())[label]
            c2_record=json.loads((HERE/f'raw/D3_c2_{cell}/{label}/mode_check.json').read_text())
            row[label]=dict(c1=c1_record['c1'],c2=c2_record['pass'],tpot=[r['arms'][label]['tpot_seconds'] for r in records],paired_vllm_ratios=[r['arms']['vllm']['e2e_seconds']/r['arms'][label]['e2e_seconds'] for r in records])
        protocol=json.loads((HERE/f'raw/D3_protocol_{cell}/summary.json').read_text())
        if not protocol['complete'] or protocol['passed']!=50 or protocol['failed']:raise ValueError('final protocol not 50/50')
        cells[cell]=row
    decision=final(cells) if not pending else None
    write(HERE/'results/D_final_acceptance.json',dict(cells=cells,pending_canary_replays=pending,selection=decision,status='requires_bounded_canary_replay' if pending else 'collected'))
    if decision:
        for model in ('llama','qwen3'):
            cfg=json.loads((ROOT/f'configs/e2e/{model}_r14.json').read_text());cfg['selected_plans']={}
            for batch in (1,16):
                cell=f'{model}_B{batch}';name=decision[cell]['selected'];arm=next(a for a in arms()[cell] if a['label']==name)
                cfg['selected_plans'][str(batch)]={key:arm[key] for key in ('prefill','decode','mode','decode_loop','prefill_mode')}
            write(ROOT/f'configs/e2e/{model}_r14_final.json',cfg)
    print('Final collection '+('needs canary replay' if pending else 'complete; hard gates '+str({c:r['gate_pass'] for c,r in decision.items()})))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','validate_round','canary','c1','c2','analyze'));p.add_argument('--cell',choices=CELLS);p.add_argument('--round',type=int);a=p.parse_args()
    if a.action=='validate_round':validate_round(a.cell,a.round)
    elif a.action in ('canary','c1','c2'):globals()[a.action](a.cell)
    else:globals()[a.action]()
if __name__=='__main__':main()
