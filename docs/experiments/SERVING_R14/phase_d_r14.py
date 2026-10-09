#!/usr/bin/env python3
"""Frozen R14 calibration, integral selection and conditional-family audit."""
import argparse,csv,json,os,shutil,statistics,subprocess
from pathlib import Path
from make_phase0 import HERE,ROOT,PYTHON,write
from phase_a import run
from tilemega.cli import Run,read_config
from tilemega.build.identity import verify,sha
from tilemega.serving.integrated_selection import integrated_ms,PASTS

TAG=''
def evidence_path(name):return HERE/'raw'/(name+TAG)
def config(model):return ROOT/f'configs/e2e/{model}_r14.json'
def prepare():
    out=evidence_path('Dpre')
    run(['cmake','--build','build-phase12','--target','tilemega','tilemega-loadbench','tilemega-unit','-j','6'],out/'build.log',6000)
    run(['ctest','--test-dir','build-phase12','-R','^(serving_search_rejection|skeleton_search_isolation|stage_flow)$','--output-on-failure'],out/'solver_regression.log',600)
    run([PYTHON,'python/tilemega/fingerprint.py','--check','build-phase12/tools/tilemega'],out/'fingerprint.log')
    for script in ('test/python/integrated_selection.py','test/python/test_attention_selection.py','test/python/serving_identity.py','docs/experiments/SERVING_R14/test_selection.py','docs/experiments/SERVING_R14/test_phase_d.py'):
        run([PYTHON,script],out/(Path(script).stem+'.log'))
    binary=next(p for p in (ROOT/'build-phase12/tilemega-loadbench',ROOT/'build-phase12/tools/tilemega-loadbench') if p.exists())
    write(out/'loadbench.json',dict(path=str(binary),sha256=sha(binary),source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=sha(ROOT/'tools/experimental/loadbench/main.cu'),timing=False))

def calibrate():
    out=evidence_path('D0');runner=Run(read_config(config('llama')),os.environ['TILEMEGA_BIN']);runner.calibrate()
    record=json.loads((evidence_path('Dpre')/'loadbench.json').read_text())
    if sha(record['path'])!=record['sha256']:raise ValueError('loading probe changed')
    run([PYTHON,ROOT/'docs/experiments/SERVING_R13/dram_ceiling.py','--binary',record['path'],'--out',out/'ceiling','--target',runner.target,'--target-out',runner.target],out/'ceiling.log',6600)
    # D1 must see the exact calibrated copy, not an editable external target.
    copy=out/'target.json';copy.write_bytes(runner.target.read_bytes())
    write(out/'result.json',dict(target=str(runner.target),target_sha256=sha(runner.target),archived_target=str(copy),loadbench=record))

def build(model):
    out=evidence_path(f'D1_{model}');cfg=read_config(config(model))
    limit=2*len(cfg['workload']['batch'])*cfg['solver']['time_budget_s']+1800
    run([PYTHON,'-m','tilemega','build','--config',config(model),'--run-dir',ROOT/f'runs/r14-{model}'],out/'build.log',limit)
    plans=json.loads((ROOT/f'runs/r14-{model}/plans.json').read_text())
    # Preserve original cache/selection data inside the committed evidence
    # tree, including rejected and budget-eliminated candidates.
    saved=out/'evidence';saved.mkdir(parents=True,exist_ok=True)
    run_dir=ROOT/f'runs/r14-{model}'
    for p in run_dir.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.jsonl','.tsv','.txt'):
            dst=saved/'run'/p.relative_to(run_dir);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    cache=Path(cfg['device']['cache_dir']).expanduser()
    events=json.loads((run_dir/'cache.json').read_text())
    for event in events:
        if event['layer'] not in ('plan','attention_variant'):continue
        folder=cache/'plans'/event['key']
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ('.json','.tsv','.log','.txt','.patch'):
                dst=saved/'plans'/event['key']/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    identities=[]
    for batch,plan in plans.items():
        for phase in ('prefill','decode'):identities.append(dict(batch=batch,phase=phase,library=plan[phase],identity=verify(plan[phase])))
        choice=plan['decode_pg_choice']
        if not choice.get('selected') or not choice.get('candidates'):raise ValueError('integral selection evidence missing')
        winner=choice['selected']
        if len(winner.get('samples_ms',[]))!=3 or len(winner.get('measurements',[]))<3:raise ValueError('missing confirmation rounds')
        for observation in winner['measurements']:
            if set(observation['by_past'])!={str(p) for p in PASTS} or observation['execution_identity']['trace']:raise ValueError('invalid integral measurement')
    write(out/'identities.json',identities)
    write(out/'result.json',dict(model=model,cells=list(plans),pass_=True,plans_sha256=sha(ROOT/f'runs/r14-{model}/plans.json')))

def smoke(model):
    rows=[];plans=json.loads((ROOT/f'runs/r14-{model}/plans.json').read_text());cfg=read_config(config(model))
    for b,plan in plans.items():
        out=evidence_path(f'D1_smoke_{model}')/f'B{b}'
        run([PYTHON,'-m','tilemega.serving.smoke','--so',plan['decode'],'--model',cfg['model']['path'],'--batch',b,'--steps','64','--out',out],out/'run.log',600)
        report=json.loads((out/'smoke.json').read_text())
        if not report['pass']:raise ValueError('selected-plan smoke failed')
        rows.append(dict(batch=b,pass_=True,artifact_id=verify(plan['decode'])['artifact_id']))
    write(evidence_path(f'D1_smoke_{model}')/'results.json',rows)

def baseline_smoke():
    rows=[]
    for cell,arm in json.loads((HERE/f'phase_d_baseline_arms{TAG}.json').read_text()).items():
        out=evidence_path('D_baseline_smoke')/cell
        run([PYTHON,'-m','tilemega.serving.smoke','--so',arm['decode'],'--model',arm['model_path'],'--batch',arm['batch'],'--steps','64','--out',out],out/'run.log',600)
        if not json.loads((out/'smoke.json').read_text())['pass']:raise ValueError('rebuilt control smoke failed '+cell)
        rows.append(dict(cell=cell,pass_=True,artifact_id=verify(arm['decode'])['artifact_id']))
    write(evidence_path('D_baseline_smoke')/'results.json',rows)

def family_candidate(rows):
    """Compare compatible Ec/attention variants with identical GEMM execution."""
    groups={}
    for candidate in rows:
        observations=candidate.get('measurements',[])
        if candidate.get('error') or not observations or 'library' not in candidate:continue
        plan=json.loads(Path(candidate['library']+'.plan.json').read_text())
        identity=verify(candidate['library'])
        if identity['trace']:raise ValueError('trace candidate in family audit')
        signature=json.dumps(dict(pg=plan['pg'],capacity=plan.get('capacity'),gemms=plan['gemms'],kappa=plan['kappa'],residency=plan['residency'],executor=candidate['mode'],loop=candidate['loop'],nonpaged_la=plan.get('nonpaged_la',False),deferred_norm=plan.get('deferred_norm'),weight_layout=plan.get('weight_layout'),nonpaged_weight_layout=plan.get('nonpaged_weight_layout'),page_bytes=(plan.get('pages') or {}).get('page_bytes'),gemm_implementations=identity['implementations'].get('gemms')),sort_keys=True)
        values={str(p):statistics.median(o['by_past'][str(p)]['mean_ms'] for o in observations) for p in PASTS}
        groups.setdefault(signature,[]).append(dict(library=candidate['library'],ec=plan['attention_kv_block'],impl=plan.get('attention_impl','mma16'),values=values))
    best=None
    for group in groups.values():
        if len(group)<2:continue
        single=min(integrated_ms({p:dict(mean_ms=v) for p,v in row['values'].items()}) for row in group)
        envelope={str(p):min(row['values'][str(p)] for row in group) for p in PASTS}
        winners=[min(group,key=lambda row:row['values'][str(p)]) for p in PASTS]
        gain=1-integrated_ms({p:dict(mean_ms=v) for p,v in envelope.items()})/single
        if len({(r['ec'],r['impl']) for r in winners})>1 and (best is None or gain>best['family_integral_gain']):
            best=dict(family_integral_gain=gain,compatible_kv_packing=True,different_past_winner=True,single_ms=single,envelope_ms=integrated_ms({p:dict(mean_ms=v) for p,v in envelope.items()}),pasts=list(PASTS),winners=winners,scope='optimistic three-point envelope; two-segment switch cost not included')
    return best or dict(family_integral_gain=0.,compatible_kv_packing=False,different_past_winner=False)

def family(model):
    cells={}
    plans=json.loads((ROOT/f'runs/r14-{model}/plans.json').read_text())
    for b,plan in plans.items():cells[f'{model}_B{b}']=family_candidate(plan['decode_pg_choice']['candidates'])
    triggered=[cell for cell,row in cells.items() if row['different_past_winner'] and row['compatible_kv_packing'] and row['family_integral_gain']>=.02]
    write(HERE/f'results/D1_planfamily_{model}{TAG}.json',dict(cells=cells,triggered=triggered,status='requires_bounded_implementation' if triggered else 'not_triggered'))
    print('D1 completed; PlanFamily gate '+('triggered: '+','.join(triggered) if triggered else 'not triggered'))

def main():
    global TAG
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','calibrate','build','smoke','baseline_smoke','family'));p.add_argument('--model',choices=('llama','qwen3'));p.add_argument('--tag',default='');a=p.parse_args()
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.tag):p.error('invalid evidence tag')
    TAG=a.tag
    if a.action in ('build','smoke','family'):globals()[a.action](a.model)
    else:globals()[a.action]()
if __name__=='__main__':main()
