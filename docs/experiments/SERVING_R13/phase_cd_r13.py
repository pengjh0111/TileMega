#!/usr/bin/env python3
"""Registered C controls/retention and D final builds; no new scan domains."""
import argparse,concurrent.futures,copy,csv,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
from arms import register
from builds_r13 import one
from choose_r13 import apply_prefill_pins,loop,final
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
CELLS=('llama_B1','qwen3_B1','llama_B16','qwen3_B16')
MATRICES={'C-L2b':('N-control','N-slim'),'C-WL':('N-control','N-tiled'),
 'C-LP2':('P-control','P-loop-split'),
 'C-PG3':('P-control','P-D64-last','P-D64-normal','P-D64-first')}
def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')
def run(cmd,out):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:code=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT).returncode
    if code==75:raise SystemExit(75)
    return code
def cbuild(out):
    jobs=read(HERE/'jobs_c.json')
    # A failed small-body check removes only C-WL, leaving other controls valid.
    check=HERE/'raw/Csmall/checks.json'
    if not check.exists() or not all(r['equal'] for r in read(check)):
        jobs=[r for r in jobs if r['label']!='N-tiled']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,jobs))
    write(HERE/'builds_c.json',rows)
    print('C builds '+str(sum(r['exit_code']==0 for r in rows))+'/'+str(len(rows)))
def records():return {(r['cell'],r['label']):r for r in read(HERE/'builds_c.json')}
def arms(matrix):
    rows=records();original=read(HERE/'arms.json');choices=read(HERE/'defaults_r13.json')['prefill'];result={}
    for cell in CELLS:
        model,b=cell.split('_B');b=int(b);result[cell]=[]
        labels=MATRICES.get(matrix)
        if matrix=='C-AT':
            if model!='qwen3':continue
            labels=('N-control',)+tuple(r['label'] for r in rows.values() if r['cell']==cell and r['label'].startswith('N-AT-'))
        pf=rows[cell,choices[cell]]
        for label in labels:
            row=rows.get((cell,label))
            if not row or row['exit_code'] or pf['exit_code']:continue
            smoke_file=HERE/'raw/Csmoke/smoke_results.json'
            if smoke_file.exists() and any(r['cell']==cell and r['label']==label and r['exit_code'] for r in read(smoke_file)):continue
            mode='L2' if matrix=='C-L2b' or label.startswith('P-') else 'L1'
            arm=register(label,model,b,pf['so'],row['so'],mode=mode,loop=label.startswith('P-'))
            arm['prefill_mode']='L1';arm['model_path']=next(r['model_path'] for r in original[cell] if r['label']=='R12bN_L1')
            if matrix=='C-L2b':arm['env']={'TILEMEGA_PLACEMENT_ABLATION':'grid_stride'}
            result[cell].append(arm)
    return result
def anchor(matrix,cell,rnd,out):
    path=out/'arms.json';write(path,arms(matrix))
    return run([sys.executable,HERE/'anchor.py','--arms',path,'--cell',cell,'--round',rnd,'--out',out],out/'anchor.log')
def smoke(out):
    results=[]
    for row in read(HERE/'builds_c.json'):
        if row['phase']!='decode' or row['exit_code']:continue
        folder=out/row['cell']/row['label'];folder.mkdir(parents=True,exist_ok=True)
        model=row['model'];vocab=json.loads(Path(row['manifest']).read_text()).get('vocab',None)
        originals=read(HERE/'arms.json')[row['cell']];model_path=next(a['model_path'] for a in originals if a['label']=='R12bN_L1')
        cmd=[sys.executable,'-m','tilemega.serving.smoke','--so',row['so'],'--model',model_path,'--batch',row['batch'],'--steps','64','--out',folder]
        code=run(cmd,folder/'stdout.log');results.append(dict(cell=row['cell'],label=row['label'],exit_code=code))
    write(out/'smoke_results.json',results)
    print('C smoke '+str(sum(r['exit_code']==0 for r in results))+'/'+str(len(results)))
    # A rejected ablation is excluded, not a reason to skip every sibling.
    return 0
def small(out):
    binary=HERE/'raw/Cpre'
    checks=[]
    for config,tn in ((0,128),(7,32),(8,256)):
        for m in (1,16):
            for split,k in ((1,264),(4,512)):
                for op in range(4):
                    key=f'{config}_{m}_{split}_{op}';files=[]
                    for tiled in (0,1):
                        dest=out/f'{key}_{tiled}.bin';files.append(dest)
                        code=run([binary/f'matrix{tiled}',config,m,tn+8,k,split,op,dest],out/f'{key}_{tiled}.log')
                        if code:raise RuntimeError('small GEMM launch failed: '+key)
                    checks.append(dict(case=key,equal=files[0].read_bytes()==files[1].read_bytes()))
    write(out/'checks.json',checks);print('tiled GEMM exact '+str(sum(r['equal'] for r in checks))+'/'+str(len(checks)))
    return int(not all(r['equal'] for r in checks))
def protocol(out,resume,matrix):
    aa=arms(matrix)['llama_B16'];labels=MATRICES[matrix]
    control=next(a for a in aa if a['label']==labels[0]);candidate=next(a for a in aa if a['label']==labels[1])
    case=dict(model=control['model_path'],batch=16,steps=64,
      prompt_ids=str(ROOT/'docs/experiments/SERVING_R10/prompts/llama_ids.json'),
      prefill=candidate['prefill'],decode=candidate['decode'],reference_prefill=control['prefill'],reference_decode=control['decode'],
      binary_sha256={k:hashlib.sha256(Path(candidate[k]).read_bytes()).hexdigest() for k in ('prefill','decode')},
      arms=[dict(label='reference',prefill=control['prefill'],decode=control['decode'],modes=[control['mode']],decode_loop=False),
            dict(label='candidate',prefill=candidate['prefill'],decode=candidate['decode'],modes=[candidate['mode']],decode_loop=candidate['decode_loop'],require_loop=candidate['decode_loop'],env=candidate['env'])])
    path=out/'cases.json';write(path,[case]);cmd=[sys.executable,ROOT/'docs/experiments/SERVING_R11/check_protocol.py','--cases',path,'--out',out/'processes','--processes','50']
    if resume:cmd+=['--resume']
    return run(cmd,out/'protocol.log')
def samples(matrix,cell):
    result={};tokens={}
    for path in sorted((HERE/'raw').glob(f'C_{matrix}_{cell}_r*/{cell}/round*.json')):
        data=read(path)
        if data.get('invalidated'):continue
        for label,row in data['arms'].items():
            if row['exit_code']:continue
            result.setdefault(label,[]).append((row['e2e_seconds']-row['ttft_seconds'])/1023)
            files=list(Path(row['out']).glob('tokens_N1024_run1.json'))
            if files:tokens.setdefault(label,[]).append(read(files[0]))
    return result,tokens
def numerical(matrix,cell):
    values,tokens=samples(matrix,cell);base=MATRICES.get(matrix,('N-control',))[0]
    reference=tokens.get(base,[])
    if len(reference)!=3:return {}
    return {label:len(rows)==3 and all(r==reference[0] for r in rows) and all(r==reference[0] for r in reference) for label,rows in tokens.items()}
def check_c(out):
    results=[]
    for cell in ('qwen3_B1','qwen3_B16'):
        values,_=samples('C-AT',cell);base=values.get('N-control',[])
        labels=[k for k,v in values.items() if k!='N-control' and len(v)==3]
        if not base or not labels:continue
        best=min(labels,key=lambda k:statistics.median(values[k]));arm=next(r for r in arms('C-AT')[cell] if r['label']==best)
        paths=list((HERE/'raw').glob(f'C_C-AT_{cell}_r*/{cell}/{best}/round*/tokens_N1024_run1.json'))
        folder=out/cell;folder.mkdir(parents=True,exist_ok=True)
        code=run([sys.executable,'-m','tilemega.serving.hf_check','--model',arm['model_path'],
          '--prompt-ids',ROOT/f'docs/experiments/SERVING_R10/prompts/qwen3_ids.json','--generated',paths[0],
          '--skip-free-greedy','--out',folder/'report.json'],folder/'hf.log')
        results.append(dict(cell=cell,label=best,exit_code=code))
    write(out/'checks.json',results);print('C attention HF '+str(sum(r['exit_code']==0 for r in results))+'/'+str(len(results)))
    return 0
def retain(out):
    smoke_rows=read(HERE/'raw/Csmoke/smoke_results.json') if (HERE/'raw/Csmoke/smoke_results.json').exists() else []
    passed={(r['cell'],r['label']) for r in smoke_rows if r['exit_code']==0}
    decisions={};comparisons=[]
    for matrix,labels in MATRICES.items():
        threshold=.02 if matrix=='C-L2b' else .01 if matrix=='C-WL' else .015 if matrix=='C-PG3' else 0
        wins={}
        for cell in CELLS:
            values,tokens=samples(matrix,cell);base=values.get(labels[0],[]);equal=numerical(matrix,cell)
            if len(base)!=3:continue
            for label in labels[1:]:
                vv=values.get(label,[])
                if len(vv)!=3:continue
                rel=statistics.median(vv)/statistics.median(base)-1
                ok=(cell,label) in passed and equal.get(label,False) and -rel>=threshold
                if matrix!='C-LP2':ok=ok and -rel>threshold
                comparisons.append(dict(rule=matrix,cell=cell,label=label,relative=rel,tokens_equal=equal.get(label,False),pass_=ok))
                if ok:wins.setdefault(label,[]).append(cell)
        needed=2 if matrix in ('C-L2b','C-WL') else 1
        options=[k for k,v in wins.items() if len(v)>=needed]
        if matrix in ('C-L2b','C-WL','C-LP2'):
            path=HERE/f'raw/Cprotocol_{matrix}/processes/summary.json'
            # The protocol driver records its aggregate result after 50 children.
            protocol_ok=path.exists() and read(path).get('complete',False) and read(path).get('passed')==50 and read(path).get('failed')==0
            if not protocol_ok:options=[]
        selected=min(options,key=lambda k:statistics.mean(r['relative'] for r in comparisons if r['rule']==matrix and r['label']==k)) if options else None
        decisions[matrix]=dict(retained=bool(selected),selected=selected,cells=wins.get(selected,[]))
    checks=read(HERE/'raw/Ccheck/checks.json') if (HERE/'raw/Ccheck/checks.json').exists() else []
    at={}
    for r in checks:
        values,_=samples('C-AT',r['cell']);candidate=values.get(r['label'],[]);base=values.get('N-control',[])
        if r['exit_code']==0 and len(candidate)==len(base)==3 and statistics.median(candidate)/statistics.median(base)-1<-.015 and (r['cell'],r['label']) in passed:at[r['cell']]=r['label']
    decisions['C-AT']=dict(retained=bool(at),selected=at)
    loop_data={'cells':{}}
    for row in csv.DictReader((HERE/'results/T6.tsv').open(),delimiter='\t'):
        if row['arm'] in ('B0l','B0-noev'):loop_data['cells'].setdefault(row['cell'],{})[row['arm']]=json.loads(row['tpot_s_samples'])
    loop_choice=loop(loop_data);write(HERE/'loop_decision.json',loop_choice)
    apply_prefill_pins(HERE)
    for model in ('llama','qwen3'):
        path=ROOT/f'configs/e2e/{model}_r13.json';cfg=read(path);features=cfg['features']
        features.update(l2_slim=int(decisions['C-L2b']['retained']),page_loop_split=int(decisions['C-LP2']['retained']),
            nonpaged_weight_layout='tiled' if decisions['C-WL']['retained'] else 'row',lookahead_bytes=0,evict_first=0,evict_last=1)
        if decisions['C-PG3']['retained']:
            label=decisions['C-PG3']['selected'];features.update(lookahead_bytes=65536,evict_last=int(label=='P-D64-last'),evict_first=int(label=='P-D64-first'))
        cfg['solver']['exclude_l1_loop']=loop_choice['exclude_l1_loop'];cfg['output']['dir']=f'runs/r13-{model}'
        cfg['solver']['time_budget_s']=cfg['solver']['candidate_guard_wait_s']=1800
        # Attention choices are reported per cell; the final search already
        # includes the registered Ec candidates and chooses its own structure.
        write(path,cfg)
    write(HERE/'phase_c_retention.json',dict(decisions=decisions,comparisons=comparisons,loop_choice=loop_choice))
    print('C retention '+json.dumps({k:v['retained'] for k,v in decisions.items()}))
def d0(out):
    from tilemega.cli import Run,read_config
    runner=Run(read_config(ROOT/'configs/e2e/llama_r13.json'),os.environ['TILEMEGA_BIN']);runner.calibrate()
    return run([sys.executable,HERE/'dram_ceiling.py','--binary','/root/r13_work/tilemega-loadbench-resident',
      '--out',out/'ceiling','--target',runner.target,'--target-out',runner.target],out/'ceiling.log')
def d1(out,model):
    code=run([sys.executable,'-m','tilemega','build','--config',ROOT/f'configs/e2e/{model}_r13.json','--run-dir',ROOT/f'runs/r13-{model}'],out/'build.log')
    if code:return code
    jobs=[r for r in read(HERE/'jobs_c.json') if r['model']==model and (r['label']=='N-control' or r['phase']=='prefill')]
    for r in jobs:r['out']=str(out/r['cell']/r['label']);r['label']='B0-D' if r['label']=='N-control' else r['label']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,jobs))
    write(out/'baseline_builds.json',rows);print('D1 complete '+model)
    return int(any(r['exit_code'] for r in rows))
def d_arms(only=None):
    result={};old=read(HERE/'arms.json');prefill=read(HERE/'defaults_r13.json')['prefill']
    for cell in (CELLS if only is None else (only,)):
        model,b=cell.split('_B');b=int(b);built=read(HERE/f'raw/D1_{model}/baseline_builds.json');plans=read(ROOT/f'runs/r13-{model}/plans.json')[str(b)]
        lookup={r['label']:r for r in built if r['cell']==cell};base=lookup['B0-D'];pf=lookup[prefill[cell]]
        final_arm=register('R13F',model,b,plans['prefill'],plans['decode'],mode='auto',loop='auto');final_arm['prefill_mode']='auto'
        baseline=register('B0-D',model,b,pf['so'],base['so'],mode='L1',loop=False);baseline['prefill_mode']='L1'
        model_path=next(r['model_path'] for r in old[cell] if r['label']=='R12bN_L1')
        result[cell]=[copy.deepcopy(r) for r in old[cell] if r['label'] in ('vllm','R10C','R12bN_L1')]+[baseline,final_arm]
        for r in result[cell]:r['model_path']=model_path
    return result
def d_anchor(cell,rnd,out):
    path=out/'arms.json';write(path,d_arms(cell));return run([sys.executable,HERE/'anchor.py','--arms',path,'--cell',cell,'--round',rnd,'--out',out],out/'anchor.log')
def dcheck(cell,out):
    arm=next(r for r in d_arms(cell)[cell] if r['label']=='R13F');model,b=cell.split('_B')
    cmd=[sys.executable,'-m','tilemega.serving.check_modes','--model',arm['model_path'],'--prefill-so',arm['prefill'],'--decode-so',arm['decode'],'--prompt-ids',ROOT/f'docs/experiments/SERVING_R10/prompts/{model}_ids.json','--batch',b,'--steps','1024','--out',out/'modes']
    code=run(cmd,out/'modes.log');results={'C-2':code}
    files=list((HERE/'raw').glob(f'D2_{cell}_r*/{cell}/R13F/round*/tokens_N1024_run1.json'))
    if files:results['C-1']=run([sys.executable,'-m','tilemega.serving.hf_check','--model',arm['model_path'],'--prompt-ids',ROOT/f'docs/experiments/SERVING_R10/prompts/{model}_ids.json','--generated',files[0],'--skip-free-greedy','--out',out/'hf.json'],out/'hf.log')
    else:results['C-1']='no token file'
    # Validate the fallback as well: a faster invalid candidate must not
    # become the default, and a fallback does not acquire correctness by name.
    base_files=list((HERE/'raw').glob(f'D2_{cell}_r*/{cell}/B0-D/round*/tokens_N1024_run1.json'))
    if base_files:
        results['C-1_B0-D']=run([sys.executable,'-m','tilemega.serving.hf_check','--model',arm['model_path'],
          '--prompt-ids',ROOT/f'docs/experiments/SERVING_R10/prompts/{model}_ids.json','--generated',base_files[0],
          '--skip-free-greedy','--out',out/'hf_baseline.json'],out/'hf_baseline.log')
    all_files={'R13F':files,'B0-D':base_files}
    for label,paths in all_files.items():
        values=[read(path) for path in paths]
        results['repeat_'+label]=0 if len(values)==3 and all(v==values[0] for v in values) else 1
    write(out/'checks.json',results);print(cell,json.dumps(results));return int(any(v!=0 for v in results.values()))
def dprotocol(out,resume):
    from tilemega.serving.execution import read_execution
    arm=next(r for r in d_arms('llama_B16')['llama_B16'] if r['label']=='R13F')
    choice=read_execution(arm['decode']);mode=choice['decode_mode'];uses_loop=bool(choice['decode_loop'])
    case=dict(model=arm['model_path'],batch=16,steps=64,
      prompt_ids=str(ROOT/'docs/experiments/SERVING_R10/prompts/llama_ids.json'),
      prefill=arm['prefill'],decode=arm['decode'],reference_prefill=arm['prefill'],reference_decode=arm['decode'],
      binary_sha256={k:hashlib.sha256(Path(arm[k]).read_bytes()).hexdigest() for k in ('prefill','decode')},
      arms=[dict(label='L1_reference',prefill=arm['prefill'],decode=arm['decode'],modes=['L1'],decode_loop=False),
            dict(label='selected',prefill=arm['prefill'],decode=arm['decode'],modes=[mode],decode_loop=uses_loop,require_loop=uses_loop)])
    path=out/'cases.json';write(path,[case]);cmd=[sys.executable,ROOT/'docs/experiments/SERVING_R11/check_protocol.py','--cases',path,'--out',out/'processes','--processes','50']
    if resume:cmd+=['--resume']
    return run(cmd,out/'protocol.log')
def finalize(out):
    data={}
    for cell in CELLS:
        for path in (HERE/'raw').glob(f'D2_{cell}_r*/{cell}/round*.json'):
            rr=read(path)
            if rr.get('invalidated'):continue
            for label in ('B0-D','R13F'):
                row=rr['arms'].get(label,{})
                if row.get('exit_code')==0:data.setdefault(cell,{}).setdefault(label,[]).append((row['e2e_seconds']-row['ttft_seconds'])/1023)
    complete={cell:row for cell,row in data.items() if all(len(row.get(label,[]))==3 for label in ('B0-D','R13F'))}
    decisions=final(complete);rejected={}
    for cell in CELLS:
        path=HERE/f'raw/D3_{cell}/checks.json'
        checks=read(path) if path.exists() else {}
        if decisions[cell]=='R13F' and any(checks.get(k)!=0 for k in ('C-1','C-2','repeat_R13F')):
            rejected[cell]='R13F correctness incomplete or failed';decisions[cell]='B0-D'
        if decisions[cell]=='B0-D' and any(checks.get(k)!=0 for k in ('C-1_B0-D','repeat_B0-D')):
            rejected[cell]='fallback correctness incomplete or failed';decisions[cell]='unvalidated'
    write(HERE/'final_decision.json',dict(choices=decisions,raw_values=data,rejected=rejected))
    if any(v=='unvalidated' for v in decisions.values()):
        print('Default publication blocked by unvalidated correctness: '+json.dumps(rejected));return 1
    # Pin every selected binary and its serving sidecar without rewriting any
    # cached manifest; a missing final comparison cannot produce a new winner.
    from tilemega.serving.execution import write_execution
    all_arms=d_arms()
    for model in ('llama','qwen3'):
        cfg=read(ROOT/f'configs/e2e/{model}_r13.json');cfg['selected_plans']={}
        for batch in (1,16):
            cell=f'{model}_B{batch}';choice=decisions[cell];arm=next(r for r in all_arms[cell] if r['label']==choice)
            cfg['selected_plans'][str(batch)]={k:arm[k] for k in ('prefill','decode','mode','decode_loop','prefill_mode','binaries')}
            if choice=='B0-D':write_execution(arm['decode'],'L1',0,'L1',origin='R13 registered final fallback')
        final_dir=ROOT/f'runs/r13-final-{model}'
        cfg['output']['dir']=str(final_dir)
        write(final_dir/'plans.json',{b:dict(prefill=r['prefill'],decode=r['decode'],selection=r)
              for b,r in cfg['selected_plans'].items()})
        write(final_dir/'config.json',cfg)
        write(ROOT/f'configs/e2e/{model}_r13_final.json',cfg)
    print('D final selection '+json.dumps(decisions))
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build-c','smoke-c','small','anchor-c','protocol-c','check-c','retain','d0','d1','anchor-d','check-d','protocol-d','final'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--cell');p.add_argument('--matrix');p.add_argument('--round',type=int,default=0);p.add_argument('--model');p.add_argument('--resume',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic()
    action=a.action
    if action in ('smoke-c','small','anchor-c','protocol-c','check-c'):
        spent=sum(read(p).get('run_s',0) for p in (HERE/'raw').glob('C*/runtime.json'))
        if spent>=21600:write(a.out/'budget.json',dict(gpu_seconds=spent,limit=21600));print('Phase C GPU budget exhausted');return 3
    try:
        if action=='build-c':result=cbuild(a.out)
        elif action=='smoke-c':result=smoke(a.out)
        elif action=='small':result=small(a.out)
        elif action=='anchor-c':result=anchor(a.matrix,a.cell,a.round,a.out)
        elif action=='protocol-c':result=protocol(a.out,a.resume,a.matrix)
        elif action=='check-c':result=check_c(a.out)
        elif action=='retain':result=retain(a.out)
        elif action=='d0':result=d0(a.out)
        elif action=='d1':result=d1(a.out,a.model)
        elif action=='anchor-d':result=d_anchor(a.cell,a.round,a.out)
        elif action=='check-d':result=dcheck(a.cell,a.out)
        elif action=='protocol-d':result=dprotocol(a.out,a.resume)
        else:result=finalize(a.out)
        return result or 0
    finally:
        if action in ('smoke-c','small','anchor-c','protocol-c','check-c'):
            previous=read(a.out/'runtime.json') if (a.out/'runtime.json').exists() else {}
            write(a.out/'runtime.json',dict(run_s=previous.get('run_s',0)+time.monotonic()-begin,attempts=previous.get('attempts',0)+1))
if __name__=='__main__':raise SystemExit(main())
