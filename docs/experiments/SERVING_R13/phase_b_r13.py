#!/usr/bin/env python3
"""Frozen Phase B builds and registered matrices, preserving rejected siblings."""
import argparse,copy,csv,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path
from arms import register
from builds_r13 import one
from fixed_builds import run_bounded
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
CELLS=('llama_B1','qwen3_B1','llama_B16','qwen3_B16')
LABELS={'B2':('B0','B0-A','NL2e','NL2g','NL2r'),
        'B3':('B0','B0-noev','B0l','PR_L2','PR_L2l'),
        'B4':('B0','B0h','PS_L1','PS_L2','PS_L2l','PSA_L2','PSA_L2l','PR_L1','PR_L2','PR_L2l')}
def load():return json.loads((HERE/'builds_b.json').read_text())
def build():
    jobs=json.loads((HERE/'jobs_b.json').read_text())
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(one,jobs))
    for row in list(records):
        if row['exit_code'] or row['phase']!='decode' or row['label'] not in ('N-R12b-noWD','P-R12bN-noWD','PS','PSA'):continue
        variants=[('trace',['TILEMEGA_TRACE_STAGE=1','TILEMEGA_TRACE_STEP=1'])]
        if row['label']=='N-R12b-noWD':variants.append(('v2',['TILEMEGA_TRACE_V2=1']))
        else:variants.append(('pages',['TILEMEGA_TRACE_V2=1','TILEMEGA_PAGE_TRACE=1','TILEMEGA_TRACE_STEP=1']))
        for tag,defines in variants:
            folder=Path(row['out']).parent/(row['label']+'-'+tag);folder.mkdir(parents=True,exist_ok=True);dest=folder/'plan.so'
            command=shlex.split(Path(row['so']+'.build_command.txt').read_text());command[command.index('-o')+1]=str(dest)
            command+=['-D'+d for d in defines]
            record=dict(row,label=row['label']+'-'+tag,out=str(folder),so=str(dest),exit_code=1)
            (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
            with (folder/'build.log').open('w') as f:record['exit_code']=run_bounded(command,f,3600)
            if record['exit_code']==0:
                record['sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
                for suffix in ('.plan.json','.cu'):
                    Path(str(dest)+suffix).write_bytes(Path(row['so']+suffix).read_bytes())
                Path(str(dest)+'.build_command.txt').write_text(shlex.join(command)+'\n')
                Path(str(dest)+'.ptxas.log').write_bytes((folder/'build.log').read_bytes())
            records.append(record);(folder/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    (HERE/'builds_b.json').write_text(json.dumps(records,indent=2)+'\n')
    print('B fixed builds: '+str(sum(r['exit_code']==0 for r in records))+'/'+str(len(records)))
def arms(matrix):
    originals=json.loads((HERE/'arms.json').read_text())
    builds={(r['cell'],r['label']):r for r in load()}
    a_builds={(r['cell'],r['label']):r for r in json.loads((HERE/'builds_a.json').read_text())}
    chosen=json.loads((HERE/'defaults_r13.json').read_text())['prefill'];result={}
    for cell in CELLS:
        model,b=cell.split('_B');b=int(b);result[cell]=[]
        for label in LABELS[matrix]:
            paged=label.startswith(('PR_','PS_','PSA_'))
            key='P-R12bN-noWD' if label.startswith('PR_') else 'PSA' if label.startswith('PSA_') else 'PS' if label.startswith('PS_') else 'N-R12bh-noWD' if label=='B0h' and cell!='qwen3_B16' else 'N-R12b-noWD'
            d=(a_builds if label=='B0-A' else builds).get((cell,key))
            pf=builds.get((cell,chosen[cell]))
            if not d or not pf:continue
            row=register(label,model,b,pf['so'],d['so'],mode='L2' if label.startswith('NL2') or '_L2' in label else 'L1',
                         loop=label=='B0l' or label.endswith('_L2l'))
            row['available']=row['available'] and d['exit_code']==0 and pf['exit_code']==0
            row['prefill_mode']='L1'
            row['model_path']=next(r['model_path'] for r in originals[cell] if r['label']=='R12bN_L1')
            if label=='NL2g':row['env']={'TILEMEGA_PLACEMENT_ABLATION':'grid_stride'}
            if label=='NL2r':row['env']={'TILEMEGA_PLACEMENT_ABLATION':'rotate'}
            if label=='B0-noev':row['step_events']=0
            result[cell].append(row)
    return result
def smoke(out):
    results=[]
    old=json.loads((HERE/'arms.json').read_text())
    rows=load()
    for row in rows:
        if row['phase']!='decode' or row['exit_code'] or row['label'].endswith(('-trace','-v2','-pages')):continue
        model=next(r['model_path'] for r in old[row['cell']] if r['label']=='R12bN_L1')
        code=subprocess.run([sys.executable,'-m','tilemega.serving.smoke','--so',row['so'],'--model',model,
                '--batch',str(row['batch']),'--steps','64','--out',str(out/row['cell']/row['label'])]).returncode
        if code==75:return 75
        results.append(dict(cell=row['cell'],label=row['label'],exit_code=code))
        if code:row.update(exit_code=code,reason='B0c smoke rejected')
    (HERE/'builds_b.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'smokes.json').write_text(json.dumps(results,indent=2)+'\n')
    print('B smokes: '+str(sum(r['exit_code']==0 for r in results))+'/'+str(len(results)))
    return 0
def trace(action,cell,out):
    table={(r['cell'],r['label']):r for r in load()}
    ordinary=next(r for r in arms('B3')[cell] if r['label']=='B0')
    if action=='B1':
        requests=[('N-R12b-noWD-trace','L1',0),('P-R12bN-noWD-trace','L1',0)]
    elif action=='B2trace':
        requests=[('N-R12b-noWD-v2','L2',0),('N-R12b-noWD-v2','L2',0)]
    elif action=='B3trace':
        requests=[('N-R12b-noWD-trace','L1',0),('N-R12b-noWD-trace','L1',1),
                  ('P-R12bN-noWD-trace','L2',0),('P-R12bN-noWD-trace','L2',1)]
    else:requests=[('P-R12bN-noWD-pages','L2',0),('PS-pages','L2',0),('PSA-pages','L2',0)]
    outputs=[]
    for i,(label,mode,loop) in enumerate(requests):
        build=table.get((cell,label))
        if not build or build['exit_code']:continue
        folder=out/f'{label}-{mode}-loop{loop}-{i}';folder.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ)
        if action=='B2trace' and i==1:env['TILEMEGA_PLACEMENT_ABLATION']='grid_stride'
        cmd=[sys.executable,'-m','tilemega.serving.trace','--model',ordinary['model_path'],
             '--prefill-so',ordinary['prefill'],'--decode-so',build['so'],'--batch',str(build['batch']),
             '--past','575','--out',str(folder),'--mode',mode,'--decode-loop',str(loop)]
        if action=='B1':cmd+=['--stage','--past-list','64,575,1000','--launches','16']
        elif action=='B3trace':cmd+=['--step','--steps','16','--launches','1']
        else:cmd+=['--launches','1'];env['TILEMEGA_PAGE_TRACE_OUT']=str(folder/'page_trace.tsv')
        with (folder/'stdout.log').open('w') as f:code=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        if code==75:return 75
        outputs.append(dict(label=label,mode=mode,loop=loop,exit_code=code,path=str(folder)))
    (out/'trace_results.json').write_text(json.dumps(outputs,indent=2)+'\n');print('trace jobs finished '+str(len(outputs)))
    return 0
def protocol(out,resume):
    selected=arms('B4')['llama_B16'];b0=next(r for r in arms('B3')['llama_B16'] if r['label']=='B0l')
    psa=next(r for r in selected if r['label']=='PSA_L2')
    cases=[]
    for base, modes in [(b0,[('L1',False),('L1',True)]),(psa,[('L1',False),('L2',False),('L2',True)])]:
        if not base['available']:continue
        candidate=dict(model=base['model_path'],batch=16,steps=64,
            prompt_ids=str(ROOT/'docs/experiments/SERVING_R10/prompts/llama_ids.json'),
            prefill=base['prefill'],decode=base['decode'],reference_prefill=base['prefill'],reference_decode=base['decode'],
            binary_sha256={k:hashlib.sha256(Path(base[k]).read_bytes()).hexdigest() for k in ('prefill','decode')})
        candidate['arms']=[dict(label=f'{mode}_loop{int(loop)}',prefill=base['prefill'],decode=base['decode'],
                               modes=[mode],decode_loop=loop,require_loop=loop) for mode,loop in modes]
        cases.append(candidate)
    path=out/'cases.json';path.write_text(json.dumps(cases,indent=2)+'\n')
    if len(cases)!=2:raise RuntimeError('LP-1 or FX-21 plan missing; cannot claim protocol coverage')
    cmd=[sys.executable,str(ROOT/'docs/experiments/SERVING_R11/check_protocol.py'),'--cases',str(path),'--out',str(out/'processes'),'--processes','50']
    if resume:cmd+=['--resume']
    return subprocess.run(cmd).returncode
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','smoke','anchor','B1','B2trace','B3trace','B4trace','protocol'))
    p.add_argument('--matrix',choices=LABELS);p.add_argument('--cell',choices=CELLS)
    p.add_argument('--round',type=int,default=0);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.action=='build':build();return 0
    if a.action=='smoke':return smoke(a.out)
    if a.action=='protocol':return protocol(a.out,a.resume)
    if a.action=='anchor':
        path=a.out/'arms.json';path.write_text(json.dumps(arms(a.matrix),indent=2)+'\n')
        return subprocess.run([sys.executable,str(HERE/'anchor.py'),'--arms',str(path),'--cell',a.cell,'--round',str(a.round),'--out',str(a.out)]).returncode
    return trace(a.action,a.cell,a.out)
if __name__=='__main__':raise SystemExit(main())
