#!/usr/bin/env python3
"""Resolve predeclared Phase B arms from finished manifests, without selection."""
import argparse,copy,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]

def original(cell,label):
    return copy.deepcopy(next(r for r in json.loads((HERE/'arms.json').read_text())[cell] if r['label']==label))
def fixed(cell,label,mode='L2',loop=True):
    source=label.replace('-sep','');folder=HERE/'raw/B0b'/cell/source;record=folder/'record.json'
    if not record.exists() or json.loads(record.read_text()).get('exit_code')!=0:return None
    base=original(cell,'R12bN_L2');base.update(label=label,decode=str(folder/'plan.so'),mode=mode,decode_loop=loop,root=str(ROOT),available=True)
    base.pop('binaries',None);base['placeholder_measurement']=True
    return base

def final(cell):
    model,b=cell.split('_B');pair=json.loads((ROOT/f'runs/r12c-{model}/plans.json').read_text())[b]
    row=original(cell,'R12bP');row.update(label='R12cP',root=str(ROOT),prefill=pair['prefill'],decode=pair['decode'],available=True);row.pop('binaries',None);return row

def anchor(matrix,cell,rnd,out):
    rows=[]
    if matrix=='B1':
        rows=[fixed(cell,n,loop=not n.endswith('-sep')) for n in ('P-base','P-D64K','P-D128K','P-noWD','P-base-sep','P-D128K-sep')]
    elif matrix=='B3':
        rows=[fixed(cell,n,loop=False) for n in ('P-base','P-noDN','P-noLA','P-noDN-noLA')]
        for name in ('N-base','N-noDN'):
            for mode in ('L2','L1'):
                row=fixed(cell,name,mode=mode,loop=False)
                if row:row['label']+='_'+mode
                rows.append(row)
    elif matrix=='B8':
        rows=[original(cell,'R10C')]
        row=original(cell,'R10C');row.update(label='R10C_new',kind='tm',root=str(ROOT),decode_loop=False);rows.append(row)
        rows += [fixed(cell,f'R10G-{i}',mode='L1',loop=False) for i in range(4)]
        row=fixed(cell,'R10G-3',mode='L2',loop=False)
        if row:row['label']='R10G-3_L2'
        rows += [row,fixed(cell,'N-R12b',mode='L1',loop=False),original(cell,'R12bN_L1')]
    elif matrix=='B5':rows=[original(cell,'vllm'),original(cell,'R10C'),original(cell,'R12bP'),final(cell)]
    rows=[r for r in rows if r and r.get('available',True)]
    out.mkdir(parents=True,exist_ok=True);arms=out/'arms.json';arms.write_text(json.dumps({cell:rows},indent=2)+'\n')
    return subprocess.run([sys.executable,str(HERE/'anchor.py'),'--arms',str(arms),'--cell',cell,'--round',str(rnd),'--out',str(out)]).returncode

def smoke(out):
    rows=[]
    for cell,names in [('llama_B16',['P-base','P-D128K','P-noLA','P-noDN','N-noDN']),('llama_B1',['R10G-0','R10G-3'])]:
        for label in names:
            arm=fixed(cell,label)
            if not arm:rows.append(dict(cell=cell,arm=label,status='missing'));continue
            folder=out/cell/label;cmd=[sys.executable,'-m','tilemega.serving.smoke','--so',arm['decode'],'--model',arm['model_path'],'--batch',str(arm['batch']),'--steps','64','--out',str(folder)]
            code=subprocess.run(cmd,cwd=ROOT).returncode
            if code==75:return 75
            rows.append(dict(cell=cell,arm=label,exit_code=code))
            if code:
                record=HERE/'raw/B0b'/cell/label/'record.json';data=json.loads(record.read_text());data['exit_code']=code;data['smoke_failed']=True;record.write_text(json.dumps(data,indent=2)+'\n')
    out.mkdir(parents=True,exist_ok=True);(out/'smokes.json').write_text(json.dumps(rows,indent=2)+'\n');return 0

def check(cell,out):
    arm=final(cell);return subprocess.run([sys.executable,'-m','tilemega.serving.check_modes','--model',arm['model_path'],'--prefill-so',arm['prefill'],'--decode-so',arm['decode'],'--prompt-ids',str(ROOT/f"docs/experiments/SERVING_R10/prompts/{arm['model']}_ids.json"),'--batch',str(arm['batch']),'--steps','1024','--out',str(out)],cwd=ROOT).returncode

def hf(out):
    codes=[]
    for cell in ('llama_B1','qwen3_B1','llama_B16','qwen3_B16'):
        arm=final(cell)
        for label in ('R12cP','vllm'):
            folder=HERE/'raw'/f'B5_{cell}_r0'/cell/label/'round0'
            if label=='vllm':folder=folder/f"B{arm['batch']}"
            metrics=folder/'measurements.json'
            if not metrics.exists():codes.append(1);continue
            data=json.loads(metrics.read_text());runs=data.get('runs',data.get('generation_runs',[]));row=next(r for r in runs if not r['warmup'] and r['N']==1024)
            generated=folder/row.get('tokens_file','tokens.json')
            if not generated.exists() and 'tokens' in row:generated=out/f'{cell}_{label}_tokens.json';generated.parent.mkdir(parents=True,exist_ok=True);generated.write_text(json.dumps(row['tokens'])+'\n')
            cmd=[sys.executable,'-m','tilemega.serving.hf_check','--model',arm['model_path'],'--prompt-ids',str(ROOT/f"docs/experiments/SERVING_R10/prompts/{arm['model']}_ids.json"),'--generated',str(generated),'--out',str(out/f'{cell}_{label}'),'--skip-free-greedy']
            code=subprocess.run(cmd,cwd=ROOT).returncode
            if code==75:return 75
            codes.append(code)
    return int(any(codes))

def protocol(out,resume=False):
    arm=final('llama_B16');out.mkdir(parents=True,exist_ok=True);cases=out/'cases.json'
    subprocess.run([sys.executable,str(ROOT/'docs/experiments/SERVING_R12B/make_protocol_cases.py'),'--prefill-so',arm['prefill'],'--decode-so',arm['decode'],'--config',str(ROOT/'configs/e2e/llama_r12c.json'),'--out',str(cases)],check=True)
    cmd=[sys.executable,str(ROOT/'docs/experiments/SERVING_R11/check_protocol.py'),'--cases',str(cases),'--out',str(out/'processes'),'--processes','50']
    if resume:cmd+=['--resume']
    return subprocess.run(cmd,cwd=ROOT).returncode

def trace(out):
    codes=[]
    for b in (1,16):
        arm=fixed(f'llama_B{b}','P-trace')
        if not arm:continue
        folder=out/f'B{b}';folder.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,TILEMEGA_PAGE_TRACE_OUT=str(folder/'pages.json'))
        code=subprocess.run([sys.executable,'-m','tilemega.serving.trace','--model',arm['model_path'],'--prefill-so',arm['prefill'],'--decode-so',arm['decode'],'--batch',str(b),'--past','575','--launches','1','--out',str(folder)],cwd=ROOT,env=env).returncode
        if code==75:return 75
        codes.append(code)
        config=json.loads((Path(arm['model_path'])/'config.json').read_text())
        manifest=json.loads(Path(arm['decode']+'.plan.json').read_text())
        h=config['hidden_size'];d=config.get('head_dim',h//config['num_attention_heads']);groups=config['num_key_value_heads'];layers=config['num_hidden_layers'];intermediate=config['intermediate_size'];vocab=config['vocab_size']
        dimensions=[((config['num_attention_heads']+2*groups)*d,h),(h,h),(2*intermediate,h),(h,intermediate)]*layers+[(vocab,h)]
        weight_bytes=sum(((n+g['tile_n']-1)//g['tile_n'])*((k+g['tile_k']-1)//g['tile_k'])*g['tile_n']*g['tile_k']*2 for (n,k),g in zip(dimensions,manifest['gemms']))
        stream_bytes=weight_bytes+4*b*groups*575*d*layers+2*b*h
        (folder/'stream_floor.json').write_text(json.dumps(dict(weight_bytes=weight_bytes,history_kv_bytes=4*b*groups*575*d*layers,stream_bytes=stream_bytes,stream_gbps=902,stream_floor_ns=stream_bytes/902,convention='tile-padded weights, history KV, embedding; all fetched from DRAM for stream-only comparison'),indent=2)+'\n')
        if code==0 and (folder/'pages.json').exists():
            chain=folder/'chain'
            subprocess.run([sys.executable,str(ROOT/'docs/experiments/TRACE_V2/analyze.py'),str(folder),'--source',str(HERE/'raw/B0b'/f'llama_B{b}'/'P-base/plan.so.cu'),'--window','1','--out',str(chain)],cwd=ROOT)
            page_cmd=[sys.executable,str(ROOT/'docs/experiments/SERVING_R11/analyze_page_chain.py'),'--page-trace',str(folder/'pages.json'),'--model','llama','--batch',str(b),'--out',str(folder/'analysis')]
            if (chain/'analysis.tsv').exists():page_cmd+=['--chain-analysis',str(chain/'analysis.tsv')]
            subprocess.run(page_cmd,cwd=ROOT)
            if (chain/'trace_v2.chain_links.tsv').exists():
                subprocess.run([sys.executable,str(ROOT/'docs/experiments/SERVING_R11/diagnose_pg_chain.py'),'--cu',str(HERE/'raw/B0b'/f'llama_B{b}'/'P-base/plan.so.cu'),'--manifest',arm['decode']+'.plan.json','--chain',str(chain/'trace_v2.chain_links.tsv'),'--out',str(folder/'chain_classes.tsv')],cwd=ROOT)
    ncu=shutil.which('ncu')
    if ncu:
        for label in ('P-base','P-D128K'):
            arm=fixed('llama_B16',label)
            if not arm:continue
            folder=out/('ncu-'+label);folder.mkdir(parents=True,exist_ok=True)
            cmd=[ncu,'--metrics','dram__bytes_read.sum,lts__t_sector_hit_rate.pct','--kernel-name','regex:.*tilemega_l2_kernel.*','--launch-count','1','--csv',sys.executable,'-m','tilemega.serving.trace','--model',arm['model_path'],'--prefill-so',arm['prefill'],'--decode-so',arm['decode'],'--batch','16','--past','575','--launches','1','--out',str(folder)]
            with (folder/'metrics.csv').open('w') as f:code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
            (folder/'status.json').write_text(json.dumps(dict(exit_code=code,command=cmd))+'\n')
    else:(out/'ncu.json').write_text(json.dumps(dict(status='not_available'))+'\n')
    return int(any(codes))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('anchor','smoke','check','hf','protocol','trace','audit'));p.add_argument('--matrix');p.add_argument('--cell');p.add_argument('--round',type=int,default=0);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='anchor':return anchor(a.matrix,a.cell,a.round,a.out)
    if a.action=='smoke':return smoke(a.out)
    if a.action=='check':return check(a.cell,a.out)
    if a.action=='hf':return hf(a.out)
    if a.action=='protocol':return protocol(a.out,a.resume)
    if a.action=='audit':
        a.out.mkdir(parents=True,exist_ok=True);binaries=[]
        for p in (HERE/'raw/B0b').glob('*/*/record.json'):
            r=json.loads(p.read_text())
            if r.get('exit_code')==0:binaries.append(r['so'])
        for model in ('llama','qwen3'):
            p=ROOT/f'runs/r12c-{model}/plans.json'
            if p.exists():
                for b,pair in json.loads(p.read_text()).items():
                    if b.isdigit():binaries+=list(pair.values())
        if not binaries:return 1
        return subprocess.run([os.environ['TILEMEGA_BIN'],'audit','sass',*binaries,'--out',str(a.out/'sass.json')],cwd=ROOT).returncode
    return trace(a.out)
if __name__=='__main__':raise SystemExit(main())
