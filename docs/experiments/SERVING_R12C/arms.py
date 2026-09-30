#!/usr/bin/env python3
"""Register existing binaries; missing arms stay explicit, never rebuilt here."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PY='/root/venvs/tilemega-torch213-cu126/bin/python'
def register(label,model,batch,prefill,decode,root=ROOT,kind='tm',mode='L2',loop=False):
    row=dict(label=label,kind=kind,model=model,batch=batch,root=str(root),python=PY,
             prefill=str(prefill),decode=str(decode),mode=mode,decode_loop=loop,env={})
    row['binaries']={}
    for phase,path in [('prefill',prefill),('decode',decode)]:
        path=Path(path);rec=dict(path=str(path),exists=path.is_file())
        if path.is_file():
            rec['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            manifest=Path(str(path)+'.plan.json')
            if manifest.exists():
                rec['manifest']=json.loads(manifest.read_text());rec['target']=rec['manifest'].get('runtime_target')
            record=path.parent/'record.json'
            if record.exists():rec['record']=json.loads(record.read_text())
            for key in ('compiler_commit','source_sha256'):
                rec[key]=rec.get('record',{}).get(key,'unknown (original build record does not supply this field)')
        row['binaries'][phase]=rec
    row['available']=all(r['exists'] for r in row['binaries'].values())
    return row

def main():
    result={}
    for model in ('llama','qwen3'):
        run=ROOT/f'runs/r12b-{model}';plans=json.loads((run/'plans.json').read_text())
        events=json.loads((run/'cache.json').read_text())
        for batch in (1,16):
            cell=f'{model}_B{batch}';pair=plans[str(batch)];rows=[]
            for mode in ('L2','L1'):
                rows.append(register('R12bN_'+mode,model,batch,pair['prefill'],pair['decode'],mode=mode))
            candidates=[e for e in events if e.get('layer')=='plan' and e.get('phase')=='decode' and e.get('pg')=='pages' and e.get('batch')==batch]
            path=Path.home()/'.cache/tilemega/plans'/candidates[-1]['key']/'plan.so' if candidates else Path('/missing/R12bP')
            rows.append(register('R12bP',model,batch,pair['prefill'],path,loop=True))
            old=Path('/root/r11_work/r10_control/plans')
            rows.append(register('R10C',model,batch,old/f'{model}_prefill_B{batch}/plan.so',old/f'{model}_decode_B{batch}/plan.so',root=Path('/root/r11_work/r10_baseline'),kind='tm_old',mode='L1'))
            if model=='llama':
                for name,pg in [('S1P','pages'),('S1N','l2')]:
                    path=ROOT/f'docs/experiments/SERVING_R12B/controls/s1_B{batch}_{pg}/decode_B{batch}.so'
                    rows.append(register(name,model,batch,pair['prefill'],path,loop=pg=='pages'))
            config=json.loads((run/'config.json').read_text())
            for row in rows:row['model_path']=config['model']['path']
            rows.insert(0,dict(label='vllm',kind='vllm',root=str(ROOT),python='/root/venv_vllm/bin/python',model=model,model_path=config['model']['path'],batch=batch,env={},available=True))
            result[cell]=rows
    (HERE/'arms.json').write_text(json.dumps(result,indent=2)+'\n')
    print('registered '+str(sum(len(r) for r in result.values()))+' arms')
if __name__=='__main__':main()
