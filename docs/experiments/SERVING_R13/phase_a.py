#!/usr/bin/env python3
"""Dispatch the registered Phase A measurements without modifying binaries."""
import argparse,copy,json,os,subprocess,sys
from pathlib import Path
from arms import register
from choose_r13 import prefill
HERE=Path(__file__).resolve().parent
DATA=Path(os.environ.get('TILEMEGA_R13_DATA',HERE))
ROOT=Path(os.environ.get('TILEMEGA_R13_ROOT',HERE.parents[2]))
CELLS=('llama_B1','qwen3_B1','llama_B16','qwen3_B16')
LABELS={'A1':('vllm','R10C','R12bN_L1','B0-pfR12b','B0-pfR10','B0-noev'),
        'A2':('B0','B0h','NL2e','PR_L1','PR_L2','PR_L2l')}

def arms_for(matrix):
    originals=json.loads((DATA/'arms.json').read_text());rows=json.loads((DATA/'builds_a.json').read_text())
    builds={(r['cell'],r['label']):r for r in rows}
    result={}
    for cell in CELLS:
        model,batch=cell.split('_B');batch=int(batch)
        old={r['label']:r for r in originals[cell]};result[cell]=[]
        for label in LABELS[matrix]:
            if label in old:
                row=copy.deepcopy(old[label])
                if row['kind']=='tm':row['root']=str(HERE.parents[2])
                row['decode_loop']=False
            else:
                decode='P-R12bN-noWD' if label.startswith('PR_') else 'N-R12bh-noWD' if label=='B0h' else 'N-R12b-noWD'
                if model=='qwen3' and batch==16 and label=='B0h':decode='N-R12b-noWD'
                pf='PF-R10-noWD' if label=='B0-pfR10' else 'PF-R12b-noWD'
                d=builds[(cell,decode)];p=builds[(cell,pf)]
                row=register(label,model,batch,p['so'],d['so'],root=HERE.parents[2],
                             mode='L2' if label in ('NL2e','PR_L2','PR_L2l') else 'L1',
                             loop=label=='PR_L2l')
                row['available']=row['available'] and d['exit_code']==0 and p['exit_code']==0
                row['model_path']=old['R12bN_L1']['model_path']
                row['compiler_commit']=d['compiler_commit'];row['target']=d['target']
                if label=='B0-noev':row['step_events']=0
            row['prefill_mode']='L1';result[cell].append(row)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('smoke','anchor','prefill'))
    p.add_argument('--matrix',choices=LABELS);p.add_argument('--cell',choices=CELLS)
    p.add_argument('--round',type=int,default=0);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.action=='anchor':
        arms=arms_for(a.matrix);path=a.out/'arms.json';path.write_text(json.dumps(arms,indent=2)+'\n')
        return subprocess.run([sys.executable,str(HERE/'anchor.py'),'--arms',str(path),'--cell',a.cell,'--round',str(a.round),'--out',str(a.out)]).returncode
    if a.action=='smoke':
        rows=json.loads((DATA/'builds_a.json').read_text());originals=json.loads((DATA/'arms.json').read_text());results=[]
        for row in rows:
            if row['phase']!='decode' or row['exit_code']:continue
            old=next(r for r in originals[row['cell']] if r['label']=='R12bN_L1')
            cmd=[sys.executable,'-m','tilemega.serving.smoke','--so',row['so'],
                 '--model',old['model_path'],'--batch',str(row['batch']),'--steps','64',
                 '--out',str(a.out/row['cell']/row['label'])]
            code=subprocess.run(cmd).returncode
            if code==75:return 75
            results.append(dict(cell=row['cell'],label=row['label'],exit_code=code))
            if code:
                row['exit_code']=code;row['reason']='A0s smoke failure';Path(row['out'],'record.json').write_text(json.dumps(row,indent=2)+'\n')
        (DATA/'builds_a.json').write_text(json.dumps(rows,indent=2)+'\n')
        (a.out/'smoke_results.json').write_text(json.dumps(results,indent=2)+'\n')
        print('smokes passed '+str(sum(r['exit_code']==0 for r in results))+'/'+str(len(results)));return 0
    values={}
    for path in (DATA/'raw').glob('A1_*/**/round*.json'):
        if path.parent.name not in CELLS:continue
        d=json.loads(path.read_text())
        if d.get('invalidated'):continue
        for label,row in d['arms'].items():
            if row.get('exit_code')==0 and label in ('B0-pfR12b','B0-pfR10'):
                values.setdefault(d['cell'],{}).setdefault(label,[]).append(row['ttft_seconds'])
    result=dict(prefill= prefill(values),raw_values=values)
    (DATA/'defaults_r13.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())
