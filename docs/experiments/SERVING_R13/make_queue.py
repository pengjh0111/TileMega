#!/usr/bin/env python3
"""Expand the prespecified Phase A matrix; Phase B is added after freezing."""
import argparse,copy,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PY='/root/venvs/tilemega-torch213-cu126/bin/python'
def main():
    p=argparse.ArgumentParser();p.add_argument('--diag-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    arms=json.loads((HERE/'arms.json').read_text());defs=HERE/'definitions';defs.mkdir(parents=True,exist_ok=True)
    for rows in arms.values():
        for r in rows:
            if r['kind']!='tm_old':r['root']=str(a.diag_root)
    def definition(name,data):
        path=defs/(name+'.json');path.write_text(json.dumps(data,indent=2)+'\n');return path
    steps=[]
    def add(name,cmd,priority,needs=12288):
        steps.append(dict(name=name,command=cmd,cwd=str(a.diag_root),env={'PYTHONPATH':str(a.diag_root/'python')},gpu=True,priority=priority,after=[],timeout_s=7200,needs_free_mib=needs,out=str(HERE/'raw'/name)))
    cells=['llama_B1','qwen3_B1','llama_B16','qwen3_B16']
    # Query total memory only; it does not create a CUDA context or timing data.
    import subprocess
    total=float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).splitlines()[0]);vlneeds=math.ceil(.85*total)+1024
    path=definition('a1',arms)
    for cell in cells:
        for rnd in range(3):add(f'A1_{cell}_r{rnd}',[PY,str(a.diag_root/'docs/experiments/SERVING_R13/anchor.py'),'--arms',str(path),'--cell',cell,'--round',str(rnd),'--out',str(HERE/'raw'/f'A1_{cell}_r{rnd}')],10 if cell.endswith('_B1') else 20,vlneeds)
    for matrix,selected in [('A4',['R12bN_L2']),('A3',['S1P'])]:
        data={}
        for cell,rows in arms.items():
            if matrix=='A4' and not cell.endswith('_B1'):continue
            if matrix=='A3' and not cell.startswith('llama'):continue
            base=next(r for r in rows if r['label']==selected[0]);data[cell]=[]
            labels=['R12bN_L2','R12bN_L2_nowd'] if matrix=='A4' else ['loop','separate','rotate','kphase0','L1','nowd']
            for label in labels:
                row=copy.deepcopy(base);row['label']=label
                if label in ('nowd','R12bN_L2_nowd'):row['env']['TILEMEGA_WATCHDOG_MS']='0'
                if label=='separate':row['decode_loop']=False
                if label=='L1':row.update(mode='L1',decode_loop=False)
                if label=='rotate':row['env']['TILEMEGA_PLACEMENT_ABLATION']='rotate'
                if label=='kphase0':row['env']['TILEMEGA_KPHASE_MASK']='0'
                data[cell].append(row)
        path=definition(matrix.lower(),data)
        for cell in data:
            for rnd in range(3):add(f'{matrix}_{cell}_r{rnd}',[PY,str(a.diag_root/'docs/experiments/SERVING_R13/anchor.py'),'--arms',str(path),'--cell',cell,'--round',str(rnd),'--out',str(HERE/'raw'/f'{matrix}_{cell}_r{rnd}')],25 if matrix=='A4' else 30)
    path=definition('a2',arms)
    for cell in cells:
        for rnd in range(3):add(f'A2_{cell}_r{rnd}',[PY,str(a.diag_root/'docs/experiments/SERVING_R13/candidates.py'),'--arms',str(path),'--cell',cell,'--round',str(rnd),'--out',str(HERE/'raw'/f'A2_{cell}_r{rnd}')],40)
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'queue_a.json').write_text(json.dumps(steps,indent=2)+'\n');print(f'Phase A: {len(steps)} steps')
if __name__=='__main__':main()
