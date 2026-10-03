#!/usr/bin/env python3
"""Pre-registered choices; absent data never satisfies a trigger."""
import argparse, json, statistics
from pathlib import Path

CELLS=('llama_B1','llama_B16','qwen3_B1','qwen3_B16')
def summary(values):
    if not values: raise ValueError('empty measurement series')
    return statistics.median(values),max(values)-min(values)
def distinguishably_faster(candidate,base):
    c,cr=summary(candidate);b,br=summary(base)
    return b-c>max(cr,br)
def prefill(data):
    return {cell:('PF-R10-noWD' if data.get(cell,{}).get('B0-pfR10') and
        distinguishably_faster(data[cell]['B0-pfR10'],data[cell]['B0-pfR12b'])
        else 'PF-R12b-noWD') for cell in CELLS}
def phase_c(data):
    mb=data.get('mb',{}); cells=data.get('cells',{})
    def count(predicate): return sum(predicate(v) for v in cells.values())
    def present(v,*keys):return all(k in v for k in keys)
    ceiling=mb.get('ceiling_gbps',0)
    rules={
      'C-L2a':count(lambda v:present(v,'NL2g_rel','NL2e_rel') and v['NL2g_rel']<=.03 and v['NL2e_rel']>=.08)>=3,
      'C-L2b':count(lambda v:'NL2g_rel' in v and v['NL2g_rel']>=.05)>=2,
      'C-PG1':bool(ceiling and 'loader1_gbps' in mb and 'loader2_gbps' in mb and mb['loader1_gbps']<.95*ceiling and mb['loader2_gbps']>=.98*ceiling),
      'C-PG3':any(v.get('PR_dependency_fraction',0)>.10 for v in cells.values()),
      'C-LP2':any(v.get('PR_loop_rel',0)>=.01 for v in cells.values()),
      'C-AT':any(v.get('attention_merge_excess_fraction',0)>=.03 for k,v in cells.items() if k.startswith('qwen3')),
      'C-BW':mb.get('gemv_best_instruction_rel',0)>.02,
      'C-WL':mb.get('gemv_tile_layout_rel',0)>.02,
    }
    return {'rules':rules,'selected':[k for k,v in rules.items() if v],
            'inputs':data,'gpu_budget_hours':6,'engineering_budget_days':5}
def final(data):
    return {cell:('R13F' if data.get(cell,{}).get('R13F') and
        distinguishably_faster(data[cell]['R13F'],data[cell]['B0-D']) else 'B0-D') for cell in CELLS}
def loop(data):
    cells=data.get('cells',{});changes={};missing=[]
    for cell in CELLS:
        row=cells.get(cell,{})
        if len(row.get('B0l',[]))!=3 or len(row.get('B0-noev',[]))!=3:
            missing.append(cell);continue
        changes[cell]=statistics.median(row['B0l'])/statistics.median(row['B0-noev'])-1
    return dict(exclude_l1_loop=bool(missing or any(v>.005 for v in changes.values())),
        relative_changes=changes,missing_cells=missing,rule='any cell worse than +0.5%; missing evidence does not enable a default loop')
def apply_prefill_pins(here):
    choices=json.loads((here/'defaults_r13.json').read_text())['prefill']
    builds={(r['cell'],r['label']):r for r in json.loads((here/'builds_a.json').read_text())}
    for model in ('llama','qwen3'):
        path=here.parents[2]/f'configs/e2e/{model}_r13.json';config=json.loads(path.read_text())
        config['solver']['prefill_pins']={}
        for batch in (1,16):
            row=builds.get((f'{model}_B{batch}',choices[f'{model}_B{batch}']))
            if not row or row['exit_code']:raise RuntimeError('selected prefill build unavailable')
            config['solver']['prefill_pins'][str(batch)]={'manifest':row['so']+'.plan.json','classes':row['so']+'.classes.tsv'}
        path.write_text(json.dumps(config,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prefill','phase_c','final','loop','apply-prefill'),required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.stage=='apply-prefill':
        apply_prefill_pins(Path(__file__).resolve().parent);return
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(globals()[a.stage](json.loads(a.input.read_text())),indent=2)+'\n')
if __name__=='__main__':main()
