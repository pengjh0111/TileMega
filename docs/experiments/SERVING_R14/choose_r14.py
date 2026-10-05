#!/usr/bin/env python3
"""R14 preregistered selection; missing observations are pending, never zero."""
import argparse,json,statistics
from pathlib import Path
CELLS=('llama_B1','llama_B16','qwen3_B1','qwen3_B16')
FLOOR_RATIOS=(1.083,1.162,1.061,1.107)
RULES={
 'C-AT3b':dict(metric='attention_page_wait_fraction',threshold=.30,comparison='ge',gain=.015,minimum_cells=1,predicate='paged',fresh_processes=50),
 'C-RW1':dict(metric='gemm_over_load_fraction',threshold=.20,comparison='gt',gain=.02,minimum_cells=1,predicate='nonpaged_B16',fresh_processes=0),
 'C-RW2':dict(metric='gemm_over_load_fraction',threshold=.20,comparison='gt',gain=.02,minimum_cells=1,predicate='paged_B16',extra='pipe_recovers_fraction_lt_half',fresh_processes=50,loader_floor_fraction=.97),
 'C-PF':dict(metric='family_integral_gain',threshold=.02,comparison='ge',gain=.01,minimum_cells=1,predicate='different_past_winner',extra='compatible_kv_packing',fresh_processes=0),
 'C-EP2':dict(metric='epilogue_task_fraction',threshold=.08,comparison='ge',gain=.01,minimum_cells=1,predicate='B16',fresh_processes=0),
 'C-AT4':dict(metric='attention_merge_us_per_layer',threshold=8,comparison='ge',gain=.015,minimum_cells=1,predicate='B1',fresh_processes=50),
 'C-GV2':dict(metric='nonpaged_gemv_selected',threshold=1,comparison='ge',gain=.01,minimum_cells=1,predicate='B1',extra='paged_faster',fresh_processes=0),
}
def stats(values):
    if len(values)!=3:raise ValueError('selection requires three rounds')
    return statistics.median(values),max(values)-min(values)
def discernible(candidate,baseline):
    c,cr=stats(candidate);b,br=stats(baseline)
    return b-c>max(cr,br)
def baseline(data):
    selected={cell:"R13D'" for cell in CELLS}
    for cell in CELLS:
        if not data[cell]["R13D'"].get('c1') or not data[cell]["R13D'"].get('r13_tokens_identical'):
            raise ValueError(f'{cell}: baseline correctness is not closed')
    n=data['llama_B1'].get("N1'")
    if n and n.get('c1') and discernible(n['tpot'],data['llama_B1']["R13D'"]['tpot']):selected['llama_B1']="N1'"
    return selected

def phase_c(data):
    result={}
    for name,rule in RULES.items():
        rows=data.get(name)
        if rows is None:result[name]={'status':'pending','rule':rule};continue
        triggered=[]
        for cell,obs in rows.items():
            if not obs.get('applicable',False):continue
            if rule['metric'] not in obs:raise ValueError(f'missing {name}/{cell} metric')
            value=obs[rule['metric']]
            yes=value>=rule['threshold'] if rule['comparison']=='ge' else value>rule['threshold']
            if rule.get('extra'):yes=yes and obs.get(rule['extra'],False)
            if yes:triggered.append(cell)
        result[name]={'status':'triggered' if triggered else 'not_triggered','cells':triggered,'rule':rule}
    return result

def retention(name,rows):
    rule=RULES[name];good=[]
    for cell,row in rows.items():
        if not row.get('correct'):continue
        if rule['fresh_processes'] and row.get('fresh_passed',0)<rule['fresh_processes']:continue
        if 'loader_floor_fraction' in rule and row.get('loader_floor_fraction',0)<rule['loader_floor_fraction']:continue
        b,_=stats(row['baseline']);c,_=stats(row['candidate'])
        if (b-c)/b>rule['gain']:good.append(cell)
    return {'retain':len(good)>=rule['minimum_cells'],'cells':good}

def final(data):
    result={}
    for cell,minimum in zip(CELLS,FLOOR_RATIOS):
        row=data[cell];candidate=row['R14F'];base=row['baseline']
        selected='R14F' if candidate.get('c1') and candidate.get('c2') and discernible(candidate['tpot'],base['tpot']) else 'baseline'
        winner=row[selected]
        if not winner.get('c1') or not winner.get('c2'):raise ValueError(f'{cell}: final correctness failed')
        ratio=statistics.median(winner['paired_vllm_ratios'])
        result[cell]={'selected':selected,'ratio':ratio,'minimum':minimum,'gate_pass':ratio>=minimum}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('baseline','phase-c','final'))
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.input.read_text());result={'baseline':baseline,'phase-c':phase_c,'final':final}[a.stage](data)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
