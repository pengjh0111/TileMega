#!/usr/bin/env python3
"""Define all final paired rounds and correctness before queue publication."""
import json,math
from make_phase0 import HERE,ROOT,PYTHON,write
from choose_r14 import CELLS

def queue(total_mib):
    steps=[];driver=HERE/'phase_d_final.py'
    env=dict(PYTHONPATH=str(ROOT/'python'),TILEMEGA_BIN=str(ROOT/'build-phase12/tools/tilemega'))
    def step(name,args,gpu,after,priority,timeout=3600,free=12288,retry_args=()):
        steps.append(dict(name=name,command=list(map(str,args)),gpu=gpu,after=list(after),
            priority=priority,timeout_s=timeout,needs_free_mib=free,cwd=str(ROOT),env=env,retry_args=list(retry_args)))
    required=['D1_smoke_llama_v2','D1_smoke_qwen3_v2','D1_family_llama_v2','D1_family_qwen3_v2','D1_reconfirm_llama_B16_v2']
    finished=[]
    for cell in CELLS:
        rounds=[];checks=[]
        for r in range(3):
            name=f'D2_{cell}_r{r}';check=f'D2_validate_{cell}_r{r}'
            step(name,[PYTHON,HERE/'anchor.py','--arms',HERE/'phase_d_final_arms.json','--cell',cell,'--round',r,'--out',HERE/f'raw/{name}'],True,required,105,free=math.ceil(.85*total_mib)+1024)
            step(check,[PYTHON,driver,'validate_round','--cell',cell,'--round',r],False,[name],105,300)
            rounds.append(name);checks.append(check)
        canary=f'D2_canary_{cell}';c1=f'D3_c1_{cell}';c2=f'D3_c2_{cell}';protocol=f'D3_protocol_{cell}'
        step(canary,[PYTHON,driver,'canary','--cell',cell],False,checks,106,300)
        step(c2,[PYTHON,driver,'c2','--cell',cell],True,required,110,7200)
        step(c1,[PYTHON,driver,'c1','--cell',cell],True,checks,111,9000)
        step(protocol,[PYTHON,ROOT/'docs/experiments/SERVING_R11/check_protocol.py','--cases',HERE/f'{protocol}_cases.json','--processes','50','--out',HERE/f'raw/{protocol}'],True,required,115,7200,retry_args=['--resume'])
        finished += [canary,c1,c2,protocol]
    step('D_final_analyze',[PYTHON,driver,'analyze'],False,finished,120,600)
    return steps

def main():
    # Device memory comes from a recorded property query, never an architecture constant.
    rows=[json.loads(line) for line in (HERE/'raw/D0/guard.jsonl').read_text().splitlines()]
    totals={row['total_mib'] for row in rows if row.get('phase')=='preflight'}
    if len(totals)!=1:raise ValueError('ambiguous recorded device memory')
    arms=json.loads((HERE/'phase_d_final_arms.json').read_text())
    if set(arms)!=set(CELLS):raise ValueError('missing final comparison cell')
    for cell in CELLS:
        if {a['label'] for a in arms[cell]}!={'vllm','R13D','baseline','R14F'}:raise ValueError('missing final arm')
        if not (HERE/f'D3_protocol_{cell}_cases.json').exists():raise ValueError('protocol case not prepared')
    steps=queue(totals.pop());write(HERE/'queue_phase_d_final.pending.json',steps)
    print(f'{len(steps)} final steps prepared: 12 paired rounds, 4 C-1, 4 C-2, 4 protocols and CPU acceptance')
if __name__=='__main__':main()
