#!/usr/bin/env python3
"""Recompute every R7 gate from raw logs and dumps, and print PASS/FAIL.

Nothing here reads a summary or a conclusion: each gate re-derives its number
from the process logs, the run manifests and the generated sources. Hard-gate
failures set the exit status, but every gate is evaluated first.
"""
import json,re,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
MODELS2=REPO/'docs/experiments/MODELS2'
rows=[]
def gate(name,hard,ok,detail,evidence):
    rows.append(dict(name=name,hard=hard,ok=bool(ok),detail=detail,evidence=str(evidence)))

def processes(folder):
    """(rounds, passing, distinct binaries, failing outputs) from raw logs."""
    logs=sorted(folder.glob('r*.log'),key=lambda p:int(re.findall(r'\d+',p.name)[0]))
    ok=bad=0;binaries=set()
    for log in logs:
        text=log.read_text()
        if 'RESULT status=PASS' in text: ok+=1
        for line in re.findall(r'^E2E_OUTPUT_DIFF .*$',text,re.M):
            if 'mismatch=0' not in line: bad+=1
        meta=log.with_suffix('.json')
        if meta.exists(): binaries.add(json.loads(meta.read_text()).get('binary_sha256'))
    return len(logs),ok,binaries,bad

# --- A-a: the maximal connected Llama graph, 50 fresh processes -------------
folder=MODELS2/'admission/admitted2/correctness'
if folder.is_dir():
    n,ok,binaries,bad=processes(folder)
    gate('A-a maximal connected Llama graph 50/50',True,n>=50 and ok==n and bad==0,
         f'rounds={n} passing={ok} failing_outputs={bad} binaries={len(binaries)}',folder)
else:
    gate('A-a maximal connected Llama graph 50/50',True,False,'no admission logs',folder)

# --- A-e: the epsilon and RoPE ablations ------------------------------------
table=MODELS2/'ablation/admitted2/ablation.tsv'
if table.is_file():
    lines=[l.split('\t') for l in table.read_text().strip().splitlines()]
    head,body={k:i for i,k in enumerate(lines[0])},lines[1:]
    by={r[head['arm']]:r for r in body}
    unchanged=all(by[a][head['v_0_463_gpu']]==by['base'][head['v_0_463_gpu']]
                  for a in ('a1','a2','a1a2') if a in by)
    fixed='refine' in by and by['refine'][head['v_0_463_gpu']]==by['refine'][head['v_0_463_fp64_rounded']]
    gate('A-e epsilon/RoPE ablation reported',False,unchanged and fixed,
         'A1, A2 and both leave V[0,463] at '+by['base'][head['v_0_463_gpu']]+
         '; refinement gives '+by.get('refine',['?'])[head['v_0_463_gpu']],table)
else:
    gate('A-e epsilon/RoPE ablation reported',False,False,'no ablation table',table)

# --- A-d: the extension cost, against R6's audited 15 sites -----------------
sites=REPO/'docs/experiments/MODELS/extension_sites.tsv'
audited=len(sites.read_text().strip().splitlines())-1 if sites.is_file() else 0
actual=HERE/'extension_cost.tsv'
if actual.is_file():
    got=[l.split('\t') for l in actual.read_text().strip().splitlines()][1:]
    gate('A-d extension cost table updated',False,bool(got),
         f'audited={audited} measured='+', '.join(f'{r[0]}:{r[1]}' for r in got),actual)
else:
    gate('A-d extension cost table updated',False,False,'not recorded',actual)

# --- D-a: the whole model, 50 fresh processes -------------------------------
for name in ('llama','qwen3'):
    folder=HERE/name/'correctness'
    if folder.is_dir():
        n,ok,binaries,bad=processes(folder)
        gate(f'D-a {name} whole model 50/50',True,n>=50 and ok==n and bad==0,
             f'rounds={n} passing={ok} failing_outputs={bad} binaries={len(binaries)}',folder)
    else:
        gate(f'D-a {name} whole model 50/50',True,False,'not run this round',folder)

# --- H2: the default build's SASS, against the baseline tree ----------------
stamp=HERE/'sass_identity/manifest.json'
if stamp.is_file():
    meta=json.loads(stamp.read_text())
    same=all(m['sha256']==m['baseline_sha256'] for m in meta['models'].values())
    gate('H2 default-build SASS identity',True,same,
         'models='+', '.join(f"{k}:{'same' if v['sha256']==v['baseline_sha256'] else 'differs'}"
                             for k,v in meta['models'].items()),stamp)
else:
    gate('H2 default-build SASS identity',True,False,'no stamp',stamp)

# --- B0 / B1 / B2 / B3 / C1-b: not implemented this round -------------------
for name in ('B0 FORK7 whole-pipeline exposed wait','B1 paging and cross-task pipelining',
             'B2 per-stage kappa','B3 intra-interval geometry',
             'C1-b preparation-phase optimization'):
    gate(name,True,False,'not implemented this round; see the stoppage ledger',
         HERE/'summary.md')

width=max(len(r['name']) for r in rows)
failed=0
for r in rows:
    mark='PASS' if r['ok'] else 'FAIL'
    if not r['ok'] and r['hard']: failed+=1
    print(f"{mark} [{'hard' if r['hard'] else 'report'}] {r['name']:<{width}}  {r['detail']}")
    print(f"       evidence: {r['evidence']}")
print(f"\n{len(rows)-failed}/{len(rows)} gates satisfied; {failed} hard gate(s) failed")
sys.exit(1 if failed else 0)
