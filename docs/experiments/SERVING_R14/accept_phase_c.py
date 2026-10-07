#!/usr/bin/env python3
"""Review completed conditional trials without turning slow trials into defaults."""
import csv,json,statistics,tarfile
from pathlib import Path
from choose_r14 import retention
from tilemega.build.identity import verify,sha,digest
from make_phase0 import HERE,write

def review():
    state=json.loads((HERE/'scheduler/state.json').read_text());guards=[]
    for step in json.loads((HERE/'queue/queue_phase_c.json').read_text()):
        record=state[step['name']]
        if step['name']=='C_arch':
            if record['status']!='failed':raise ValueError('unexpected original architecture status')
            continue
        if record['status']!='done' or record['exit_code']:raise ValueError(step['name'])
        if step['gpu']:
            guard=json.loads((HERE/'raw'/step['name']/'guard_result.json').read_text())
            if guard['code']:raise ValueError('guard rejected '+step['name'])
            guards.append(dict(step=step['name'],attempts=record['attempts'],result=guard))
    rows=[];decisions={};canaries=[]
    for cell,name,label in [('llama_B16','C-RW1','C_RW1'),('qwen3_B16','C-EP2','C_EP2'),('qwen3_B1','C-AT4','C_AT4')]:
        arms=json.loads((HERE/f'C_{cell}_accepted.json').read_text());identities={a['label']:verify(a['decode']) for a in arms}
        samples={a['label']:[] for a in arms};tokens={a['label']:[] for a in arms};execution={}
        for r in range(3):
            result=json.loads((HERE/f'raw/C_matrix_{cell}_r{r}/{cell}/round{r}.json').read_text())['arms']
            for arm,x in result.items():
                i=x['execution_identity']
                if x['exit_code'] or i['trace'] or i['artifact_id']!=identities[arm]['artifact_id'] or digest({k:v for k,v in i.items() if k!='execution_id'})!=i['execution_id']:raise ValueError('invalid measured identity')
                if arm in execution and execution[arm]!=i:raise ValueError('execution changed')
                execution[arm]=i;samples[arm].append((x['e2e_seconds']-x['ttft_seconds'])*1000/1023)
                tokens[arm].append(next(v['tokens'] for v in x['runs'] if v['N']==1024 and not v['warmup']))
        correct=json.loads((HERE/f'raw/C_correct_{cell}/results.json').read_text())
        if any(not r['c1'] or not r['c2'] or r['same_as_baseline'] is False for r in correct):raise ValueError('correctness failed')
        if any(any(t!=ts[0] for t in ts) for ts in tokens.values()):raise ValueError('timed tokens changed')
        protocol=HERE/f'raw/C_protocol_{cell}/processes/summary.json';passed=0
        if protocol.exists():
            p=json.loads(protocol.read_text())
            if not p['complete'] or p['passed']!=50 or p['failed']:raise ValueError('protocol failed')
            passed=p['passed']
        baseline=statistics.median(samples['baseline'])
        for r,x in enumerate(samples['baseline']):
            if abs(x/baseline-1)>.02:canaries.append(dict(cell=cell,round=r,deviation=x/baseline-1))
        decision=retention(name,{cell:dict(correct=True,baseline=samples['baseline'],candidate=samples[label],fresh_passed=passed)})
        decisions[name]=dict(**decision,cell=cell,relative=statistics.median(samples[label])/baseline-1,baseline=samples['baseline'],candidate=samples[label],fresh_passed=passed,correct=True)
        for arm,ts in samples.items():rows.append(dict(cell=cell,label=arm,tpot_ms=statistics.median(ts),range_ms=max(ts)-min(ts),relative=statistics.median(ts)/baseline-1,execution_id=execution[arm]['execution_id'],spill=execution[arm]['spill'],smem=identities[arm]['shared_memory_bytes'],samples=json.dumps(ts)))
    with (HERE/'results/T8_phase_c.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
    write(HERE/'phase_c_retention.json',decisions)
    write(HERE/'results/phase_c_acceptance.json',dict(steps_done=21,architecture_retry_required=True,guards=guards,canary_rounds=canaries,retention=decisions))
    print(json.dumps({k:dict(retain=v['retain'],relative=v['relative'],fresh_passed=v['fresh_passed']) for k,v in decisions.items()}))

def archive():
    paths=set()
    for step in json.loads((HERE/'queue/queue_phase_c.json').read_text()):
        if step['name']=='C_archive':continue
        paths.update(p for p in (HERE/'raw'/step['name']).rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.tsv','.log','.txt','.patch'))
        paths.update(p for p in (HERE/'scheduler').glob(step['name']+'.*') if p.suffix in ('.log','.done','.failed'))
    paths.update([HERE/'phase_c_retention.json',HERE/'results/T8_phase_c.tsv',HERE/'results/phase_c_acceptance.json'])
    with (HERE/'raw/phase_c_evidence_manifest.tsv').open('w') as f:
        f.write('path\tsha256\tbytes\n')
        for p in sorted(paths):f.write(f'{p.relative_to(HERE)}\t{sha(p)}\t{p.stat().st_size}\n')
    with tarfile.open(HERE/'raw/phase_c_completed.tar.xz','w:xz',preset=1) as tar:
        for p in sorted(paths):tar.add(p,arcname=str(p.relative_to(HERE)),recursive=False)
    print('Phase C evidence archived, including original architecture failure')
if __name__=='__main__':
    import sys
    archive() if '--archive' in sys.argv else review()
