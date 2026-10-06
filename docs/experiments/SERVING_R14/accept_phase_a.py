#!/usr/bin/env python3
"""Accept guarded Phase A, preserve identity evidence and qualify predictions."""
import json,statistics,tarfile
from pathlib import Path
from tilemega.build.identity import sha,verify
HERE=Path(__file__).resolve().parent

def main():
    queue=json.loads((HERE/'queue/queue_phase_a.json').read_text())
    state=json.loads((HERE/'scheduler/state.json').read_text())
    for step in queue:
        r=state[step['name']]
        if r['status']!='done' or r['exit_code']!=0:raise ValueError(step['name'])
        if step.get('gpu'):
            g=json.loads((HERE/'raw'/step['name']/'guard_result.json').read_text())
            if g['code']!=0:raise ValueError('guard rejected '+step['name'])
    rows=json.loads((HERE/'results/T1_anchor.json').read_text());group={}
    for row in rows:group.setdefault(row['cell'],{})[row['label']]=row
    report={}
    for cell,arms in group.items():
        c2=json.loads((HERE/'raw'/('A3_'+cell)/'c2.json').read_text())
        if not c2['r13_tokens_identical'] or not c2['timed_repeats_identical']:raise ValueError('token regression')
        checks={p.stem:json.loads(p.read_text()) for p in (HERE/'raw'/('A3_'+cell)).glob('*_hf.json')}
        if not all(x['pass'] for x in checks.values()):raise ValueError('C-1 failed')
        measurements={}
        for label,row in arms.items():
            runs=[json.loads(Path(p).read_text())['arms'][label] for p in row['raw']]
            ratios=[r['e2e_seconds']/v['e2e_seconds'] for r,v in zip(
                [json.loads(Path(p).read_text())['arms']['vllm'] for p in row['raw']],runs)]
            measurements[label]=dict(tpot_ms=row['tpot_median']*1000,range_ms=row['tpot_range']*1000,
                ttft_ms=statistics.median(x['ttft_seconds']*1000 for x in runs),
                e2e_seconds=statistics.median(x['e2e_seconds'] for x in runs),
                tm_vllm=statistics.median(ratios),ratio_range=max(ratios)-min(ratios))
        drift=arms['R13D_prime']['tpot_median']/arms['R13D']['tpot_median']-1
        # Report canaries; never silently remove a slow observation.
        canary=arms['vllm'];flags=[i for i,v in enumerate(canary['tpot_samples']) if abs(v/canary['tpot_median']-1)>.02]
        report[cell]=dict(metrics=measurements,rebuild_relative=drift,prediction_met=abs(drift)<=.005,c2=c2,
            c1={k:{f:v[f] for f in ('pass','gap_le_0_5_ratio','max_gap','required_ratio','allowed_maximum')} for k,v in checks.items()},
            canary_rounds=flags)
    for job in json.loads((HERE/'phase_a_builds.json').read_text()):verify(Path(job['out'])/'plan.so')
    result=dict(steps=len(queue),cells=report,baselines=json.loads((HERE/'baselines_r14.json').read_text()),
        trace=json.loads((HERE/'results/T2_trace_overhead.json').read_text()),
        qualification='Model correctness passes. Rebuild slowdown exceeds the prediction; no synchronization reliability claim. Guard rejection/retry records preserved.')
    (HERE/'results/phase_a_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    selected=set()
    for name in [s['name'] for s in queue]:
        for p in (HERE/'raw'/name).rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.tsv','.log','.txt','.patch'):
                selected.add(p)
        for p in (HERE/'scheduler').glob(name+'.*'):
            if p.suffix in ('.log','.done'):selected.add(p)
    for p in (HERE/'results').glob('*.json'):selected.add(p)
    selected.update([HERE/'scheduler/progress.tsv',HERE/'scheduler/state.json'])
    with (HERE/'raw/phase_a_evidence_manifest.tsv').open('w') as f:
        f.write('path\tsha256\tbytes\n')
        for p in sorted(selected):f.write(f'{p.relative_to(HERE)}\t{sha(p)}\t{p.stat().st_size}\n')
    with tarfile.open(HERE/'raw/phase_a_completed.tar.xz','w:xz',preset=1) as tar:
        for p in sorted(selected):tar.add(p,arcname=str(p.relative_to(HERE)),recursive=False)
    print('Phase A: 36/36 completed, C-1/C-2 pass; rebuild drift retained')
if __name__=='__main__':main()
