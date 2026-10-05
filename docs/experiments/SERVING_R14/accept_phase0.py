#!/usr/bin/env python3
"""Archive completed Phase-0 evidence and separate passes from timing certainty."""
import hashlib,json,statistics,subprocess,sys,tarfile
from pathlib import Path
from tilemega.build.identity import verify,sha
HERE=Path(__file__).resolve().parent

def main():
    state=json.loads((HERE/'scheduler/state.json').read_text());checks={}
    names=['P0_correctness','P0_build','P0_smoke','P0_trace_round0','P0_trace_round1','P0_trace_round2','P0_trace_analyze','RW3_numeric','RW3_paged_numeric']
    for n in names:
        if state[n]['status']!='done' or state[n]['exit_code']!=0:raise ValueError(f'{n} not successful')
        checks[n]=dict(status='passed',attempts=state[n]['attempts'])
    smoke=json.loads((HERE/'raw/P0_smoke/smoke.json').read_text())
    if not smoke['pass'] or any(n for a in smoke['mismatches'].values() for n in a.values()):raise ValueError('smoke mismatch')
    for r in range(3):
        guard=json.loads((HERE/f'raw/P0_trace_round{r}/guard_result.json').read_text())
        if guard['code']!=0:raise ValueError('unclean final diagnostic attempt')
    diagnostics=json.loads((HERE/'results/T2_diagnostic.json').read_text());details=[]
    for label,row in diagnostics.items():
        so=HERE/'raw/P0_build/qwen3_B16'/label/'plan.so';identity=verify(so)
        if identity['artifact_id']!=row['identity']['artifact_id']:raise ValueError('measurement/build identity mismatch')
        med=statistics.median(row['tpot']);outliers=[r for r,t in enumerate(row['tpot']) if abs(t/med-1)>.02]
        details.append(dict(label=label,tpot_ms=[x*1000 for x in row['tpot']],median_overhead=row['relative_to_uninstrumented'],
            outlier_rounds=outliers,artifact_id=identity['artifact_id'],resources=identity['resources'][row['identity']['kernel']]))
    subprocess.run([sys.executable,str(HERE/'summarize_flow.py')],check=True)
    result=dict(checks=checks,fx24=json.loads((HERE/'results/T12_audit.json').read_text()),diagnostics=details,
        acceptance='correctness/build/smoke prerequisites pass; TR-4 performance acceptance remains pending Phase A',
        timing_limit='Guard accepted final attempts, but non-base outliers prevent stable overhead attribution. Retain every observation.')
    (HERE/'results/phase0_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    roots=[HERE/'raw'/n for n in ['P0_build','P0_smoke','P0_trace_round0','P0_trace_round1','P0_trace_round2','P0_trace_analyze','phase0_checks','RW3_numeric','RW3_paged_numeric','FX24/after']]
    selected=[]
    for root in roots:
        for f in sorted(root.rglob('*')):
            if f.is_file() and f.suffix in ('.json','.tsv','.log','.txt','.gz','.patch') and not f.name.endswith('.cu'):
                selected.append(f)
    selected+=list((HERE/'scheduler').glob('P0*.log'))+list((HERE/'scheduler').glob('RW3*.log'))
    manifest=HERE/'raw/phase0_evidence_manifest.tsv'
    with manifest.open('w') as out:
        out.write('path\tsha256\tbytes\n')
        for f in selected:out.write(f'{f.relative_to(HERE)}\t{sha(f)}\t{f.stat().st_size}\n')
    with tarfile.open(HERE/'raw/phase0_completed.tar.xz','w:xz',preset=1) as tar:
        for f in selected:tar.add(f,arcname=str(f.relative_to(HERE)),recursive=False)
    print('Phase-0 prerequisites verified and archived; trace confidence qualified')
if __name__=='__main__':main()
