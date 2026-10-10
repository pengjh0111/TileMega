#!/usr/bin/env python3
"""Close R14 from existing evidence; never launch GPU work or invent a winner."""
import argparse,collections,csv,hashlib,json,statistics,tarfile
from pathlib import Path
from make_phase0 import HERE,ROOT,write
from tilemega.build.identity import bind_execution,sha,verify
from ledger_r14 import stage_ledger,tasks

def load(p):return json.loads(Path(p).read_text())
def tsv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys,delimiter='\t');w.writeheader();w.writerows(rows)

def review():
    state=load(HERE/'scheduler_remaining/state.json')
    counts=dict(collections.Counter(v['status'] for v in state.values()))
    if any(v['status'] in ('pending','running') for v in state.values()):raise ValueError('unfinished queue still active')
    guards={}
    for name,row in state.items():
        if row['status']!='done':continue
        path=HERE/'raw'/name/'guard_result.json'
        if path.exists():
            guards[name]=load(path)
            if guards[name]['code']!=0:raise ValueError('completed step has invalid guard')
    narrow=[]
    for out in sorted((HERE/'raw').glob('R14_remaining_narrow_*')):
        modes=load(out/'modes/mode_check.json');hf=load(out/'hf.json');identity=load(out/'identity.json')
        if not modes['pass'] or not hf['pass'] or modes['steps']!=1024:raise ValueError('narrow correctness failed')
        narrow.append(dict(label=out.name.removeprefix('R14_remaining_narrow_'),steps=modes['steps'],
            c1=hf['pass'],near_tie_ratio=hf['gap_le_0_5_ratio'],max_gap=hf['max_gap'],
            c2=modes['pass'],mismatches=sum(modes['mismatches'].values()),artifact_id=identity['artifact_id'],
            qualification='nonpaged: disabling K-phase adds no K-phase coverage'))
    if len(narrow)!=4:raise ValueError('missing narrow-family result')
    tsv(HERE/'results/T11_supplemental.tsv',narrow)
    overhead=load(HERE/'raw/R14_remaining_overhead/result.json');timing=[];token_equal=[]
    for r in range(3):
        record=load(HERE/f'raw/R14_remaining_overhead_r{r}/llama_B1/round{r}.json')
        tokens={label:next(v['tokens'] for v in row['runs'] if not v['warmup'] and v['N']==1024) for label,row in record['arms'].items()}
        token_equal.append(all(v==tokens['baseline'] for v in tokens.values()))
        for label,row in record['arms'].items():
            if row['exit_code'] or row['execution_identity']['trace']!=(label!='baseline'):raise ValueError('invalid overhead identity')
            timing.append(dict(label=label,round=r,execution_id=row['execution_identity']['execution_id'],
                artifact_id=row['execution_identity']['artifact_id'],spill=row['execution_identity']['spill'],
                tpot_ms=(row['e2e_seconds']-row['ttft_seconds'])*1000/1023))
    tsv(HERE/'results/T2_supplemental_overhead.tsv',timing)
    baseline=overhead['arms']['baseline']['samples_ms'];middle=statistics.median(baseline)
    canaries=[i for i,v in enumerate(baseline) if abs(v/middle-1)>.02]
    profile=[]
    for p in sorted((HERE/'raw/R14_remaining_trace/task').glob('past*')):
        for kind,row in tasks(p/'task_profile.tsv').items():
            values=row['samples'];nonzero=[v for v in values if v['bytes']]
            profile.append(dict(past=int(p.name.removeprefix('past')),kind=kind,samples=len(values),
                operand_ready_observed=sum(v['operand_ready_observed'] for v in nonzero),nonzero_samples=len(nonzero),
                mean_run_ns=statistics.mean(v['run_ns'] for v in values),mean_bytes=statistics.mean(v['bytes'] for v in values),
                **row['fit'],qualification='Llama B1; 1/8 CTA diagnostic sampling; mixed geometries'))
    if any(r['operand_ready_observed']!=r['nonzero_samples'] for r in profile if r['kind']=='gemm'):raise ValueError('unobserved GEMM readiness')
    tsv(HERE/'results/T5_supplemental_tasks.tsv',profile)
    for p in sorted((HERE/'raw/R14_remaining_trace/stage').glob('past*')):
        write(HERE/f'results/T2_supplemental_stage_{p.name}.json',stage_ledger(p,1))
    controls=[]
    for cell,arm in load(HERE/'logic_completion_arms.json').items():
        identity=verify(arm['decode']);execution=bind_execution(identity,'L1',False)
        resource=identity['resources'][execution['kernel']]
        controls.append(dict(cell=cell,artifact_id=identity['artifact_id'],execution_id=execution['execution_id'],
            pg=identity['plan']['pg'],registers=resource.get('registers'),stack=resource.get('stack_bytes'),
            spill=execution['spill'],spill_stores=resource.get('spill_store_bytes'),spill_loads=resource.get('spill_load_bytes'),
            smem=identity['shared_memory_bytes'],role='audited frozen control; no new final binary selected'))
    tsv(HERE/'results/T3_closure_controls.tsv',controls)
    ceiling=load(HERE/'raw/D0_remaining/ceiling/dram_ceiling.json')
    if len(ceiling['processes'])<5 or any(p['contaminated'] for p in ceiling['processes']):raise ValueError('invalid bandwidth cohort')
    attempts=[]
    for line in (HERE/'scheduler_remaining/progress.tsv').read_text().splitlines():
        parts=line.split('\t')
        if len(parts)>=6 and parts[1]=='pending':attempts.append(dict(step=parts[0],attempt=parts[2],reason=parts[5]))
    predictions=load(HERE/'predictions.json')['items'];observed=[]
    phase_a=load(HERE/'results/phase_a_acceptance.json')
    phase_b=list(csv.DictReader((HERE/'results/T4_phase_b.tsv').open(),delimiter='\t'))
    for prediction in predictions:
        ident=prediction['id'];rows=[]
        if ident=='R13D_prime_vs_R13D':rows=[dict(cell=c,relative=v['rebuild_relative']) for c,v in phase_a['cells'].items()]
        elif ident=='N1_prime_vs_N1':
            m=phase_a['cells']['llama_B1']['metrics'];rows=[dict(cell='llama_B1',relative=m['N1_prime']['tpot_ms']/m['N1']['tpot_ms']-1,qualification='old N1 correctness-invalid and timing-unstable; no optimization attribution')]
        elif ident in ('SK_fill','GV','RW_pipe','EP_arg'):
            label={'SK_fill':'SK_fill','GV':'GV_','RW_pipe':'RW_pipe','EP_arg':'EP_arg'}[ident]
            rows=[dict(cell=r['cell'],label=r['label'],relative=float(r['relative'])) for r in phase_b if (r['label'].startswith(label) if ident=='GV' else r['label']==label)]
        elif ident in ('C-RW1','C-EP2','C-AT4'):
            with (HERE/'results/T8_phase_c.tsv').open() as f:
                rows=[dict(r) for r in csv.DictReader(f,delimiter='\t') if r.get('label')==ident.replace('-','_')]
        observed.append(dict(id=ident,prediction=prediction,observed=rows,status='measured' if rows else 'not_isolated_or_not_completed'))
    write(HERE/'results/prediction_review.json',observed)
    report=dict(status='closed_at_user_request; implementation/correctness evidence accepted, final performance unaccepted',
        prompt_sha256=sha('/root/Prompt/TileMega_R14_prompt.md'),queue=counts,accepted_guards=guards,narrow=narrow,
        trace_overhead=overhead,trace_tokens_equal=token_equal,canary_rounds=canaries,
        ceiling=dict(median_gbps=ceiling['calibration_median_gbps'],range_gbps=ceiling['process_range_gbps'],range_relative=ceiling['process_range_relative'],processes=len(ceiling['processes']),definition=ceiling['calibration_definition']),
        controls=controls,build_retry_events=attempts,completed_fresh_models=[],final_e2e='not_run',
        final_performance_gates='not_evaluated',planfamily='not_evaluated; no complete new D1',
        historical=dict(phase_a_steps=36,phase_b_steps=72,phase_b_variants=50,phase_b_artifacts=100,
            required_protocols=load(HERE/'results/phase_b_acceptance.json')['protocols'],phase_c_retention=load(HERE/'phase_c_retention.json')),
        limitation='shared correctness diagnostics are timing-ineligible; only final accepted guards enter overhead/ceiling; no synchronization conclusion beyond archived 50-process cases')
    write(HERE/'results/closure_acceptance.json',report)
    write(HERE/'results/T10_final_status.json',dict(status='not_evaluated',reason='user cancelled remaining D1/D2/D3 collection; no complete fresh selection or final four-cell paired matrix'))
    print(json.dumps(dict(queue=counts,narrow_cases=len(narrow),trace_overhead_percent={k:100*v['relative'] for k,v in overhead['arms'].items()},ceiling_gbps=report['ceiling']['median_gbps'],final_e2e='not_run')))

def archive():
    directories=list((HERE/'raw').glob('R14_remaining*'))+list((HERE/'raw').glob('D*_remaining'))+[HERE/'scheduler_remaining',HERE/'raw/closure_20261010']
    directories += [ROOT/f'runs/r14-remaining-{m}' for m in ('llama','qwen3')]
    paths=set()
    for directory in directories:
        if directory.exists():paths.update(p for p in directory.rglob('*') if p.is_file())
    excluded=[];records=[];destination=HERE/'raw/supplemental_closure.tar.xz'
    with tarfile.open(destination,'w:xz',preset=3) as tar:
        for p in sorted(paths):
            if p.name=='archive.log' or p.suffix=='.xz':continue
            if p.suffix in ('.so','.cu','.o','.cubin','.fatbin') or any(t in p.name for t in ('.edges.','.tasks.','.prices.')):
                excluded.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size));continue
            member=str(p.relative_to(ROOT));tar.add(p,arcname=member,recursive=False)
            records.append(dict(source=str(p),member=member,sha256=sha(p),bytes=p.stat().st_size))
    write(HERE/'raw/supplemental_closure_manifest.json',dict(archive=str(destination.relative_to(HERE)),sha256=sha(destination),bytes=destination.stat().st_size,members=records,excluded_artifacts=excluded))
    print(f'Archived {len(records)} evidence files: {destination.stat().st_size} bytes')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',action='store_true');a=p.parse_args();review()
    if a.archive:archive()
