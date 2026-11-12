"""Continuously consolidate accepted evidence; never promote incomplete gates."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('mx8_bandwidth',ROOT/'experiments/mx8/bandwidth.py')
bw=importlib.util.module_from_spec(spec);spec.loader.exec_module(bw)


def consolidate(queue,out):
    routes=ROOT/'runs/mx8/routes';routes.mkdir(parents=True,exist_ok=True)
    manifest=queue.parent/'ABLATION_QUEUE.json'
    formal_checks={name for arm in json.loads(manifest.read_text())['arms'] for name in arm['qualifications']} if manifest.exists() else set()
    groups={};failed=[]
    for path in sorted(queue.glob('jobs/*/ACCEPTED.json')):
        accepted=json.loads(path.read_text());definition=json.loads(Path(accepted['definition']['path']).read_text())
        if definition['kind'] not in ('routing','ablation_qualify','ablation_timing'):continue
        # Smoke plans are separate artifacts, never members of the selected
        # plan's five-process qualification cohort.
        if definition['kind']=='ablation_qualify' and path.parent.name not in formal_checks:continue
        result_path=Path(accepted['folder'])/'result/RESULT.json'
        result=json.loads(result_path.read_text())
        if definition['kind']=='routing':
            target=routes/f'{result["model"]}_B{result["batch"]}.json'
            target.write_text(json.dumps(dict(**result,raw_result=str(result_path),raw_result_sha256=bw.sha(result_path)),indent=2)+'\n')
            continue
        cell_index=definition['command'].index('--cell')+1;cell_path=Path(definition['command'][cell_index])
        cell=json.loads(cell_path.read_text());key=(cell['model'],cell['batch'],result['variant'])
        groups.setdefault(key,dict(cell=cell,checks=[],timing=[]))['checks' if definition['kind']=='ablation_qualify' else 'timing'].append((result,str(result_path)))
    for path in sorted(queue.glob('jobs/*/FAILED.json')):
        failed.append(str(path))
    rows=[]
    for (model,batch,variant),group in sorted(groups.items()):
        checks=[r for r,_ in group['checks']];timing=[r for r,_ in group['timing']]
        hashes={r['tokens_sha256'] for r in checks}
        other=groups.get((model,batch,'hostlaunch' if variant=='graph' else 'graph'))
        other_hashes={r['tokens_sha256'] for r,_ in other['checks']} if other else set()
        exact=bool(checks) and all(r['single_loop_exact_1024'] and r['frozen_reference_exact'] for r in checks) and len(hashes)==1
        hf=[r['hf']['pass'] for r in checks if 'hf' in r]
        qualified=len(checks)==5 and exact and hf==[True]
        cross_arm=bool(other_hashes) and hashes==other_hashes and len(hashes)==1
        complete=qualified and cross_arm and len(timing)==3
        tpot=statistics.median(r['tpot_ms'] for r in timing) if complete else ''
        event_values=[r['event_tpot_ms'] for r in timing]
        rows.append(dict(evidence='verified',model=model,tp=1,batch=batch,variant='AB4c_'+variant,
            baseline='frozen main experiment; no AB4b retest',main_tpot_ms=group['cell']['tpot_ms'],
            qualification_processes=len(checks),qualification_passes=sum(r['single_loop_exact_1024'] and r['frozen_reference_exact'] for r in checks),
            single_loop_and_fresh_exact_1024=exact,hf_256_pass=hf==[True],cross_arm_exact_1024=cross_arm,
            main_reference_tokens=min((r['reference_tokens'] for r in checks),default=0),
            timing_processes=len(timing),warmup_per_count=1,timed_per_count=3,
            tpot_ms=tpot,
            slowdown_vs_main_event_tpot=(statistics.median(r['event_tpot_ms'] for r in timing)/group['cell']['tpot_ms'] if complete else ''),
            ttft_ms=statistics.median(r['ttft_ms'] for r in timing) if complete else '',
            e2e_ms=statistics.median(r['e2e_ms'] for r in timing) if complete else '',
            output_tokens_per_second=statistics.median(r['output_tokens_per_second'] for r in timing) if complete else '',
            event_tpot_ms=statistics.median(r['event_tpot_ms'] for r in timing) if complete else '',
            event_tpot_processes_ms=json.dumps(event_values),
            event_tpot_process_min_ms=min(event_values) if event_values else '',
            event_tpot_process_max_ms=max(event_values) if event_values else '',
            event_tpot_process_spread_pct=(100*(max(event_values)-min(event_values))/statistics.median(event_values) if len(event_values)>1 else ''),
            timing_formula='(median(E2E_N1024)-median(E2E_N1))/1023',
            comparison_limit='historical main reference; not a contemporaneous P-PAIR',
            status='complete' if complete else 'pending',
            raw_results=json.dumps([p for _,p in group['checks']+group['timing']])))
    bw.write_csv(out/'MX8_ab4c_c550.csv',rows)
    bw.collect(out,queue,routes)
    status=dict(arms_observed=len(rows),arms_complete=sum(r['status']=='complete' for r in rows),
        arms_expected=18,routing_captures=len(list(routes.glob('*.json'))),routing_expected=8,failed_jobs=failed)
    (out/'COLLECTION.json').write_text(json.dumps(status,indent=2)+'\n')
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--poll-s',type=float,default=30)
    parser.add_argument('--out',type=Path,default=ROOT/'runs/mx8/tables')
    parser.add_argument('--queue',type=Path,default=ROOT/'runs/mx8/queue')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    while True:
        print(json.dumps(consolidate(args.queue,args.out)),flush=True)
        if not args.watch:break
        time.sleep(args.poll_s)
