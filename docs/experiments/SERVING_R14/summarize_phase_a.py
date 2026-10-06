#!/usr/bin/env python3
"""Export complete anchor and sampled ledger tables from accepted raw artifacts."""
import csv,json,statistics
from pathlib import Path
from ledger_r14 import execution
HERE=Path(__file__).resolve().parent

def tsv(name,rows):
    if not rows:raise ValueError('empty '+name)
    with (HERE/'results'/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
def main():
    accepted=json.loads((HERE/'results/phase_a_acceptance.json').read_text());anchors=[];stages=[];tasks=[]
    for cell,summary in accepted['cells'].items():
        for label in summary['metrics']:
            runs=[json.loads((HERE/f'raw/A1_{cell}_r{r}/{cell}/round{r}.json').read_text())['arms'][label] for r in range(3)]
            if runs[0].get('execution_identity'):
                ids={execution(x)['execution_id'] for x in runs}
                if len(ids)!=1:raise ValueError('mixed execution')
            for run in runs:
                if label!='vllm':
                    timed=[x for x in run['runs'] if not x['warmup'] and x['N']==1024]
                    b=int(cell.split('_B')[1]);tokens=timed[0]['tokens']
                    if len(tokens)!=b or any(len(t)!=1024 for t in tokens):raise ValueError('incomplete tokens')
            r=dict(cell=cell,arm=label,**summary['metrics'][label])
            for key in ('tpot_mean_seconds','tpot_p50_seconds','tpot_p90_seconds','output_tokens_per_second'):
                values=[x.get(key) for x in runs]
                r[key]=statistics.median(values) if all(v is not None for v in values) else 'unavailable'
            r['execution_id']=runs[0].get('execution_identity',{}).get('execution_id') if runs[0].get('execution_identity') else 'pinned legacy/reference'
            anchors.append(r)
    for folder in sorted(HERE.glob('raw/A2_trace_*/*/past*')):
        cell=folder.parent.parent.name.removeprefix('A2_trace_');arm=folder.parent.name;past=int(folder.name[4:])
        if (folder/'ledger.json').exists():
            rows=json.loads((folder/'ledger.json').read_text())['stages'];groups={}
            for r in rows:groups.setdefault(r['stage'],[]).append(r)
            for stage,group in groups.items():
                row=dict(cell=cell,arm=arm,past=past,stage=stage,kind=group[0]['semantic_kind'],layer=group[0]['layer'])
                for k in ('duration_ns','bytes','gbps','tail_ns','floor_ns_884_5','excess_ns_884_5','floor_ns_981_6','excess_ns_981_6','floor_ns_r13_measured','excess_ns_r13_measured'):
                    row[k]=statistics.median(r[k] for r in group)
                row['qualification']='sampled extrema, diagnostic only';stages.append(row)
        if (folder/'task_summary.json').exists():
            for kind,group in json.loads((folder/'task_summary.json').read_text()).items():
                rows=group['samples'];row=dict(cell=cell,arm=arm,past=past,kind=kind,samples=len(rows))
                for k in ('run_ns','bytes','query_ns','first_page_wait_ns','later_page_wait_ns','wave_compute_ns','la_ns','epilogue_ns'):
                    row[k]=statistics.mean(r.get(k,0) for r in rows)
                row.update(group['fit']);row['qualification']='mixed geometries; zero readiness on nonpaged GEMM means unavailable';tasks.append(row)
    tsv('T1_anchor.tsv',anchors);tsv('T2_stage_ledger.tsv',stages);tsv('T5_phase_a_tasks.tsv',tasks)
    print('Phase-A complete anchors and qualified sampled ledgers exported')
if __name__=='__main__':main()
