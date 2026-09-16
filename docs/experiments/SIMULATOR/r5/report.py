#!/usr/bin/env python3
"""Budget and historical calibration ordering; input measurements stay labelled."""
import csv,json,statistics
from pathlib import Path
def spearman(x,y):
    def ranks(v):
        ordered=sorted(v)
        return [(ordered.index(a)+len(ordered)-ordered[::-1].index(a)+1)/2 for a in v]
    a,b=ranks(x),ranks(y);am,bm=statistics.mean(a),statistics.mean(b)
    return sum((u-am)*(v-bm) for u,v in zip(a,b))/(sum((u-am)**2 for u in a)*sum((v-bm)**2 for v in b))**.5
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
def read(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def main():
    rows=read(HERE/'evaluations.tsv');measure=read(REPO/'docs/experiments/SIMULATOR/raw/time/l2.tsv')
    modes={'legacy_grid_stride':'0','balanced':'4','rotate':'5'}
    selected=[r for r in rows if r['model']!='real' and r['candidate'] in modes]
    actual=[statistics.median(float(x['l2_ms'])*1e6 for x in measure if (x['model'],x['seq'],x['place'])==(r['model'],r['seq'],modes[r['candidate']])) for r in selected]
    output=dict(measured_source='R2 raw/time/l2.tsv: historical calibration set, not R5 GPU performance',points=len(selected))
    for column in ('coarse_ns','full_ns'):
        predicted=[float(r[column]) for r in selected]
        rank=sorted(range(len(selected)),key=lambda i:predicted[i]);best=min(range(len(selected)),key=lambda i:actual[i])
        output[column]=dict(spearman=spearman(predicted,actual),actual_top1_predicted_rank=rank.index(best)+1)
    for model in ('reference','real'):
        group=[r for r in rows if (r['model']=='real')==(model=='real')]
        output[model]={k:max(float(r[k]) for r in group) for k in ('prepare_us','coarse_us','full_us')}
    print(json.dumps(output,indent=2));(HERE/'report.json').write_text(json.dumps(output,indent=2)+'\n')
    (HERE/'README.md').write_text('''# R5 hierarchical Plan evaluation

`PreparedPlanBounds` walks the task DAG once per configuration. Per-Plan
`EvaluatePlanBounds` validates task coverage and computes queue/work bounds
in O(tasks); same-worker dependency node weights remain in the prepared CP.
`RankPlans` simulates top-k (k=3), including exact cutoff ties rather than
arbitrarily discarding equal-bound placements. `ranks.tsv` gives k=1..5.
The EFT materialization requires an explicit solved table and is covered by
JOINT, not invented by this historical-template driver.

The complete event simulator remains over its original per-Plan budget.
Only coarse evaluation fits. Therefore S1c-a is FAIL, not a renamed coarse
budget PASS. R5 §9.3 permits the reduced, fully measured candidate pool for
EX-S3. Exact preparation, coarse, full and batch times are all retained.
The graph preparation time is never presented as free.

`evaluations.tsv` and `ranks.tsv` are fresh CPU executions; measured ranking
uses R2's explicitly requested 18-point calibration set. It is historical
validation, not a fresh GPU speedup. New candidate ordering is measured in
JOINT. Default solver/runtime behavior is unchanged until explicitly opting
into the new ranking API. Existing full-simulator tests and the new bounds
contract test pass. Raw test commands/output: `bounds_test.json`.
''')
if __name__=='__main__':main()
