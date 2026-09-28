#!/usr/bin/env python3
"""Compare shortlisted Level-1 predictions with the measured L2 candidates."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,action='append',required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=[]
for run in a.run_dir:
    plans=run/'plans.json'
    if not plans.exists():continue
    data=json.loads(plans.read_text())
    for batch,pair in data.items():
        if not batch.isdigit():continue
        for phase,so in pair.items():
            stem=Path(so)
            predicted=Path(str(stem)+'.top3.tsv');measured=Path(str(stem)+'.top3_measured.tsv')
            if not predicted.exists() or not measured.exists():continue
            pred={int(r['rank']):float(r['predicted_ns'])/1e6 for r in csv.DictReader(predicted.open(),delimiter='\t')}
            obs={int(r['rank']):float(r['mean_ms']) for r in csv.DictReader(measured.open(),delimiter='\t') if r['mode']=='L2'}
            for rank in sorted(pred.keys() & obs.keys()):
                rows.append(dict(run=str(run),batch=int(batch),phase=phase,rank=rank,
                                 predicted_ms=pred[rank],measured_ms=obs[rank],
                                 measured_over_predicted=obs[rank]/pred[rank]))
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(rows=len(rows),out=str(a.out))))
