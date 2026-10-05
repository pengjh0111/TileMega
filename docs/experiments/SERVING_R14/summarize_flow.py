#!/usr/bin/env python3
"""Summarize signed audit residuals without printing the per-task search log."""
import gzip,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=[]
for when in ('before','after'):
    result=HERE/'raw/FX24'/when/'results.json'
    if not result.exists():continue
    for case in json.loads(result.read_text()):
        path=result.parent/case['name']/'audit.log.gz'
        negative=[];partitions=0;conservation=0;minimum=0;max_abs_error=0
        with gzip.open(path,'rt') as stream:
            for line in stream:
                fields=dict(re.findall(r'(\w+)=([^ ]+)',line.strip()))
                if line.startswith('FLOW_AUDIT '):
                    partitions+=1;r=float(fields['remaining']);minimum=min(minimum,r)
                    if r<0 and len(negative)<3:negative.append(fields)
                if line.startswith('FLOW_CONSERVATION '):
                    conservation+=1
                    max_abs_error=max(max_abs_error,abs(float(fields['delivered'])-float(fields['expected'])))
        rows.append(dict(when=when,case=case['name'],exit_code=case['exit_code'],partitions=partitions,
            negative_examples=negative,min_remaining=minimum,conservation_checks=conservation,max_step_error_bytes=max_abs_error,
            evidence=str(path.relative_to(HERE))))
(HERE/'results').mkdir(exist_ok=True)
(HERE/'results/T12_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ('when','case','exit_code','min_remaining','conservation_checks')} for r in rows]))
