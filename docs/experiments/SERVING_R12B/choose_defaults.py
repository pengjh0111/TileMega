#!/usr/bin/env python3
"""Freeze the R12b build defaults from the preregistered S-1b medians."""
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--s1b',type=Path,required=True)
p.add_argument('--out',type=Path,default=Path('docs/experiments/SERVING_R12B/defaults.json'))
a=p.parse_args()
try:data=json.loads(a.s1b.read_text())
except (OSError,ValueError):data={}
base=data.get('baseline_ms');reasons=[];mask=31;lookahead=-1;poll=0
if not base or not data.get('complete'):
    reasons.append('S-1b missing or failed; using fixed fallback')
else:
    for cls,median in data.get('class_off_ms',{}).items():
        if float(median)<0.99*float(base):mask &= ~(1<<int(cls))
    d=data.get('lookahead_ms',{})
    if all(k in d for k in ('0','131072')) and abs(d['0']-d['131072'])>0.01*min(d.values()):
        lookahead=min(d,key=d.get);lookahead=int(lookahead)
    poll_data=data.get('poll_ms',{})
    if all(k in poll_data for k in ('0','200')) and abs(poll_data['0']-poll_data['200'])>0.01*min(poll_data.values()):
        poll=int(min(poll_data,key=poll_data.get))
result=dict(kphase_mask=mask,lookahead_bytes=lookahead,v3_poll_ns=poll,reasons=reasons,
            rule='class off >1%; D/poll smaller median, <=1% tie to search/zero')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
for model in ('llama','qwen3'):
    source=Path(f'configs/e2e/{model}_r12.json')
    config=json.loads(source.read_text());config['solver']['mode']='L2'
    config['output']['dir']=f'runs/r12b-{model}'
    config['features'].update(kphase_mask=mask,lookahead_bytes=lookahead,v3_poll_ns=poll)
    Path(f'configs/e2e/{model}_r12b.json').write_text(json.dumps(config,indent=2)+'\n')
print(json.dumps(result))
