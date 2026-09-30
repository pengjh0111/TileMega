#!/usr/bin/env python3
"""Preregistered B1c rules. D and watchdog are the only selectable defaults."""
import argparse,json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def choose(samples):
    med={b:{arm:statistics.median(v) for arm,v in arms.items() if len(v)==3} for b,arms in samples.items()}
    if any('P-base' not in med.get(b,{}) for b in ('1','16')):raise ValueError('B1 requires three baseline rounds for both batches')
    eligible=[]
    for d in (65536,131072):
        name='P-D'+str(d//1024)+'K';improvements=[]
        for b in ('1','16'):
            if name not in med[b]:break
            improvements.append(1-med[b][name]/med[b]['P-base'])
        if len(improvements)==2 and min(improvements)>.01:eligible.append((statistics.mean(improvements),d))
    depth=max(eligible)[1] if eligible else 0
    watchdog=0 if any('P-noWD' in med[b] and 1-med[b]['P-noWD']/med[b]['P-base']>.01 for b in ('1','16')) else 1
    return dict(lookahead_bytes=depth,watchdog=watchdog,kphase_mask=31,v3_poll_ns=0,medians_s=med)
def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,default=HERE/'raw');p.add_argument('--out',type=Path,default=HERE/'defaults.json');a=p.parse_args();samples={}
    for path in a.raw.glob('B1_*/llama_B*/round*.json'):
        data=json.loads(path.read_text())
        if data.get('invalidated'):continue
        batch=data['cell'].split('_B')[-1]
        for arm,r in data['arms'].items():
            if r.get('exit_code')==0:samples.setdefault(batch,{}).setdefault(arm,[]).append((r['e2e_seconds']-r['ttft_seconds'])/1023)
    result=choose(samples);result['rule']='D improves both batches >1%, choose largest mean improvement; noWD improves either >1%; mask=31, poll=0'
    for model in ('llama','qwen3'):
        source=ROOT/f'configs/e2e/{model}_r12b.json';cfg=json.loads(source.read_text());cfg['features'].update({k:result[k] for k in ('lookahead_bytes','watchdog','kphase_mask','v3_poll_ns')});cfg['features']['pg']='pages';cfg['solver'].update(mode='L2',candidate_guard_wait_s=1800,jobs=3);cfg['output']['dir']=f'runs/r12c-{model}'
        (ROOT/f'configs/e2e/{model}_r12c.json').write_text(json.dumps(cfg,indent=2)+'\n')
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
