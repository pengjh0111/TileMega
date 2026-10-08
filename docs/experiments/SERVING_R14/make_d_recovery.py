#!/usr/bin/env python3
"""Preserve failed D1 records and publish a fresh dependency chain after repair."""
import argparse,json
from pathlib import Path
from make_phase0 import HERE,write

def definitions():
    original=json.loads((HERE/'queue/queue_phase_d1.json').read_text())
    renamed={s['name']:s['name']+'_v2' for s in original if s['name'] not in ('D0','C_arch_v2')}
    rows=[]
    for step in original:
        if step['name'] not in renamed:continue
        step['name']=renamed[step['name']]
        step['after']=[renamed.get(dep,dep) for dep in step['after']]
        command=step['command']
        if any(str(x).endswith('/phase_d_r14.py') for x in command):command+=['--tag','_v2']
        else:
            step['command']=[str(x).replace('phase_d_baseline_builds.json','phase_d_baseline_builds_v2.json').replace('/raw/D_baseline_build/','/raw/D_baseline_build_v2/') for x in command]
        if step['name'].startswith('D1_') and 'smoke' not in step['name'] and 'family' not in step['name']:
            step['after'].append('D0')  # Reuse valid device calibration stamps.
        rows.append(step)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    jobs=json.loads((HERE/'phase_d_baseline_builds.json').read_text())
    for job in jobs:job['out']=job['out'].replace('/raw/D_baseline_build/','/raw/D_baseline_build_v2/')
    write(HERE/'phase_d_baseline_builds_v2.json',jobs)
    arms=json.loads((HERE/'phase_d_baseline_arms.json').read_text())
    for arm in arms.values():arm['decode']=arm['decode'].replace('/raw/D_baseline_build/','/raw/D_baseline_build_v2/')
    write(HERE/'phase_d_baseline_arms_v2.json',arms)
    rows=definitions()
    for step in rows:step['after']=list(dict.fromkeys(step['after']))
    write(a.out,rows)
    print(f'{len(rows)} fresh steps; old failed records preserved; D0 calibration reused')

if __name__=='__main__':main()
