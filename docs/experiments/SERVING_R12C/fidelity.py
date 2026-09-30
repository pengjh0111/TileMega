#!/usr/bin/env python3
"""All shortlisted candidates, measured ordering, and optional E2E anchors."""
import argparse,csv,json,re
from pathlib import Path

def read(path):return list(csv.DictReader(path.open(),delimiter='\t')) if path.exists() else []
def fidelity(so):
    so=Path(so);pred=read(Path(str(so)+'.top3.tsv'));obs=read(Path(str(so)+'.top3_measured.tsv'));times={int(r['rank']):float(r['mean_ms']) for r in obs if r.get('mode')=='L2'}
    valid=[r for r in pred if int(r['rank']) in times];concordant=discordant=0
    for i,a in enumerate(valid):
        for b in valid[i+1:]:
            product=(float(a['predicted_ns'])-float(b['predicted_ns']))*(times[int(a['rank'])]-times[int(b['rank'])])
            concordant+=product>0;discordant+=product<0
    pairs=len(valid)*(len(valid)-1)//2;tau=(concordant-discordant)/pairs if pairs else None
    selected=min(times,key=times.get) if times else None;rows=[]
    for r in pred:
        rank=int(r['rank']);key=r['key'];path=Path(str(so)+f'.top{rank}.candidate.so.plan.json');m=json.loads(path.read_text()) if path.exists() else {}
        pred_ms=float(r['predicted_ns'])/1e6;observed=times.get(rank)
        rows.append(dict(plan=str(so),rank=rank,origin=r.get('origin','unrecorded'),key=key,gemms=m.get('gemms',[]),pages=m.get('pages'),predicted_ms=pred_ms,measured_ms=observed,measured_over_predicted=observed/pred_ms if observed is not None and pred_ms else None,kendall_tau=tau,selected=rank==selected))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan-dir',type=Path,action='append',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--anchor',type=Path);a=p.parse_args();rows=[]
    for directory in a.plan_dir:
        for path in directory.glob('*.top3.tsv'):rows+=fidelity(Path(str(path)[:-9]))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(candidates=rows,anchor=json.loads(a.anchor.read_text()) if a.anchor else None),indent=2)+'\n');print('fidelity candidates='+str(len(rows)))
if __name__=='__main__':main()
