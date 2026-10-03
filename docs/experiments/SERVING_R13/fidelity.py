#!/usr/bin/env python3
"""Both selection levels and candidate rank fidelity, without changing prices."""
import argparse,csv,json,statistics
from pathlib import Path
from ledger import read,write

def tau(predicted,measured):
    common=sorted(set(predicted)&set(measured));terms=[]
    for i,a in enumerate(common):
        for b in common[i+1:]:
            x=predicted[a]-predicted[b];y=measured[a]-measured[b]
            if x and y:terms.append(1 if x*y>0 else -1)
    return sum(terms)/len(terms) if terms else None

def candidates(stem):
    top=Path(str(stem)+'.top3.tsv');obs=Path(str(stem)+'.top3_measured.tsv')
    if not top.exists() or not obs.exists():return []
    predicted={int(r['rank']):r for r in read(top)};measured={int(r['rank']):r for r in read(obs) if r.get('mean_ms') and r.get('status','')!='rejected'}
    score={k:float(r['predicted_ns']) for k,r in predicted.items()};times={k:float(r['mean_ms'])*1e6 for k,r in measured.items()}
    winner=min(times,key=times.get) if times else None;correlation=tau(score,times);rows=[]
    for rank,row in predicted.items():
        so=Path(str(stem)+f'.top{rank}.so');manifest=Path(str(so)+'.plan.json');geometry=json.loads(manifest.read_text()) if manifest.exists() else {}
        rows.append(dict(plan=str(stem),rank=rank,origin=row.get('origin','unrecorded'),key=row['key'],
            predicted_ns=score[rank],measured_ns=times.get(rank),ratio=times[rank]/score[rank] if rank in times else None,
            selected=rank==winner,kendall_tau=correlation,mode=measured.get(rank,{}).get('mode'),loop=measured.get(rank,{}).get('loop'),
            gemms=json.dumps(geometry.get('gemms',[])),pages=json.dumps(geometry.get('pages'))))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,action='append',default=[]);p.add_argument('--run-dir',type=Path,action='append',default=[]);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
    stems=set(a.plan)
    for run in a.run_dir:
        path=run/'plans.json'
        if not path.exists():continue
        for b,pair in json.loads(path.read_text()).items():
            if not b.isdigit():continue
            for phase in ('prefill','decode'):
                if phase in pair:stems.add(Path(pair[phase]))
            for c in pair.get('decode_pg_choice',{}).get('candidates',[]):
                stems.add(Path(c['library']));rows.append(dict(run=str(run),batch=b,stage='joint',pg=c['pg'],mode=c['mode'],loop=c['loop'],samples_ms=json.dumps(c['samples_ms']),median_ms=statistics.median(c['samples_ms']) if c['samples_ms'] else None,error=c.get('error')))
    for stem in sorted(stems):rows+=candidates(stem)
    write(a.out,rows);print('fidelity records '+str(len(rows)))
if __name__=='__main__':main()
