#!/usr/bin/env python3
"""Pool raw full-arm placement pairs using R1's statistic and bootstrap seed.

Reference pooling is over 100 within-cell ratios, not the four cell medians.
Real-width pooling is reported separately over 50 ratios. No pairs are dropped.
This is descriptive pooling across cells, not a claim of one shared session.
"""
import argparse,csv,json,random,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'JOINT2'));from analyze import timing

def calculate(raw):
 pairs=[];results=[]
 for group,names in [('reference',[f'{m}_s{s}' for m in ('gqa2','mha4') for s in (4,128)]),('real',['real_s4','real_s128'])]:
  values=[]
  for name in names:
   for i in range(25):
    paths=[raw/name/'measure'/arm/f'r{i}.log' for arm in ('rotate__full','legacy_grid_stride__full')]
    metadata=[json.loads(p.with_suffix('.json').read_text()) for p in paths]
    if metadata[0]['session']!=metadata[1]['session'] or any(m['exit_code']!=0 or m['round']!=i for m in metadata):raise ValueError('invalid paired process '+name)
    value=timing(paths[0])['l2_ms']/timing(paths[1])['l2_ms'];values.append(value)
    pairs.append(dict(group=group,cell=name,round=i,ratio=value,rotate_log=str(paths[0]),legacy_log=str(paths[1])))
  rng=random.Random(20260906);n=len(values);draws=20000
  samples=sorted(statistics.median(values[rng.randrange(n)] for _ in range(n)) for _ in range(draws))
  results.append(dict(group=group,cells=len(names),pairs=n,median_ratio=statistics.median(values),ci_lo=samples[int(.025*draws)],ci_hi=samples[int(.975*draws)-1],bootstrap_draws=draws,seed=20260906))
 return results,pairs

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--raw',type=Path,default=HERE/'bounded_raw');ap.add_argument('--out',type=Path,default=HERE/'bounded_analysis');a=ap.parse_args();results,pairs=calculate(a.raw);a.out.mkdir(parents=True,exist_ok=True)
 for filename,rows in [('placement_pool.tsv',results),('placement_pairs.tsv',pairs)]:
  with (a.out/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
 for r in results:print('PLACEMENT_POOL',json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
