#!/usr/bin/env python3
"""Describe hardware context without filtering or adjusting any timing sample.

--raw points to one completed cell (50 arms x 25 processes). --out is a fresh
TSV path. Utilization-positive samples identify observed GPU activity, not the
exact CUDA event interval; the external sampler also sees correctness kernels.
Spearman correlations describe association and do not establish causation.
"""
import argparse,csv,json,math,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'JOINT2'))
from analyze import timing

def ranks(values):
 order=sorted(range(len(values)),key=values.__getitem__);out=[0.]*len(values);i=0
 while i<len(order):
  j=i+1
  while j<len(order) and values[order[j]]==values[order[i]]:j+=1
  for k in order[i:j]:out[k]=(i+j-1)/2
  i=j
 return out

def correlation(a,b):
 if len(a)<2:return float('nan')
 a=ranks(a);b=ranks(b);ma=statistics.mean(a);mb=statistics.mean(b)
 aa=sum((x-ma)**2 for x in a);bb=sum((y-mb)**2 for y in b)
 return sum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(aa*bb) if aa and bb else float('nan')

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 specs=json.loads((a.raw/'specs.json').read_text());results=[]
 for arm in specs:
  samples=[]
  for i in range(25):
   p=a.raw/'measure'/arm/f'r{i}.log';t=timing(p);meta=json.loads(p.with_suffix('.json').read_text())
   gpu=[]
   with p.with_suffix('.gpu.csv').open() as f:
    for r in csv.reader(f):
     if len(r)!=8:raise ValueError('malformed hardware row '+str(p))
     try:gpu.append(tuple(float(r[j].strip()) for j in (2,3,4,5,6,7)))
     except ValueError:continue  # e.g. driver reports N/A; absence is counted.
   active=[r for r in gpu if r[4]>0]
   samples.append(dict(l1=t['l1_ms'],l2=t['l2_ms'],elapsed=meta['elapsed_ns']/1e9,active=active,
                       clock=statistics.median(r[0] for r in active) if active else float('nan'),
                       power=statistics.median(r[3] for r in active) if active else float('nan')))
  complete=all(math.isfinite(r['clock']) for r in samples)
  results.append(dict(cell=a.raw.name,arm=arm,processes=len(samples),processes_with_active_telemetry=sum(bool(r['active']) for r in samples),
    l1_min_ms=min(r['l1'] for r in samples),l1_max_ms=max(r['l1'] for r in samples),l1_median_ms=statistics.median(r['l1'] for r in samples),
    l2_median_ms=statistics.median(r['l2'] for r in samples),process_seconds_median=statistics.median(r['elapsed'] for r in samples),
    active_clock_mhz_median=statistics.median(r['clock'] for r in samples) if complete else float('nan'),
    active_power_w_median=statistics.median(r['power'] for r in samples) if complete else float('nan'),
    l1_clock_spearman=correlation([r['l1'] for r in samples],[r['clock'] for r in samples]) if complete else float('nan'),
    l2_clock_spearman=correlation([r['l2'] for r in samples],[r['clock'] for r in samples]) if complete else float('nan')))
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]),delimiter='\t');w.writeheader();w.writerows(results)
 print('CONTEXT_AUDIT',a.raw.name,'processes',25*len(specs),'timing_filter=none','output',a.out)
if __name__=='__main__':main()
