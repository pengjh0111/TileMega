#!/usr/bin/env python3
"""Supplemental identical-kernel control diagnostic, not replacement B1 data.

Four labels (two per original executable), 25 rotated fresh-process rounds.
The original 7500 samples and their statistics remain unchanged. No compile,
plan, protocol, timing, fixture, tolerance or sample-filter change is made.
"""
import json,sys,time,csv
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'JOINT2'))
from process import run
from analyze import timing,interval

def main():
 root=HERE/'bounded_w1_repeat';(root/'bin').mkdir(parents=True,exist_ok=True)
 original=HERE/'bounded_raw/mha4_s128';names=['c2_a','c3_a','c2_b','c3_b'];session=str(time.time_ns())
 for name in names:
  link=root/'bin'/name;source=original/'bin'/(name[:2]+'__full')
  if link.exists():raise RuntimeError('refusing repeated diagnostic '+str(link))
  link.symlink_to(source)
 (root/'specs.json').write_text(json.dumps(dict(original=str(original),rounds=25,arms=names,session=session,purpose=__doc__),indent=2)+'\n')
 for i in range(25):
  for k in range(4):
   arm=names[(i+k)%4];run(root,'mha4',128,arm,root/'measure'/arm,i,k,session)
  print('W1_CONTROL_ROUND',i+1,flush=True)
 rows=[]
 for numerator,denominator in [('c3_a','c2_a'),('c2_b','c2_a'),('c3_b','c3_a')]:
  ratios=[timing(root/'measure'/numerator/f'r{i}.log')['l2_ms']/timing(root/'measure'/denominator/f'r{i}.log')['l2_ms'] for i in range(25)]
  med,lo,hi=interval(ratios);row=dict(pair=numerator+'/'+denominator,ratio=med,lo=lo,hi=hi);rows.append(row);print(row,flush=True)
 with (root/'diagnostic.tsv').open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');writer.writeheader();writer.writerows(rows)
if __name__=='__main__':main()
