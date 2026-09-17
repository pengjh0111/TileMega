#!/usr/bin/env python3
"""Compare task-zero prices with the same physical task in fresh trace dumps.

All-task medians are a separate diagnostic; boundary tiles need not have the
same cost as task zero. This does not change any R6 selection or gate.
"""
import csv,json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;JOINT=HERE.parent/'JOINT2';REPO=HERE.parents[2]
def main():
 rows=[]
 for name,path in json.loads((JOINT/'cells.json').read_text()).items():
  p=REPO/path;arm=json.loads((p/'choice.json').read_text())['arm'];dump=p/'trace'/arm/'dump'
  slots=list(csv.DictReader((dump/'slots.tsv').open(),delimiter='\t'))
  for r in csv.DictReader((HERE/'selected_prices'/f'{name}.tsv').open(),delimiter='\t'):
   stage=[s for s in slots if s['stage']==r['runtime_stage']];zero=next(s for s in stage if s['logical_task']=='0');durations=[int(s['run_end'])-int(s['run_begin']) for s in stage]
   duration=int(zero['run_end'])-int(zero['run_begin']);pred=float(r['price_task0_ns'])
   rows.append(dict(cell=name,**r,measured_task0_ns=duration,task0_measured_predicted=duration/pred if pred else '',all_task_p50_ns=statistics.median(durations),all_task_max_ns=max(durations),dump=str(dump)))
 with (HERE/'selected_prices/comparison.tsv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
 for name in dict.fromkeys(r['cell'] for r in rows):
  for combine in ('0','1'):
   selected=[r for r in rows if r['cell']==name and r['combine']==combine]
   if selected:print(name,'combine='+combine,'task0_ratio_p50='+str(statistics.median(r['task0_measured_predicted'] for r in selected)),'stages='+str(len(selected)),flush=True)
if __name__=='__main__':main()
