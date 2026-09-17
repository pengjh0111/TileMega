#!/usr/bin/env python3
"""Compare production solver prices with every matching physical trace task.

No task-zero extrapolation, no timing filtering, and no change to frozen choices.
The old selected_prices task-zero audit remains reproducible as historical data.
"""
import argparse,csv,json,re,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];JOINT=HERE.parent/'JOINT2'
def rows(p):return list(csv.DictReader(p.open(),delimiter='\t'))
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prices',type=Path,default=HERE/'bounded_prices');ap.add_argument('--models',nargs='+',default=['gqa2','mha4','real']);ap.add_argument('--seqs',nargs='+',type=int,default=[4,128]);a=ap.parse_args();out=[]
 for name,path in json.loads((JOINT/'cells.json').read_text()).items():
  if name.split('_s')[0] not in a.models or int(name.split('_s')[1]) not in a.seqs:continue
  root=REPO/path;choice=json.loads((root/'choice.json').read_text())['arm'];dump=root/'trace'/choice/'dump'
  source=Path(json.loads((root/'specs.json').read_text())[choice]['source'])
  stages=re.search(r'constexpr StageDesc kStages\[\] = \{(.*?)\n\};',source.read_text(),re.S)
  if not stages:raise ValueError('missing generated stage identities '+str(source))
  kinds=re.findall(r'\{TaskKind::(\w+),',stages[1])
  slots={(int(r['stage']),int(r['logical_task'])):r for r in rows(dump/'slots.tsv')}
  groups={}
  for r in rows(a.prices/(name+'.tsv')):
   key=(int(r['runtime_stage']),int(r['logical_task']));actual=slots.pop(key)
   duration=int(actual['run_end'])-int(actual['run_begin']);price=float(r['price_ns'])
   if duration<0 or price<=0:raise ValueError('invalid duration/price '+str(key))
   groups.setdefault((r['runtime_stage'],r['logical_stage'],r['combine']),[]).append((duration,price,duration/price))
  if slots:raise ValueError('unpriced physical tasks '+name)
  for (runtime,logical,combine),v in groups.items():
   out.append(dict(cell=name,runtime_stage=runtime,logical_stage=logical,task_kind=kinds[int(logical)],combine=combine,tasks=len(v),
    measured_p50_ns=statistics.median(x[0] for x in v),predicted_p50_ns=statistics.median(x[1] for x in v),
    measured_predicted_p50=statistics.median(x[2] for x in v),measured_total_ns=sum(x[0] for x in v),predicted_total_ns=sum(x[1] for x in v),dump=str(dump)))
 target=a.prices/('comparison_'+'_'.join(a.models)+('' if a.seqs==[4,128] else '_s'+'_'.join(map(str,a.seqs)))+'.tsv')
 with target.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]),delimiter='\t');w.writeheader();w.writerows(out)
 for name in dict.fromkeys(r['cell'] for r in out):
  for combine in ('0','1'):
   v=[r['measured_predicted_p50'] for r in out if r['cell']==name and r['combine']==combine]
   if v:print(name,'combine='+combine,'stage_ratio_p50='+str(statistics.median(v)),'stages='+str(len(v)))
if __name__=='__main__':main()
