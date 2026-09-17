#!/usr/bin/env python3
"""R6 comparisons from fresh rotated raw processes and corrected task graphs."""
import argparse,csv,importlib.util,json,math,random,re,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('r6_trace',HERE.parent/'TRACE_V2/analyze.py');trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)
def table(p):return list(csv.DictReader(p.open(),delimiter='\t'))
def timing(p):
 lines=re.findall(r'^E2E_TIME (.*)$',p.read_text(),re.M)
 if len(lines)!=1:raise ValueError('missing timing: '+str(p))
 return {k:float(v) for k,v in re.findall(r'(\w+)=([\d.eE+-]+)',lines[0])}
def interval(v):
 rng=random.Random(6006);boot=sorted(statistics.median(rng.choices(v,k=len(v))) for _ in range(5000))
 return statistics.median(v),boot[125],boot[4874]
def paired(cell,a,b):
 values=[];sessions=set();starts=set();order=['control','champion','top1','top2','top3']
 for i in range(25):
  times={}
  for arm in (a,b):
   p=cell/'measure'/arm/f'r{i}.log';m=json.loads(p.with_suffix('.json').read_text())
   assert m['exit_code']==0 and 'RESULT status=PASS' in p.read_text(),str(p)
   assert m['round']==i and m['order']==(order.index(arm)-i)%5,str(p)
   sessions.add(m['session']);starts.add(m['started_ns']);times[arm]=timing(p)['l2_ms']
  values.append(times[a]/times[b])
 assert len(sessions)==1 and len(starts)==50
 return interval(values)
def chosen(cell):
 frozen=json.loads((cell/'choice.json').read_text());excluded=json.loads((cell/'excluded.json').read_text()) if (cell/'excluded.json').exists() else {}
 med={a:statistics.median(timing(cell/'pilot'/a/f'r{i}.log')['l2_ms'] for i in range(5)) for a in ('top1','top2','top3') if a not in excluded}
 assert frozen['arm']==min(med,key=med.get)
 for p in (cell/'measure'/frozen['arm']).glob('r*.json'):assert json.loads(p.read_text())['started_ns']>frozen['frozen_ns']
 return frozen['arm']
def trace_cell(cell,arm):
 specs=json.loads((cell/'specs.json').read_text());dump=cell/'trace'/arm/'dump'
 a=trace.analyze(dump,Path(specs[arm]['source']),1)
 _,slots,_,_=trace.load(dump);by={(r['stage'],r['logical_task']):r for r in slots}
 path=[by[tuple(map(int,p.split(':')))] for p in a['cp_corrected_path'].split(',')]
 durations=[r['run_end']-r['run_begin'] for r in path]
 cp=a['cp_corrected_ns'];queue=a['queue_lb_ms']*1e6
 return dict(cp_ns=cp,queue_lb_ns=queue,floor_ns=max(cp,queue),nodes=len(path),node_mean_ns=statistics.mean(durations),queue_over_cp=queue/cp,dump=str(dump))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['gqa2','mha4','real']);ap.add_argument('--seqs',nargs='+',type=int,default=[4,128]);ap.add_argument('--raw',type=Path);ap.add_argument('--out',type=Path,default=HERE);a=ap.parse_args();rows=[]
 mappings=({f'{m}_s{s}':str(a.raw/f'{m}_s{s}') for m in a.models for s in a.seqs} if a.raw else json.loads((HERE/'cells.json').read_text()))
 for name,path in mappings.items():
  if name.split('_s')[0] not in a.models or int(name.split('_s')[1]) not in a.seqs:continue
  cell=REPO/path;arm=chosen(cell);row=dict(cell=name,arm=arm)
  for control in ('control','champion'):
   med,lo,hi=paired(cell,arm,control);row.update({control+'_ratio':med,control+'_ci_lo':lo,control+'_ci_hi':hi})
  med,lo,hi=paired(cell,'champion','control');row.update(r5_champion_control_ratio=med,r5_champion_control_ci_lo=lo,r5_champion_control_ci_hi=hi)
  times=[timing(cell/'measure'/arm/f'r{i}.log') for i in range(25)]
  row['l2_ms']=statistics.median(t['l2_ms'] for t in times);row['l2_l1_ratio']=statistics.median(t['l2_ms']/t['l1_ms'] for t in times)
  top=[x for x in ('top1','top2','top3') if (cell/'measure'/x/'r24.log').exists()]
  ranking=sorted(top,key=lambda x:statistics.median(timing(cell/'measure'/x/f'r{i}.log')['l2_ms'] for i in range(25)))
  row['predicted_top1_rank']=ranking.index('top1')+1 if 'top1' in ranking else -1
  for x in (arm,'champion','control'):
   data=trace_cell(cell,x)
   for k,v in data.items():row[x+'_'+k]=v
  row['measured_floor_ratio']=row['l2_ms']*1e6/row[arm+'_floor_ns'];rows.append(row)
  print(json.dumps(row),flush=True)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 a.out.mkdir(parents=True,exist_ok=True)
 with (a.out/('comparisons_'+('_'.join(a.models))+('' if a.seqs==[4,128] else '_s'+'_'.join(map(str,a.seqs)))+'.tsv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
