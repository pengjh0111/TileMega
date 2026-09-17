#!/usr/bin/env python3
"""Paired five-arm attribution from unmodified raw R6 measurements."""
import argparse,csv,importlib.util,json,math,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'JOINT2'));import analyze as joint

def analyze(root):
 specs=json.loads((root/'specs.json').read_text());configs=list(dict.fromkeys(x.split('__')[0] for x in specs));selected=json.loads((root/'selection.json').read_text())['placement'];rows=[]
 for config in configs:
  raw=[]
  for i in range(25):
   arms={a:joint.timing(root/'measure'/(config+'__'+a)/f'r{i}.log') for a in ('full','nofence','nowait','neither','l1nosync')}
   full=arms['full']['l2_ms'];wait=full-arms['nowait']['l2_ms'];notify=arms['nowait']['l2_ms']-arms['neither']['l2_ms'];barrier=arms['full']['l1_ms']-arms['l1nosync']['l1_ms']
   raw.append(dict(l2_ms=full,wait_ms=wait,notify_ms=notify,barrier_ms=barrier,fence_ms=full-arms['nofence']['l2_ms'],protocol_ms=wait+notify,protocol_barrier=(wait+notify)/barrier if barrier else float('nan'),l2_l1=full/arms['full']['l1_ms'],relative_selected=full/joint.timing(root/'measure'/(selected+'__full')/f'r{i}.log')['l2_ms'],relative_legacy=full/joint.timing(root/'measure/legacy_grid_stride__full'/f'r{i}.log')['l2_ms']))
  row=dict(cell=root.name,config=config,selected=selected)
  for key in raw[0]:
   values=[r[key] for r in raw]
   med,lo,hi=joint.interval(values) if all(math.isfinite(v) for v in values) else (float('nan'),)*3
   row.update({key:med,key+'_lo':lo,key+'_hi':hi})
  # Signed interventions remain in all 25 pairs. A zero denominator marks
  # the ratio unidentifiable; it does not silently drop a sample.
  row['nonpositive_barrier_pairs']=sum(r['barrier_ms']<=0 for r in raw)
  row['zero_barrier_pairs']=sum(r['barrier_ms']==0 for r in raw)
  dump=root/'trace'/(config+'__full')/'dump';source=Path(specs[config+'__full']['source'])
  if dump.exists():
   r=joint.trace.analyze(dump,source,1);cp=r['cp_corrected_ns'];queue=r['queue_lb_ms']*1e6;row.update(cp_ns=cp,queue_lb_ns=queue,floor_ns=max(cp,queue),measured_floor=row['l2_ms']*1e6/max(cp,queue),trace=str(dump))
  else:row.update(cp_ns='',queue_lb_ns='',floor_ns='',measured_floor='',trace='MISSING')
  rows.append(row)
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['gqa2','mha4','real']);ap.add_argument('--seqs',nargs='+',type=int,default=[4,128]);a=ap.parse_args();rows=[]
 for m in a.models:
  for s in a.seqs:rows+=analyze(HERE/'raw'/f'{m}_s{s}')
 with (HERE/('attribution_'+ '_'.join(a.models)+('' if a.seqs==[4,128] else '_s'+'_'.join(map(str,a.seqs)))+'.tsv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
 for r in rows:print(json.dumps(r),flush=True)
if __name__=='__main__':main()
