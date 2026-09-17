#!/usr/bin/env python3
"""Re-materialize each frozen reference configuration at seq 4/128, past 0/512.

Geometry, kappa, residency and placement family stay those of the frozen
selection. The compiler pass regenerates theta-dependent EFT queues; no table
is reused across theta. The full default 2048-cell scan is in WRITEBACK/legacy.
"""
import argparse,concurrent.futures,csv,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'JOINT'));import measure as r5

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['generate','build','run']);a=ap.parse_args();root=HERE/'selected_seqscan';root.mkdir(exist_ok=True);session=str(time.time_ns());cells=[]
 for m in ('gqa2','mha4'):
  for s in (4,128):
   src=REPO/json.loads((HERE/'cells.json').read_text())[f'{m}_s{s}'];choice=json.loads((src/'choice.json').read_text())['arm'];row=next(r for r in csv.DictReader((src/'auto.cu.top3.tsv').open(),delimiter='\t') if r['rank']==choice[3:])
   for p in (0,512):
    cell=root/f'{m}_s{s}_p{p}';cell.mkdir(exist_ok=True);cells.append((cell,m,s,p,row))
 if a.action=='generate':
  for cell,m,s,p,row in cells:
   cmd=['/tmp/r6_seqscan',str(REPO/f'docs/experiments/SEQSCAN/raw/export/{m}.json'),row['cg'],str(HERE.parent/'COSTMODEL/event_fit/target.json'),str(s),str(p),str(cell/'model.cu'),str(HERE.parent/'SIMULATOR/hop_ns.tsv')]
   with (cell/'generate.log').open('w') as f:subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
   (cell/'generate.json').write_text(json.dumps(dict(command=cmd,source_choice=row,session=session),indent=2)+'\n')
 elif a.action=='build':
  import shutil
  free=shutil.disk_usage(root).free//2**20;print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
  if free<8192:raise RuntimeError('disk budget')
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   fs=[pool.submit(r5.build,c,m,'selected',dict(source=str(c/'model.cu'),kappa=r['kappa'],residency=r['residency'],placement_macro='0'),'sm_89') for c,m,s,p,r in cells]
   if any(f.result()!=0 for f in fs):raise RuntimeError('selected SEQSCAN compile failed')
 else:
  failed=[]
  for c,m,s,p,r in cells:
   try:
    for i in range(50):r5.run(c,m,s,'selected',c/'correctness',i,0,session,past=p)
    print('SELECTED_SEQSCAN',m,s,p,'50/50',flush=True)
   except Exception as e:failed.append(str(e));print('SELECTED_SEQSCAN FAIL',m,s,p,str(e),flush=True)
  if failed:raise RuntimeError('; '.join(failed))
if __name__=='__main__':main()
