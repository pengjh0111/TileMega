#!/usr/bin/env python3
"""R8 B-a: the role-granularity release litmus, one fresh process per round.

Matrix follows §5.3: grid in {64,128,256}, tiles including the <=4096 elements
F-10 requires, address reuse (two slots rewritten every iteration),
cooperative writes, and the two negative controls. 50 fresh processes per cell.
"""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ARMS=('roles','nofence','nobarrier')
GRIDS=(64,128,256)
ELEMENTS=(256,1024,4096)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--binary',type=Path,required=True)
 ap.add_argument('--out',type=Path,default=HERE/'raw')
 ap.add_argument('--rounds',type=int,default=50)
 ap.add_argument('--iterations',type=int,default=64)
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=[];session=str(time.time_ns())
 for arm in ARMS:
  for grid in GRIDS:
   for elements in ELEMENTS:
    cell=a.out/f'{arm}_g{grid}_e{elements}';cell.mkdir(parents=True,exist_ok=True)
    passing=mismatch_rounds=0
    for round_ in range(a.rounds):
     log=cell/f'r{round_}.log'
     if log.exists():
      text=log.read_text()
     else:
      env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
      cmd=[str(a.binary),str(grid),str(elements),str(a.iterations),arm]
      with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
       fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns()
       r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=900)
      text=r.stdout+r.stderr
      log.write_text(text)
      log.with_suffix('.json').write_text(json.dumps(dict(command=cmd,round=round_,
        session=session,started_ns=start,elapsed_ns=time.time_ns()-start,
        exit_code=r.returncode))+'\n')
     field=dict(kv.split('=',1) for kv in text.split() if '=' in kv)
     if field.get('mismatches')=='0':passing+=1
     else:mismatch_rounds+=1
    rows.append(dict(arm=arm,grid=grid,elements=elements,rounds=a.rounds,
      passing=passing,mismatching=mismatch_rounds))
    print('LITMUS_CELL',arm,grid,elements,'passing',passing,'/',a.rounds,flush=True)
 columns=list(rows[0])
 with (a.out/'litmus.tsv').open('w') as f:
  f.write('\t'.join(columns)+'\n')
  for r in rows:f.write('\t'.join(str(r[c]) for c in columns)+'\n')
 compliant=[r for r in rows if r['arm']=='roles']
 print('B-a compliant cells',len(compliant),'all passing',
       all(r['passing']==r['rounds'] for r in compliant),flush=True)
 for control in ('nofence','nobarrier'):
  cells=[r for r in rows if r['arm']==control]
  failing=[r for r in cells if r['mismatching']==r['rounds']]
  print('B-a control',control,'cells',len(cells),'failing_every_round',len(failing),flush=True)
if __name__=='__main__':main()
