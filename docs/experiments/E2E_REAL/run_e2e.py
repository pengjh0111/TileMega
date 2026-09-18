#!/usr/bin/env python3
"""Solve, generate, build and check a whole exported decoder in one place.

Every decision below the export -- geometry, split-K, kappa, residency,
placement, slot order -- is the solver's; this script only invokes it, records
what it chose, and runs the result in fresh processes.
"""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'));import measure

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--root',type=Path,required=True,help='directory holding exported_program.pt2 and fixture/')
 ap.add_argument('--arch',default='sm_89');ap.add_argument('--seq',type=int,default=4)
 ap.add_argument('--past',type=int,default=3);ap.add_argument('--rounds',type=int,default=50)
 ap.add_argument('--capacity',type=int,default=12)
 ap.add_argument('--extra',default='MIDPOINT_REFINE=1')
 ap.add_argument('--solve-only',action='store_true')
 a=ap.parse_args();root=a.root.resolve()
 free=shutil.disk_usage(root).free//2**20;print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
 if free<8192:raise RuntimeError('disk budget')
 source=root/'auto.cu'
 if not source.exists():
  command=[str(REPO/'build-portable/tools/tilemega-compile'),str(root/'exported_program.pt2'),
   str(source),'--solve',str(REPO/'docs/experiments/COSTMODEL/event_fit/target.json'),
   '--seq',str(a.seq),'--past',str(a.past),'--search-capacity',str(a.capacity),
   '--search-domain',str(REPO/'docs/experiments/COSTMODEL/event_fit/search_domain.json'),
   '--dump-cg',str(root/'auto.mlir'),
   '--hop-curve',str(REPO/'docs/experiments/SIMULATOR/hop_ns.tsv')]
  start=time.time_ns()
  with (root/'solve.log').open('w') as f:
   r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,cwd=REPO)
  (root/'solve.json').write_text(json.dumps(dict(command=command,exit_code=r.returncode,
   elapsed_ns=time.time_ns()-start,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()),indent=2)+'\n')
  if r.returncode:raise RuntimeError('solve failed; see '+str(root/'solve.log'))
  print('SOLVED',source,flush=True)
 if a.solve_only:return
 text=source.read_text()
 macro=lambda name,default:(lambda m:m[1] if m else default)(__import__('re').search(r'^#define '+name+r' (\d+)$',text,__import__('re').M))
 spec=dict(source=str(source),kappa=macro('TILEMEGA_EVENT_KAPPA','1'),
           residency=macro('TILEMEGA_RESIDENCY_CAP','0'),placement_macro='0',
           extra=[x for x in a.extra.split(',') if x])
 if measure.build(root,'real','selected',spec,a.arch):raise RuntimeError('compile failed')
 binary=root/'bin/selected';folder=root/'correctness';folder.mkdir(exist_ok=True)
 session=str(time.time_ns());digest=sha(binary)
 for i in range(a.rounds):
  log=folder/f'r{i}.log'
  if log.exists():raise RuntimeError('refusing overwrite '+str(log))
  env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
  env.update(TILEMEGA_WARMUP='0',TILEMEGA_REPEAT='1')
  command=[str(binary),str((root/'fixture').resolve())]
  with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns()
   r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=3600)
  log.write_text(r.stdout+r.stderr)
  log.with_suffix('.json').write_text(json.dumps(dict(command=command,round=i,session=session,
   started_ns=start,elapsed_ns=time.time_ns()-start,exit_code=r.returncode,binary_sha256=digest))+'\n')
  if r.returncode or 'RESULT status=PASS' not in r.stdout:raise RuntimeError(str(log))
  print('E2E',root.name,i+1,'/',a.rounds,flush=True)
if __name__=='__main__':main()
