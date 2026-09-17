#!/usr/bin/env python3
"""Build the solver-selected covered graph and verify all outputs in 50 processes."""
import argparse,fcntl,hashlib,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'));import measure

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=HERE/'covered_llama');ap.add_argument('--fixture',type=Path,default=HERE/'covered_llama/fixture');ap.add_argument('--arch',default='sm_89');a=ap.parse_args();ROOT=a.root.resolve()
 free=shutil.disk_usage(ROOT).free//2**20;print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
 if free<8192:raise RuntimeError('disk budget')
 source=ROOT/'auto.cu';text=source.read_text()
 macro=lambda name:re.search(r'^#define '+name+r' (\d+)$',text,re.M)[1]
 spec=dict(source=str(source),kappa=macro('TILEMEGA_EVENT_KAPPA'),residency=macro('TILEMEGA_RESIDENCY_CAP'),placement_macro='0')
 if measure.build(ROOT,'llama','selected',spec,a.arch):raise RuntimeError('compile failed')
 binary=ROOT/'bin/selected';folder=ROOT/'correctness';folder.mkdir(exist_ok=True);session=str(time.time_ns())
 for i in range(50):
  log=folder/f'r{i}.log'
  if log.exists():raise RuntimeError('refusing overwrite '+str(log))
  env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')};env.update(TILEMEGA_WARMUP='0',TILEMEGA_REPEAT='1')
  command=[str(binary),str(a.fixture.resolve())]
  with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns();r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=600)
  log.write_text(r.stdout+r.stderr);log.with_suffix('.json').write_text(json.dumps(dict(command=command,round=i,session=session,started_ns=start,elapsed_ns=time.time_ns()-start,exit_code=r.returncode,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))+'\n')
  if r.returncode or 'RESULT status=PASS' not in r.stdout:raise RuntimeError(str(log))
  print('COVERED_LLAMA',i+1,'/50',flush=True)
if __name__=='__main__':main()
