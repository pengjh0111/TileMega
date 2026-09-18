#!/usr/bin/env python3
"""Re-run the maximal connected Llama graph for numerical admission in 50 processes.

The graph, the geometry, the fixture, the seed, the residual edges and the
0.0231875014 tolerance are R6's.  What changed is the build: the switches under
test are passed in, and every process is fresh.
"""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'));import measure
GEOMETRIES={'admitted2':'covered_llama_admitted2','split8':'covered_llama'}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--arch',default='sm_89');ap.add_argument('--geometry',default='admitted2',choices=sorted(GEOMETRIES))
 ap.add_argument('--arm',default='refine');ap.add_argument('--rounds',type=int,default=50)
 ap.add_argument('--extra',default='NORM_EPSILON=1e-5f,ROPE_FP32_PHASE=1,MIDPOINT_REFINE=1')
 a=ap.parse_args()
 models=REPO/'docs/experiments/MODELS';source=models/GEOMETRIES[a.geometry]/'auto.cu'
 fixture=models/'covered_llama/fixture';cell=HERE/'admission'/a.geometry;cell.mkdir(parents=True,exist_ok=True)
 free=shutil.disk_usage(HERE).free//2**20;print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
 if free<8192:raise RuntimeError('disk budget')
 extra=[x for x in a.extra.split(',') if x]
 spec=dict(source=str(source),kappa='1',residency='3',placement_macro='0',extra=extra)
 if measure.build(cell,'llama',a.arm,spec,a.arch):raise RuntimeError('compile failed')
 binary=cell/'bin'/a.arm;folder=cell/'correctness';folder.mkdir(exist_ok=True);session=str(time.time_ns())
 digest=hashlib.sha256(binary.read_bytes()).hexdigest()
 for i in range(a.rounds):
  log=folder/f'r{i}.log'
  if log.exists():raise RuntimeError('refusing overwrite '+str(log))
  env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
  env.update(TILEMEGA_WARMUP='0',TILEMEGA_REPEAT='1')
  command=[str(binary.resolve()),str(fixture.resolve())]
  with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns()
   r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=1800)
  log.write_text(r.stdout+r.stderr)
  log.with_suffix('.json').write_text(json.dumps(dict(command=command,round=i,session=session,defines=extra,
   started_ns=start,elapsed_ns=time.time_ns()-start,exit_code=r.returncode,binary_sha256=digest))+'\n')
  if r.returncode or 'RESULT status=PASS' not in r.stdout:raise RuntimeError(str(log))
  print('ADMIT',a.geometry,a.arm,i+1,'/',a.rounds,flush=True)
if __name__=='__main__':main()
