#!/usr/bin/env python3
"""Compile and measure R9b candidates, with raw process and SASS evidence.

--top3 reads the compiler's shortlist and selects the fastest internally exact
candidate. Every timing uses ten fresh processes and MIDPOINT_REFINE=0.
--processes 50 supplies the separate synchronization check on an existing source.
Only R9b directories are written; prior rounds remain immutable.
"""
import argparse,csv,fcntl,hashlib,json,os,pathlib,shutil,statistics,sys,re
E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2]
sys.path.insert(0,str(E.parent/'SOLVER_V2'))
from gpu_admission import wait_for_device
# Avoid importing this module under its own basename.
import importlib.util
spec=importlib.util.spec_from_file_location('r9_measure',E.parent/'SOLVER_V2/measure.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from run_controls import audit

def measure(source,fixture,out,processes=10):
 out.mkdir(parents=True,exist_ok=True)
 env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
 env.update(TILEMEGA_WARMUP='2',TILEMEGA_REPEAT='10')
 need=8192;free=shutil.disk_usage(out).free//2**20
 (out/'disk.json').write_text(json.dumps(dict(NEED_MIB=need,FREE_MIB=free))+'\n')
 if free<need:raise RuntimeError('insufficient compile disk')
 binary=out/'kernel'
 cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-arch=sm_89','-lineinfo','-Xptxas=-v','-DTILEMEGA_MIDPOINT_REFINE=0']
 for sub in ('include','third_party/cutlass/include','third_party/cutlass/tools/util/include','third_party/cutlass/test'):cmd+=['-I'+str(ROOT/sub)]
 cmd += [source,ROOT/'build-portable/libtilemega.a','-L/usr/local/cuda/lib64','-lcudart','-o',binary]
 if (out/'build.command.json').exists():
  previous=json.loads((out/'source.json').read_text())
  if previous['sha256']!=hashlib.sha256(source.read_bytes()).hexdigest():raise RuntimeError('source changed; use a new evidence directory')
 else:
  (out/'source.json').write_text(json.dumps(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))+'\n')
  if old.run(cmd,out/'build.log',env):return dict(status='build_failed',source=str(source))
  audit(binary,out)
 samples=[]
 for i in range(processes):
  if (E/'global_stop.json').exists():raise RuntimeError('R9b global stop recorded; inspect global_stop.json')
  log=out/f'process_{i:02}.log'
  if not log.exists():
   # Release the lock between fresh processes so independent cells can advance.
   with open('/tmp/tilemega-r9-gpu.lock','w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);wait_for_device(fixture,out/'gpu_admission.jsonl')
    code=old.run([binary,fixture],log,env,timeout=600)
  good,times=old.parse(log.read_text())
  print(f'R9B_MEASURE source={source} process={i} internal={good}',flush=True)
  if not good or not times:
   if 'E2E_HASH ' in log.read_text():
    (E/'global_stop.json').write_text(json.dumps(dict(reason='internal bitwise correctness failure after execution',source=str(source),log=str(log)),indent=2)+'\n')
    raise RuntimeError('R9b §7.4 internal correctness regression; inspect global_stop.json')
   return dict(status='internal_or_preflight_failed',source=str(source),log=str(log))
  cg=source.with_suffix('.mlir')
  if cg.exists():
   floor=re.search(r'floor_value_ns = ([0-9.eE+-]+)',cg.read_text())
   if floor:
    cal=json.loads((E/'fit/target.json').read_text())['calibration_by_dtype']['bf16']['pipelines']
    threshold=float(floor[1])-cal['l2_knee_bytes']/cal['dram_gbps']
    if times[2]*1e6<threshold:
     (E/'global_stop.json').write_text(json.dumps(dict(reason='measured L2 below DRAM floor minus maximum L2 residue',source=str(source),log=str(log),l2_ns=times[2]*1e6,threshold_ns=threshold),indent=2)+'\n')
     raise RuntimeError('R9b §7.4 floor violation; all dependent measurements must stop')
  samples.append(times)
 result=dict(status='ok',source=str(source),directory=str(out),processes=processes,
             **dict(zip(['l05_ms','l1_ms','l2_ms'],map(statistics.median,zip(*samples)))))
 cg=source.with_suffix('.mlir')
 bound=re.search(r'floor_value_ns = ([0-9.eE+-]+)',cg.read_text()) if cg.exists() else None
 if bound:
  result['floor_ns']=float(bound[1]);result['l2_over_floor']=result['l2_ms']*1e6/result['floor_ns']
 print('R9B_RESULT '+json.dumps(result),flush=True)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--fixture',type=pathlib.Path,required=True)
 p.add_argument('--top3',action='store_true');p.add_argument('--processes',type=int,default=10);p.add_argument('--out',type=pathlib.Path);a=p.parse_args()
 source=a.source.resolve();fixture=a.fixture.resolve()
 if a.top3:
  with pathlib.Path(str(source)+'.top3.tsv').open() as f:candidates=list(csv.DictReader(f,delimiter='\t'))
 else:candidates=[dict(source=str(source),cg=str(source.with_suffix('.mlir')))]
 records=[]
 for candidate in candidates:
  cu=pathlib.Path(candidate['source']);out=a.out or pathlib.Path(str(cu)+'.measurement')
  result=measure(cu,fixture,out,a.processes);result['candidate']=candidate;records.append(result)
 valid=[x for x in records if x['status']=='ok'];winner=min(valid,key=lambda x:x['l2_ms']) if valid else None
 if winner:
  # The selected artifact is concrete; provenance keeps the original top-3 key.
  for field,suffix in [('source','.cu'),('cg','.mlir')]:
   path=pathlib.Path(winner['candidate'][field]);destination=pathlib.Path(str(source)+'.measured'+suffix)
   if path.exists():shutil.copy2(path,destination)
 pathlib.Path(str(source)+'.measurements.json').write_text(json.dumps(dict(candidates=records,winner=winner),indent=2)+'\n')
 if not winner:raise SystemExit(1)
if __name__=='__main__':main()
