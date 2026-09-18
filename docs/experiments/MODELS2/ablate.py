#!/usr/bin/env python3
"""Ablate the epsilon, RoPE phase and midpoint refinement changes on the covered graph.

Every arm compiles the frozen R6 sources -- the graph, the geometry, the fixture,
the seed and the tolerance are the ones that failed admission, so the only moving
part is the switch under test.  Each arm reports the first connected V projection
element that F-203 localized, recomputed against its own dumped operands.
"""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'));import measure

# Llama-3.2-1B's rms_norm_eps.  Stated here only to drive the A1 arm of the
# ablation: the generated path takes it from the imported model.
ARMS={'base':[],
      'a1':['NORM_EPSILON=1e-5f'],
      'a2':['ROPE_FP32_PHASE=1'],
      'a1a2':['NORM_EPSILON=1e-5f','ROPE_FP32_PHASE=1'],
      'refine':['NORM_EPSILON=1e-5f','ROPE_FP32_PHASE=1','MIDPOINT_REFINE=1']}
GEOMETRIES={'admitted2':'covered_llama_admitted2','split8':'covered_llama'}

def element(dump):
 """V[0,463] of the first layer, beside the FP64 value it should round from."""
 import torch
 read=lambda i,shape:torch.frombuffer(bytearray((dump/f'buffer_{i}.bin').read_bytes()),dtype=torch.bfloat16).reshape(shape)
 x,weight,v=read(0,(4,2048)),read(5,(512,2048)),read(6,(4,512))
 exact=torch.nn.functional.linear(x.double(),weight.double())
 return dict(gpu=float(v[0,463]),fp64_accumulator=float(exact[0,463]),
             fp64_rounded=float(exact[0,463].to(torch.bfloat16)),
             elements_differing_from_fp64=int((v!=exact.to(torch.bfloat16)).sum()))

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--arch',default='sm_89');ap.add_argument('--geometry',default='admitted2',choices=sorted(GEOMETRIES))
 ap.add_argument('--arms',default='base,a1,a2,a1a2,refine');ap.add_argument('--keep-dump',action='store_true')
 a=ap.parse_args()
 models=REPO/'docs/experiments/MODELS';source=models/GEOMETRIES[a.geometry]/'auto.cu'
 fixture=models/'covered_llama/fixture';cell=HERE/'ablation'/a.geometry;cell.mkdir(parents=True,exist_ok=True)
 free=shutil.disk_usage(HERE).free//2**20;print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
 if free<8192:raise RuntimeError('disk budget')
 for arm in a.arms.split(','):
  folder=cell/arm;folder.mkdir(exist_ok=True);log=folder/'run.log'
  if log.exists():raise RuntimeError('refusing overwrite '+str(log))
  spec=dict(source=str(source),kappa='1',residency='3',placement_macro='0',extra=ARMS[arm])
  if measure.build(cell,'llama',arm,spec,a.arch):raise RuntimeError('compile failed '+arm)
  binary=cell/'bin'/arm;dump=folder/'dump';dump.mkdir(exist_ok=True)
  env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
  env.update(TILEMEGA_WARMUP='0',TILEMEGA_REPEAT='1',TILEMEGA_DIFF_DUMP='8',TILEMEGA_DUMP_BUFFERS=str(dump.resolve()))
  command=[str(binary.resolve()),str(fixture.resolve())]
  with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns()
   r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=1800)
 
  log.write_text(r.stdout+r.stderr)
  row=dict(command=command,arm=arm,geometry=a.geometry,defines=ARMS[arm],started_ns=start,
           elapsed_ns=time.time_ns()-start,exit_code=r.returncode,
           source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
           binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
           head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
           passed='RESULT status=PASS' in r.stdout)
  row['v_0_463']=element(dump)
  (folder/'run.json').write_text(json.dumps(row,indent=2)+'\n')
  if not a.keep_dump:
   for f in dump.iterdir():
    if f.name not in ('buffer_0.bin','buffer_5.bin','buffer_6.bin'):f.unlink()
  print('ABLATE',a.geometry,arm,'pass' if row['passed'] else 'fail','v=%r'%row['v_0_463']['gpu'],flush=True)
if __name__=='__main__':main()
