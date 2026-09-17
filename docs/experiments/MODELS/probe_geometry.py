#!/usr/bin/env python3
"""Numerical diagnostics only; these manually requested plans are not solver results."""
import concurrent.futures,json,os,fcntl,hashlib,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];ROOT=HERE/'covered_geometry_probes'
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'));import measure
ROOT.mkdir(exist_ok=True)
def probe(shape):
 m,n,k=shape;root=ROOT/f'm{m}n{n}k{k}split1';root.mkdir(exist_ok=True)
 plan=dict(schema='tilemega.runtime_variants.v1',variants=[dict(seq_begin=1,seq_end=4,uniform=dict(tile_m=m,tile_n=n,tile_k=k,stages=2,split_k=1),rope_tile_per_block=True,kv_tile_per_block=True,activation_tile_per_block=True,combiner_tile_per_block=True)])
 (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 cmd=[str(REPO/'build-portable/tools/tilemega-compile'),str(HERE/'covered_llama/auto.cu.export.json'),str(root/'model.cu'),'--variants',str(root/'plan.json')]
 with (root/'generate.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 (root/'generate.json').write_text(json.dumps(dict(command=cmd,scope=__doc__))+'\n')
 import shutil
 free=shutil.disk_usage(root).free//2**20;print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
 if free<8192:raise RuntimeError('disk')
 if measure.build(root,'llama','probe',dict(source=str(root/'model.cu'),kappa='1',residency='1',placement_macro='5'),'sm_89'):raise RuntimeError('build failed')
 binary=root/'bin/probe';cmd=[str(binary),str(HERE/'covered_llama/fixture')]
 env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')};env.update(TILEMEGA_WARMUP='0',TILEMEGA_REPEAT='1',TILEMEGA_DIFF_DUMP='10')
 with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns();r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=600)
 (root/'run.log').write_text(r.stdout+r.stderr);(root/'run.json').write_text(json.dumps(dict(command=cmd,started_ns=start,exit_code=r.returncode,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),scope=__doc__))+'\n')
 print(root.name,'PASS' if 'RESULT status=PASS' in r.stdout and not r.returncode else 'FAIL',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for f in [pool.submit(probe,s) for s in [(64,128,16),(32,16,64),(32,16,32)]]:f.result()
