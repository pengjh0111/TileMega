#!/usr/bin/env python3
"""Fifty fresh processes for the explicitly cut Llama MLP program."""
import argparse,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=HERE/'llama_mlp');args=ap.parse_args()
root=args.root;binary=root/'bin/selected';output=root/'correctness';output.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
session=str(time.time_ns())
for i in range(50):
 p=output/f'r{i}.log'
 if p.exists():raise RuntimeError('refusing overwrite '+str(p))
 env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')};env.update(TILEMEGA_WARMUP='0',TILEMEGA_REPEAT='1')
 cmd=[str(binary),str(root/'fixture')]
 with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns();r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=300)
 p.write_text(r.stdout+r.stderr);p.with_suffix('.json').write_text(json.dumps(dict(command=cmd,exit_code=r.returncode,session=session,round=i,started_ns=start,elapsed_ns=time.time_ns()-start,binary_sha256=sha(binary),environment={k:v for k,v in env.items() if k.startswith('TILEMEGA_')}),indent=2)+'\n')
 if r.returncode or 'RESULT status=PASS' not in p.read_text():raise RuntimeError('MLP subset correctness failed: '+str(p))
 print('MLP_CORRECTNESS',i+1,'/50 PASS',flush=True)
