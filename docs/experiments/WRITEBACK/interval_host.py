#!/usr/bin/env python3
"""Dump the actual host queues selected by a single interval binary."""
import csv,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'interval_closure'
for seq in range(1,6):
 folder=ROOT/'host'/f's{seq}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'run.log').exists():raise RuntimeError('refusing overwrite')
 env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
 env.update(TILEMEGA_WARMUP='0',TILEMEGA_REPEAT='1',TILEMEGA_PLAN_DUMP=str(folder))
 cmd=[str(ROOT/'bin/interval'),str(ROOT/'fixtures'/f's{seq}')]
 with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns();r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=300)
 (folder/'run.log').write_text(r.stdout+r.stderr)
 (folder/'run.json').write_text(json.dumps(dict(command=cmd,started_ns=start,exit_code=r.returncode,binary_sha256=hashlib.sha256((ROOT/'bin/interval').read_bytes()).hexdigest()))+'\n')
 if r.returncode or 'RESULT status=PASS' not in r.stdout:raise RuntimeError(str(folder))
 print('INTERVAL_HOST',seq,'PASS',flush=True)
