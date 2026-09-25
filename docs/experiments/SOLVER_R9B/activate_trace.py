#!/usr/bin/env python3
"""Run existing TRACE_V2 builds with both runtime activation and output path.

Initial SV-9 processes set only the output path and therefore did not allocate
trace buffers. Preserve those logs; new activated processes have separate names.
"""
import fcntl,json,os,pathlib,sys
E=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(E.parent/'SOLVER_V2'))
from measure import run,parse
from gpu_admission import wait_for_device
for model in ('llama','qwen3'):
 for seq in (1,64):
  cell=f'{model}_s{seq}';out=E/'trace'/cell
  if (out/'activated.log').exists():raise RuntimeError('refusing overwrite '+str(out))
  fixture=pathlib.Path(json.loads((E/'floor'/f'{cell}.command.json').read_text())[5])
  env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
  env.update(TILEMEGA_TRACE_V2='1',TILEMEGA_TRACE_V2_OUT=str(out/'dump'),TILEMEGA_WARMUP='1',TILEMEGA_REPEAT='1')
  with open('/tmp/tilemega-r9-gpu.lock','w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);wait_for_device(fixture,out/'activated_admission.jsonl')
   code=run([out/'kernel',fixture],out/'activated.log',env,timeout=600)
  good,_=parse((out/'activated.log').read_text())
  if not good:raise RuntimeError('trace internal equality failed '+cell)
  if not (out/'dump/slots.tsv').exists():raise RuntimeError('trace dump missing '+cell)
  print(cell,'exit',code,'internal=PASS',flush=True)
