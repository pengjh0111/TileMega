#!/usr/bin/env python3
"""Solve, compile and internally compare the four BF16 reference cells."""
import json,pathlib,subprocess,sys,time,hashlib
E=pathlib.Path(__file__).resolve().parent;root=E.parents[2]
for model in ('gqa2','mha4'):
 for seq in (4,128):
  cell=f'{model}_s{seq}';old=E.parent/'SOLVER_V2/reference'/cell;out=E/'reference'/cell;out.mkdir(parents=True,exist_ok=True)
  args=json.loads((old/'launch.command.json').read_text())['command'];fixture=args[args.index('--fixture')+1];bridge=args[args.index('--bridge')+1]
  binary=root/'build-portable/tools/tilemega-compile'
  cmd=[str(binary),bridge,str(out/'selected.cu'),'--solver','skeleton','--solve',str(E/'fit/target.json'),'--legacy-seed',str(old/'legacy/selected.mlir'),'--seq',str(seq),'--past','3','--hop-curve',str(E.parent/'SIMULATOR/hop_ns.tsv'),'--variant-cache','/root/r9_work/variant_resources','--flow-fixture',fixture,'--search-jobs','1','--search-passes','3','--k-base','8','--dump-cg',str(out/'selected.mlir')]
  if (out/'solve.log').exists():raise RuntimeError('refusing overwrite '+cell)
  record=dict(command=cmd,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest());(out/'solve.command.json').write_text(json.dumps(record,indent=2)+'\n');start=time.monotonic()
  with (out/'solve.log').open('w') as log:code=subprocess.call(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  record.update(exit=code,wall_seconds=time.monotonic()-start);(out/'solve.exit.json').write_text(json.dumps(record,indent=2)+'\n')
  if code:print(cell,'solve failed',code,flush=True);continue
  with (out/'measurement_runner.log').open('w') as log:code=subprocess.call([sys.executable,str(E/'measure.py'),'--source',str(out/'selected.cu'),'--fixture',fixture,'--top3'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
  if code:raise RuntimeError('reference measurement failed '+cell)
