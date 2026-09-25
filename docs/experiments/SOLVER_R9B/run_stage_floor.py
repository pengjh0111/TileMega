#!/usr/bin/env python3
"""Per-logical-stage unique external bytes for the four trace controls."""
import json,pathlib,subprocess,sys,hashlib
E=pathlib.Path(__file__).resolve().parent;root=E.parents[2]
for model in ('llama','qwen3'):
 for seq in (1,64):
  cell=f'{model}_s{seq}';out=E/'stage_floor'/cell;out.mkdir(parents=True,exist_ok=True)
  original=json.loads((E/'floor'/f'{cell}.command.json').read_text());cmd=original[:]
  cmd[0]=str(root/'build-portable/tools/tilemega-dram-floor');cmd[6]=str(out/'floor.mlir');cmd[7]=str(out/'tensors.tsv');cmd.append(str(out/'stages.tsv'))
  if (out/'run.log').exists():raise RuntimeError('refusing overwrite')
  (out/'command.json').write_text(json.dumps(dict(command=cmd,binary_sha256=hashlib.sha256(pathlib.Path(cmd[0]).read_bytes()).hexdigest()),indent=2)+'\n')
  with (out/'run.log').open('w') as log:code=subprocess.call(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  if code:raise SystemExit(code)
