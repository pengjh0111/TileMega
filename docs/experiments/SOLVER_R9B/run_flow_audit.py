#!/usr/bin/env python3
"""Audit all eight legacy geometries with fresh CPU flow and FIFO evaluations."""
import argparse,hashlib,json,pathlib,re,subprocess,time,shutil
ROOT=pathlib.Path(__file__).resolve().parents[3]
E=pathlib.Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--label',default='flow_controls');parser.add_argument('--cells');args=parser.parse_args()
selected=set(args.cells.split(',')) if args.cells else None
binary=pathlib.Path('/root/r9b_work')/('flow-audit-'+args.label)
if not binary.exists():shutil.copy2(ROOT/'build-portable/tools/tilemega-flow-audit',binary)
for model in ('llama','qwen3'):
 for seq in (1,4,16,64):
  cell=f'{model}_s{seq}'
  if selected and cell not in selected:continue
  out=E/args.label/cell;out.mkdir(parents=True,exist_ok=True)
  if (out/'run.log').exists():raise RuntimeError('refusing to overwrite '+str(out))
  cg=E.parent/'SOLVER_V2/legacy_r8_domain'/cell/'selected.mlir';text=cg.read_text()
  r=re.search(r'tmexec.solved_residency = (\d+)',text)[1];k=re.search(r'tmexec.solved_kappa = (\d+)',text)[1]
  fixture=json.loads((E/'floor'/f'{cell}.command.json').read_text())[5]
  cmd=[str(binary),str(cg),str(E/'fit/target.json'),str(seq),fixture,r,k,str(out/'home')]
  (out/'command.json').write_text(json.dumps(dict(command=cmd,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()),indent=2)+'\n')
  start=time.monotonic()
  with (out/'run.log').open('w') as log:code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  (out/'exit.json').write_text(json.dumps(dict(exit=code,wall_seconds=time.monotonic()-start))+'\n')
  print(cell,'exit',code,flush=True)
  if code:raise RuntimeError('flow audit failed; inspect '+str(out/'run.log'))
