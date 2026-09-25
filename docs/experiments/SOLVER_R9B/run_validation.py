#!/usr/bin/env python3
import argparse,hashlib,json,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('model',choices=['llama','qwen3']);p.add_argument('--label');p.add_argument('--colocate',action='store_true');p.add_argument('--prices-only',action='store_true');a=p.parse_args()
E=pathlib.Path(__file__).resolve().parent;root=E.parents[2];old=E.parent/'SOLVER_V2/legacy_r8_domain'/f'{a.model}_s4'
export=json.loads((old/'solve.json').read_text())['command'][1];fixture=json.loads((E/'floor'/f'{a.model}_s4.command.json').read_text())[5]
out=E/(a.label or ('piece_validation' if a.prices_only else 'validation_colocated' if a.colocate else 'validation'))/a.model;out.mkdir(parents=True,exist_ok=True)
if (out/'run.log').exists():raise RuntimeError('refusing overwrite')
binary=root/'build-portable/tools/tilemega-flow-validation'
cmd=[str(binary),export,str(E/'fit/target.json'),fixture,str(E/'resources'),str(out),'3' if a.prices_only else '100','90109',str(int(a.colocate))]
if a.prices_only:cmd+=['--prices-only']
(out/'command.json').write_text(json.dumps(dict(command=cmd,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()),indent=2)+'\n')
start=time.monotonic()
with (out/'run.log').open('w') as log:code=subprocess.call(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
(out/'exit.json').write_text(json.dumps(dict(exit=code,wall_seconds=time.monotonic()-start))+'\n');raise SystemExit(code)
