#!/usr/bin/env python3
"""One fresh replay, with exact binary provenance and round-trippable floats."""
import argparse,hashlib,json,pathlib,subprocess,time
parser=argparse.ArgumentParser();parser.add_argument('label');parser.add_argument('binary');parser.add_argument('dtype',choices=['f32','bf16']);parser.add_argument('--mode');parser.add_argument('--target');a=parser.parse_args()
E=pathlib.Path(__file__).resolve().parent;root=E.parents[2];out=E/'replay'/a.label;out.mkdir(parents=True,exist_ok=True)
if (out/'run.log').exists():raise RuntimeError('refusing overwrite')
binary=pathlib.Path(a.binary).resolve();raw=root/'docs/experiments/ORACLE'/('raw_bf16' if a.dtype=='bf16' else 'raw')
cmd=[str(binary),'--repo',str(root),'--out',str(out),'--target',a.target or str(E/'fit/target.json'),'--dtype',a.dtype,'--screen-dir',str(raw),'--register-dir',str(raw/'cost' if a.dtype=='bf16' else root/'docs/experiments/COST_MODEL/raw'),'--full-only']
if a.mode:cmd+=['--regime-a',a.mode]
(out/'command.json').write_text(json.dumps(dict(command=cmd,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()),indent=2)+'\n')
start=time.monotonic()
with (out/'run.log').open('w') as log:code=subprocess.call(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
(out/'exit.json').write_text(json.dumps(dict(exit=code,wall_seconds=time.monotonic()-start))+'\n')
raise SystemExit(code)
