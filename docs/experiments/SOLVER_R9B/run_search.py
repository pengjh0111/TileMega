#!/usr/bin/env python3
"""Run a recorded R9b search. --search-only skips top-M and GPU compilation."""
import argparse,hashlib,json,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('model',choices=['llama','qwen3']);p.add_argument('seq',type=int);p.add_argument('--search-only',action='store_true');p.add_argument('--pilot',action='store_true');p.add_argument('--label');a=p.parse_args()
E=pathlib.Path(__file__).resolve().parent;root=E.parents[2];cell=f'{a.model}_s{a.seq}';old=E.parent/'SOLVER_V2/legacy_r8_domain'/cell
out=E/(a.label or ('search_pilot' if a.pilot else 'search_cpu' if a.search_only else 'matrix'))/cell;out.mkdir(parents=True,exist_ok=True)
if (out/'solve.log').exists():raise RuntimeError('refusing overwrite')
export=json.loads((old/'solve.json').read_text())['command'][1];fixture=json.loads((E/'floor'/f'{cell}.command.json').read_text())[5]
binary=root/'build-portable/tools/tilemega-compile'
cmd=[str(binary),export,str(out/'selected.cu'),'--solver','skeleton','--solve',str(E/'fit/target.json'),'--legacy-seed',str(old/'selected.mlir'),'--seq',str(a.seq),'--past','3','--hop-curve',str(E.parent/'SIMULATOR/hop_ns.tsv'),'--variant-cache','/root/r9_work/variant_resources','--flow-fixture',fixture,'--search-jobs','1','--search-passes','3','--k-base','8','--dump-cg',str(out/'selected.mlir')]
if a.search_only:cmd+=['--flow-search-only','1']
if a.pilot:cmd+=['--search-domain','/root/r9_work/smoke_domain.json']
meta=dict(command=cmd,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),started_ns=time.time_ns(),pilot=a.pilot)
(out/'solve.command.json').write_text(json.dumps(meta,indent=2)+'\n');start=time.monotonic()
with (out/'solve.log').open('w') as log:code=subprocess.call(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
meta.update(exit=code,wall_seconds=time.monotonic()-start)
(out/'solve.exit.json').write_text(json.dumps(meta,indent=2)+'\n');raise SystemExit(code)
