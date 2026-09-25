#!/usr/bin/env python3
"""Search seven theta points from one imported model; no GPU compilation.

Usage: run_theta.py llama|qwen3. Reads the existing symbolic s64 export, cached
variant resources, and uniform legacy seeds. For seq 2/8/32, the starting seed
is the next measured legacy seq (4/16/64); completed.tsv records that choice.
All seven points run the full domain and both coordinate-descent starting points.
"""
import argparse,hashlib,json,pathlib,shutil,subprocess,time
p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',choices=['llama','qwen3']);a=p.parse_args()
E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2]
old=E.parent/'SOLVER_V2/legacy_r8_domain';cell=f'{a.model}_s64'
export=json.loads((old/cell/'solve.json').read_text())['command'][1]
fixture=json.loads((E/'floor'/f'{cell}.command.json').read_text())[5]
out=E/'theta';out.mkdir(exist_ok=True)
binary=pathlib.Path('/root/r9b_work')/f'flow-grid-{a.model}'
shutil.copy2(ROOT/'build-portable/tools/tilemega-flow-grid',binary)
cmd=[str(binary),export,str(E/'fit/target.json'),fixture,str(E/'resources'),str(old),a.model,str(out)]
record=dict(command=cmd,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
(out/f'{a.model}.command.json').write_text(json.dumps(record,indent=2)+'\n');start=time.monotonic()
with (out/f'{a.model}.log').open('x') as log:code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
record.update(exit=code,wall_seconds=time.monotonic()-start)
(out/f'{a.model}.exit.json').write_text(json.dumps(record,indent=2)+'\n');raise SystemExit(code)
