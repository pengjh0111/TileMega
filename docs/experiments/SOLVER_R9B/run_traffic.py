#!/usr/bin/env python3
"""Audit full-graph nominal/physical task traffic; wait for measured winners.

CPU only. Counts are summed task accesses (including repeated reads), not the
unique-byte DRAM floor and not inferred device transactions. Each space is
counted exactly from boundary pieces or its coordinate-varying pieces.
"""
import hashlib,json,pathlib,re,shutil,subprocess,time
E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2]
binary=pathlib.Path('/root/r9b_work/flow-traffic-audit-v2')
if not binary.exists():shutil.copy2(ROOT/'build-portable/tools/tilemega-flow-audit',binary)
pending={(arm,f'{model}_s{seq}') for arm in ('legacy','skeleton') for model in ('llama','qwen3') for seq in (1,4,16,64)}
failed=[]
while pending:
 progressed=False
 for arm,cell in sorted(pending):
  out=E/'traffic'/arm/cell;result=out/'exit.json'
  if result.exists():
   if json.loads(result.read_text())['exit']:failed.append(f'{arm}/{cell}')
   pending.remove((arm,cell));continue
  if arm=='legacy':cg=E.parent/'SOLVER_V2/legacy_r8_domain'/cell/'selected.mlir'
  else:
   winner=E/'matrix'/cell/'selected.cu.measurements.json'
   if not winner.exists():continue
   cg=pathlib.Path(json.loads(winner.read_text())['winner']['candidate']['cg'])
  text=cg.read_text();seq=cell.split('_s')[1]
  residency=re.search(r'tmexec.solved_residency = (\d+)',text)[1]
  kappa=re.search(r'tmexec.solved_kappa = (\d+)',text)[1]
  fixture=json.loads((E/'floor'/f'{cell}.command.json').read_text())[5]
  out.mkdir(parents=True,exist_ok=True)
  cmd=[str(binary),str(cg),str(E/'fit/target.json'),seq,fixture,residency,kappa,'--traffic',str(out/'spaces.tsv')]
  (out/'command.json').write_text(json.dumps(dict(command=cmd,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),cg_sha256=hashlib.sha256(cg.read_bytes()).hexdigest()),indent=2)+'\n')
  start=time.monotonic()
  with (out/'run.log').open('x') as log:code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  result.write_text(json.dumps(dict(exit=code,seconds=time.monotonic()-start))+'\n')
  print(f'TRAFFIC {arm} {cell} exit={code}',flush=True)
  if code:failed.append(f'{arm}/{cell}')
  pending.remove((arm,cell));progressed=True
 (E/'traffic_progress.json').write_text(json.dumps(dict(pending=sorted(pending),failed=failed),indent=2)+'\n')
 if pending and not progressed:time.sleep(30)
if failed:raise SystemExit(1)
