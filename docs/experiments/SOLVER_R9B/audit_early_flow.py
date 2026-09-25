#!/usr/bin/env python3
"""Price the two frozen R9 snapshot geometries with the R9b Level-1 model.

CPU only; the measured R9 placements are unchanged. These predictions use
the uncolocated global-pool model, as the legacy-geometry comparison does.
Products: early_flow/llama_s{1,4}/{command.json,run.log,exit.json}.
"""
import hashlib,json,pathlib,re,shutil,subprocess,time

E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2]
binary=pathlib.Path('/root/r9b_work/early-flow-audit')
if not binary.exists():shutil.copy2(ROOT/'build-portable/tools/tilemega-flow-audit',binary)
for seq in (1,4):
    cell=f'llama_s{seq}';out=E/'early_flow'/cell;out.mkdir(parents=True,exist_ok=True)
    cg=E/'early_resident'/f'{cell}.mlir';text=cg.read_text()
    resident=re.search(r'tmexec.solved_residency = (\d+)',text)[1]
    kappa=re.search(r'tmexec.solved_kappa = (\d+)',text)[1]
    fixture=json.loads((E/'floor'/f'{cell}.command.json').read_text())[5]
    cmd=[str(binary),str(cg),str(E/'fit/target.json'),str(seq),fixture,resident,kappa]
    record=dict(command=cmd,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                cg_sha256=hashlib.sha256(cg.read_bytes()).hexdigest())
    (out/'command.json').write_text(json.dumps(record,indent=2)+'\n')
    started=time.monotonic()
    with (out/'run.log').open('x') as log:
        code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    (out/'exit.json').write_text(json.dumps(dict(exit=code,seconds=time.monotonic()-started))+'\n')
    print(cell,'exit',code,flush=True)
    if code:raise SystemExit(code)
