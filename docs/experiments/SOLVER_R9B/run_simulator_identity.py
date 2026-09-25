#!/usr/bin/env python3
"""Compare every binary64 task/result field on four reference plans, fluid off.

Compiles the identical audit source separately against baseline/current ABI.
No GPU is used; fixed historical trace task durations are replay inputs only.
"""
import hashlib,json,pathlib,shlex,subprocess
E=pathlib.Path(__file__).resolve().parent
ROOT=E.parents[2]
OUT=E/'simulator_identity'
OUT.mkdir(exist_ok=True)
if (OUT/'commands.json').exists():raise RuntimeError('refuse to replace evidence')
records=[]
inputs=[E/'simulator_identity.cpp',E.parent/'SIMULATOR/cell_inputs.h',E.parent/'SIMULATOR/hop_ns.tsv']
for model in ('gqa2','mha4'):
    inputs.append(E.parent/'PLAN_CONTRACT/legacy_identity/plan'/f'{model}.cu')
    for seq in (4,128):
        inputs.extend([E.parent/'SIMULATOR/raw/run'/f'{model}_p5_s{seq}.out',
                       E.parent/'SIMULATOR/raw/dump'/f'{model}_s{seq}_p5/slots.tsv'])
(OUT/'inputs.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2)+'\n')
for arm,build in [('baseline',pathlib.Path('/root/r9b_work/baseline_build')),('current',ROOT/'build-portable')]:
    commands=subprocess.check_output(['ninja','-C',str(build),'-t','commands','tilemega-costmodel'],text=True).splitlines()
    compile=shlex.split(next(x for x in commands if ' -c ' in x and 'tools/tilemega-costmodel.cpp' in x))
    compile=compile[:compile.index('-MD')]+['-o',str(OUT/(arm+'.o')),'-c',str(E/'simulator_identity.cpp')]
    link=shlex.split(commands[-1]);link=link[2:-2]
    link[link.index('-o')+1]=str(OUT/arm)
    object_index=next(i for i,x in enumerate(link) if x.endswith('tilemega-costmodel.cpp.o'))
    link[object_index]=str(OUT/(arm+'.o'))
    record=dict(arm=arm,build=str(build),compile=compile,link=link,
                library_sha256=hashlib.sha256((build/'libtilemega.a').read_bytes()).hexdigest())
    records.append(record)
    (OUT/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
    with (OUT/(arm+'.build.log')).open('x') as log:
        subprocess.run(compile,cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(link,cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
    record['run']=[str(OUT/arm),str(ROOT)]
    record['binary_sha256']=hashlib.sha256((OUT/arm).read_bytes()).hexdigest()
    with (OUT/(arm+'.tsv')).open('x') as output,(OUT/(arm+'.run.log')).open('x') as log:
        result=subprocess.run(record['run'],cwd=ROOT,stdout=output,stderr=log)
    record['exit']=result.returncode
    (OUT/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
    result.check_returncode()
left=(OUT/'baseline.tsv').read_bytes();right=(OUT/'current.tsv').read_bytes()
same=left==right
print(f'SIMULATOR_IDENTITY {"PASS" if same else "FAIL"} rows={len(left.splitlines())} bytes={len(left)} baseline_sha256={hashlib.sha256(left).hexdigest()} current_sha256={hashlib.sha256(right).hexdigest()}')
raise SystemExit(0 if same else 1)
