#!/usr/bin/env python3
"""Compile the same window audit against both revisions; compare fitted tables."""
import hashlib,json,pathlib,shlex,subprocess,time
E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2]
OUT=E/'wait_window_identity';OUT.mkdir(exist_ok=True)
if (OUT/'commands.json').exists():raise RuntimeError('refuse overwrite')
records=[]
for arm,build in [('baseline',pathlib.Path('/root/r9b_work/baseline_build')),('current',ROOT/'build-portable')]:
    commands=subprocess.check_output(['ninja','-C',str(build),'-t','commands','tilemega-costmodel'],text=True).splitlines()
    compile=shlex.split(next(x for x in commands if ' -c ' in x and 'tools/tilemega-costmodel.cpp' in x))
    compile=compile[:compile.index('-MD')]+['-o',str(OUT/(arm+'.o')),'-c',str(E/'wait_window_identity.cpp')]
    link=shlex.split(commands[-1])[2:-2];link[link.index('-o')+1]=str(OUT/arm)
    link[next(i for i,x in enumerate(link) if x.endswith('tilemega-costmodel.cpp.o'))]=str(OUT/(arm+'.o'))
    record=dict(arm=arm,build=str(build),compile=compile,link=link,
                library_sha256=hashlib.sha256((build/'libtilemega.a').read_bytes()).hexdigest())
    records.append(record);(OUT/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
    with (OUT/(arm+'.build.log')).open('x') as log:
        for cmd in (compile,link):subprocess.run(cmd,cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
    start=time.monotonic()
    with (OUT/(arm+'.tsv')).open('x') as output,(OUT/(arm+'.run.log')).open('x') as log:
        result=subprocess.run([str(OUT/arm)],cwd=ROOT,stdout=output,stderr=log)
    record.update(exit=result.returncode,wall_seconds=time.monotonic()-start)
    (OUT/'commands.json').write_text(json.dumps(records,indent=2)+'\n');result.check_returncode()
left=(OUT/'baseline.tsv').read_bytes();right=(OUT/'current.tsv').read_bytes()
print(f'WAIT_WINDOW_IDENTITY {"PASS" if left==right else "FAIL"} cases={len(left.splitlines())} sha256={hashlib.sha256(right).hexdigest()}')
raise SystemExit(0 if left==right else 1)
