#!/usr/bin/env python3
"""Build an R5 standalone driver with the repository's existing CMake flags."""
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[4]

def build(source,binary):
    source=Path(source).resolve();binary=Path(binary).resolve()
    binary.parent.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(binary.parent).free//2**20
    print(f'DISK NEED_MIB=512 FREE_MIB={free}',flush=True)
    if free<512: raise RuntimeError('disk budget')
    lines=subprocess.check_output(['ninja','-C',str(REPO/'build-portable'),'-t','commands','execution_simulator_test'],text=True).splitlines()
    original=next(l for l in lines if 'test/unit/execution_simulator_test.cpp' in l and ' -c ' in l)
    cmd=shlex.split(original);cmd[cmd.index('-o')+1]=str(binary)+'.o';cmd[cmd.index('-c')+1]=str(source)
    for flag in ('-MF','-MT'):
        if flag in cmd:
            i=cmd.index(flag);del cmd[i:i+2]
    cmd=[x for x in cmd if x not in ('-MD','-DNDEBUG')]
    link=shlex.split(lines[-1].split('&&')[1]);link=[x.replace('CMakeFiles/execution_simulator_test.dir/test/unit/execution_simulator_test.cpp.o',str(binary)+'.o') for x in link]
    link[link.index('-o')+1]=str(binary)
    with Path(str(binary)+'.build.log').open('w') as f:
        for c in (cmd,link): subprocess.run(c,cwd=REPO/'build-portable',stdout=f,stderr=subprocess.STDOUT,check=True)
    Path(str(binary)+'.build.json').write_text(json.dumps(dict(compile=cmd,link=link),indent=2)+'\n')

if __name__=='__main__':build(*sys.argv[1:])
