#!/usr/bin/env python3
"""Compile an R6 host driver with the repository's configured toolchain.
Usage: build_driver.py SOURCE OUTPUT. Records exact compile and link commands.
"""
import json,shlex,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
def build(source,output):
    source=Path(source).resolve();output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    b=REPO/'build-portable'
    lines=subprocess.check_output(['ninja','-C',str(b),'-t','commands','tools/tilemega-compile'],text=True).splitlines()
    cc=shlex.split(next(x for x in lines if ' -c ' in x and x.endswith('/tools/tilemega-compile.cpp')))
    link=shlex.split(lines[-1].removeprefix(': && ').removesuffix(' && :'))
    obj=str(output)+'.o';cmd=[];i=0
    while i<len(cc):
        x=cc[i]
        if x in ('-MT','-MF'):i+=2;continue
        if x=='-MD':i+=1;continue
        if x in ('-o','-c'):cmd.extend([x,obj if x=='-o' else str(source)]);i+=2;continue
        cmd.append('-UNDEBUG' if x=='-DNDEBUG' else x);i+=1
    link=[obj if x.endswith('tilemega-compile.cpp.o') else str(output) if x=='tools/tilemega-compile' else x for x in link]
    with Path(str(output)+'.build.log').open('w') as f:
        subprocess.run(cmd,cwd=b,stdout=f,stderr=subprocess.STDOUT,check=True)
        subprocess.run(link,cwd=b,stdout=f,stderr=subprocess.STDOUT,check=True)
    Path(str(output)+'.build.json').write_text(json.dumps(dict(compile=cmd,link=link),indent=2)+'\n')
if __name__=='__main__':build(*sys.argv[1:])
