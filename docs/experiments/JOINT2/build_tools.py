#!/usr/bin/env python3
"""Build R6 evidence tools with this machine's configured CMake/Ninja flags."""
import argparse,json,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);b=REPO/'build-portable'
 subprocess.run(['ninja','-C',str(b),'tools/tilemega-compile','tools/tilemega-calibrate'],check=True)
 lines=subprocess.check_output(['ninja','-C',str(b),'-t','commands','tools/tilemega-compile'],text=True).splitlines()
 compile=shlex.split(next(x for x in lines if ' -c ' in x and x.endswith('/tools/tilemega-compile.cpp')))
 link=shlex.split(lines[-1].removeprefix(': && ').removesuffix(' && :'))
 for name,source in [('rebase_generate','REBASE/generate.cpp'),('seqscan','JOINT2/seqscan.cpp'),('parametric','SYMBOLIC/r6_templates.cpp'),('replay','COSTMODEL/replay.cpp'),('fuse_bound','JOINT2/fuse_bound.cpp'),('symbolic_cg','SYMBOLIC/cg_roundtrip.cpp'),('cross_grid','SYMBOLIC/cross_grid.cpp'),('champion_fit','SYMBOLIC/champion_fit.cpp')]:
  obj=str((a.out/(name+'.o')).resolve());exe=str((a.out/name).resolve());cmd=[];i=0
  while i<len(compile):
   x=compile[i]
   if x in ('-MT','-MF'):i+=2;continue
   if x=='-MD':i+=1;continue
   if x=='-o':cmd+=['-o',obj];i+=2;continue
   if x=='-c':cmd+=['-c',str(HERE.parent/source)];i+=2;continue
   cmd.append('-UNDEBUG' if x=='-DNDEBUG' else x);i+=1
  lnk=[obj if x.endswith('tilemega-compile.cpp.o') else exe if x=='tools/tilemega-compile' else x for x in link]
  with (a.out/(name+'.build.log')).open('w') as f:
   for c in (cmd,lnk):subprocess.run(c,cwd=b,stdout=f,stderr=subprocess.STDOUT,check=True)
  (a.out/(name+'.build.json')).write_text(json.dumps(dict(compile=cmd,link=lnk),indent=2)+'\n')
if __name__=='__main__':main()
