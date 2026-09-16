#!/usr/bin/env python3
"""Export fresh models/fixtures and unmaterialized sources on the target host."""
import argparse,json,subprocess,shutil,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();root=a.out.resolve()
 for d in ('export','fixture','src','logs'):(root/d).mkdir(parents=True,exist_ok=True)
 def run(name,cmd):
  log=root/'logs'/(name+'.log')
  if log.exists():raise RuntimeError('fresh output required: '+str(log))
  with log.open('w') as f:r=subprocess.run([str(x) for x in cmd],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
  log.with_suffix('.json').write_text(json.dumps(dict(command=[str(x) for x in cmd],exit_code=r.returncode,time_ns=time.time_ns()))+'\n');r.check_returncode()
 run('gqa_export',['python3',REPO/'docs/experiments/V_H/export_probe.py','--out',root/'export/gqa2','--dtype','bf16','--seq-max','2048','--past-max','512'])
 run('mha_export',['python3',REPO/'docs/experiments/P3_GENERALIZATION/export_second.py','--repo',REPO,'--out',root/'export/mha4','--dtype','bf16','--seq-max','2048','--past-max','512','--fixture-seq','4','--fixture-past','3'])
 for s in (4,128):
  d=root/'export'/f'real_s{s}'
  run(f'real_export_s{s}',['python3',REPO/'docs/experiments/REALMODEL/export_real.py','--repo',REPO,'--out',d,'--layers','4','--hidden','4096','--intermediate','14336','--heads','32','--kv-heads','8','--seq-max','2048','--past-max','512','--fixture-seq',s,'--fixture-past','3'])
  shutil.copytree(d/'fixture',root/'fixture'/f'real_s{s}_p3')
 for m in ('gqa2','mha4','real'):
  program=root/'export'/('real_s4' if m=='real' else m)/'exported_program.pt2'
  run(m+'_bridge',['python3',REPO/'python/tilemega/export_bridge.py',program,'--out',root/'export'/f'{m}.json'])
  run(m+'_source',[REPO/'build-portable/tools/tilemega-compile',root/'export'/f'{m}.json',root/'src'/f'{m}.cu','--variants',REPO/'docs/experiments/OWNERSHIP/plan_structured.json'])
  if m=='real':continue
  for s,p in ((4,3),(128,3),(1,0),(128,512),(2048,0)):
   dest=root/'fixture'/f'{m}_s{s}_p{p}'
   cmd=(['python3',REPO/'docs/experiments/E2E/prepare_e2e.py','--vh-raw',root/'export/gqa2'] if m=='gqa2' else ['python3',REPO/'docs/experiments/P3_GENERALIZATION/prepare_fixture.py','--repo',REPO,'--program',program])
   run(f'{m}_fixture_s{s}_p{p}',cmd+['--out',dest,'--seq',s,'--past',p])
if __name__=='__main__':main()
