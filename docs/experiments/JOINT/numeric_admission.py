#!/usr/bin/env python3
"""Reproduce the real-s4 split sweep that repairs numerical candidate admission.

Run in a NEW search raw root after `search.py screen`. This promotes five
catalog points; it never changes golden tolerances or overwrites earlier logs.
The initial split8 exclusions are raw failures retained in rejected_split8/.
"""
import argparse,concurrent.futures,json,subprocess,time
from pathlib import Path
from search import table,write,REPO,exported
from measure import build,run
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['expand','probe']);ap.add_argument('--cell',type=Path,required=True);ap.add_argument('--driver',type=Path,required=True);ap.add_argument('--arch',default='sm_89');a=ap.parse_args();c=a.cell
 if a.action=='expand':
  extra=[r for r in table(c/'screen.tsv') if (r['m'],r['n'],r['k'],r['stages'],r['kappa'],r['residency'])==('32','128','64','2','1','2') and r['split'] in ('1','2','4','16','32')]
  assert len(extra)==5;write(c/'numeric_expansion.tsv',extra);manifest=table(c/'project_manifest.tsv');seen={r['key'] for r in manifest};write(c/'project_manifest.tsv',manifest+[r for r in extra if r['key'] not in seen])
  def one(r):
   d=c/'plans'/r['key'];d.mkdir(parents=True,exist_ok=True)
   if (d/'process.json').exists():raise RuntimeError('refusing project overwrite')
   cmd=[str(a.driver),str(REPO),str(exported('real')),'real','4',*[r[k] for k in ('m','n','k','stages','split','kappa','residency')],str(d.resolve()),'3'];begin=time.time_ns()
   with (d/'project.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
   (d/'process.json').write_text(json.dumps(dict(command=cmd,exit_code=p.returncode,started_ns=begin,elapsed_ns=time.time_ns()-begin))+'\n');p.check_returncode()
   assert build(c,'real','admit_'+r['key'],dict(r,source=str(d/'eft.cu'),placement_macro='0'),a.arch)==0
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,extra))
 else:
  rows=[];session=str(time.time_ns())
  for i,r in enumerate(table(c/'numeric_expansion.tsv')):
   arm='admit_'+r['key'];folder=c/'numeric_admission'/'correctness'/arm
   try:run(c,'real',4,arm,folder,0,i,session);status='PASS'
   except RuntimeError:status='FAIL'
   rows.append(dict(key=r['key'],status=status,log=str(folder/'r0.log')));write(c/'numeric_admission.tsv',rows);print(arm,status,flush=True)
if __name__=='__main__':main()
