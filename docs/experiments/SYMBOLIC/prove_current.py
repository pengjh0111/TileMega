#!/usr/bin/env python3
"""Prove every integer in [1,128] for a current winner's exact template family.

Singleton certificate pieces avoid expensive multi-parameter elimination.
The template formula is unchanged. These pieces are proofs, not GPU variants.
All 128 integer points are proved; five native-table comparisons are additional.
"""
import argparse,concurrent.futures,csv,hashlib,json,os,re,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];JOINT=HERE.parent/'JOINT2'
def rows(p):return list(csv.DictReader(p.open(),delimiter='\t'))
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--driver',type=Path,required=True);ap.add_argument('--models',nargs='+',default=['gqa2','mha4','real']);ap.add_argument('--seqs',nargs='+',type=int,default=[4,128]);ap.add_argument('--jobs',type=int,default=4);a=ap.parse_args();jobs=[]
 for name,path in json.loads((JOINT/'cells.json').read_text()).items():
  model,seq=name.split('_s')
  if model not in a.models or int(seq) not in a.seqs:continue
  fit=rows(HERE/'bounded_fit'/name/'fit.tsv');families=[r['family'] for r in fit if int(r['different_entries'])==0]
  if not families:print('CURRENT_PROOF',name,'outside tested template families',flush=True);continue
  root=REPO/path;arm=json.loads((root/'choice.json').read_text())['arm'];row=next(r for r in rows(root/'auto.cu.top3.tsv') if r['rank']==arm[3:]);family=row['placement'] if row['placement'] in families else families[0];cg=REPO/row['cg'];grid=int(re.search(r'tilemega.solved_grid = (\d+)',cg.read_text())[1]);out=HERE/'bounded_certificates'/name;out.mkdir(parents=True,exist_ok=True)
  meta=dict(source=str(cg),source_sha256=hashlib.sha256(cg.read_bytes()).hexdigest(),family=family,grid=grid,interval=[1,128],shards=[])
  for begin in range(1,129,16):
   shard=out/f's{begin}_{begin+15}';meta['shards'].append(shard.name);jobs.append((name,cg,shard,family,grid,begin,begin+15))
  (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
 def run(job):
  name,cg,out,family,grid,begin,end=job;out.mkdir(parents=True,exist_ok=True)
  if (out/'run.log').exists():raise RuntimeError('refusing overwrite '+str(out))
  cmd=[str(a.driver.resolve()),str(cg),str(out),family,str(grid),str(begin),str(end)];env=dict(os.environ,SYMBOLIC_PIECE_STEP='1')
  meta=dict(command=cmd,environment=dict(SYMBOLIC_PIECE_STEP='1'),started_ns=time.time_ns())
  with (out/'run.log').open('w') as f:r=subprocess.run(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
  meta['exit_code']=r.returncode;(out/'command.json').write_text(json.dumps(meta,indent=2)+'\n');print('PROOF_SHARD',name,begin,end,'exit',r.returncode,flush=True);return r.returncode
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:codes=list(pool.map(run,jobs))
 if any(codes):raise RuntimeError('some proof shards failed; all independent shards attempted')
if __name__=='__main__':main()
