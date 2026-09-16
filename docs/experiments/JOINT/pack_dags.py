#!/usr/bin/env python3
"""Lossless raw DAG storage; decode and hash every original byte before replacing."""
import argparse,gzip,hashlib,json,subprocess,tempfile,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--codec',type=Path,required=True);ap.add_argument('--raw',type=Path,default=HERE/'raw');a=ap.parse_args();store=HERE/'dag_store';store.mkdir(exist_ok=True)
 directories=sorted(set(p.parent for pat in ('*/plans/*/task_dag.tsv','*/plans/*/task_dag.tsv.gz','*/plans/*/task_dag.parts','*/seqscan_plans/*/task_dag.tsv','*/seqscan_plans/*/task_dag.tsv.gz','*/seqscan_plans/*/task_dag.parts') for p in a.raw.glob(pat)))
 known={}
 for d in directories:
  meta=d/'process.json'
  if not meta.exists() or json.loads(meta.read_text())['exit_code']:continue
  original=d/'task_dag.tsv';gz=d/'task_dag.tsv.gz';parts=d/'task_dag.parts'
  sources=[original] if original.exists() else [gz] if gz.exists() else sorted(parts.glob('*.gz'))
  if original.exists() and original.stat().st_size<40*2**20:continue
  h=hashlib.sha256();size=0
  with tempfile.TemporaryDirectory(prefix='r5-dag-') as tmp:
   encoded=Path(tmp)/'runs.tsv'
   with encoded.open('wb') as f:
    process=subprocess.Popen([str(a.codec),'encode'],stdin=subprocess.PIPE,stdout=f)
    for src in sources:
     with (gzip.open(src,'rb') if src.suffix=='.gz' else src.open('rb')) as data:
      while b:=data.read(2**20):h.update(b);size+=len(b);process.stdin.write(b)
    process.stdin.close();assert process.wait()==0
   digest=h.hexdigest();dest=store/(digest+'.runs.gz')
   if digest not in known:
    check=hashlib.sha256()
    with encoded.open('rb') as f:
     process=subprocess.Popen([str(a.codec),'decode'],stdin=f,stdout=subprocess.PIPE)
     while b:=process.stdout.read(2**20):check.update(b)
     assert process.wait()==0 and check.hexdigest()==digest
    with encoded.open('rb') as f,gzip.GzipFile(filename=str(dest),mode='wb',mtime=0) as out:shutil.copyfileobj(f,out)
    known[digest]=True
   (d/'task_dag.json').write_text(json.dumps(dict(encoding='sorted-adjacency-runs-v1',original_bytes=size,original_sha256=digest,runs=str(dest.relative_to(HERE)),decode='gzip -dc RUNS | dag_codec decode'),indent=2)+'\n')
   for p in (original,gz):
    if p.exists():p.unlink()
   if parts.exists():shutil.rmtree(parts)
   print(d,size,dest.stat().st_size,flush=True)
if __name__=='__main__':main()
