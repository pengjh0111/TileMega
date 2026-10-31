#!/usr/bin/env python3
"""Losslessly archive or restore a closed trace CSV bound to its published SHA256."""
from pathlib import Path
import argparse,gzip,hashlib,json,os,tempfile

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  while block:=f.read(1048576):h.update(block)
 return h.hexdigest()

def paths(path):
 path=Path(path).resolve()
 if path.name!='mx_stage_trace.csv':raise ValueError('only closed mx_stage_trace.csv files are supported')
 return path,Path(str(path)+'.gz'),Path(str(path)+'.lossless_archive.json')

def archive(path,expected):
 path,compressed,index=paths(path)
 if len(expected)!=64 or digest(path)!=expected:raise ValueError('published trace SHA256 differs; source retained')
 before=path.stat();temporary=Path(str(compressed)+'.pending')
 with path.open('rb')as source,gzip.open(temporary,'xb',compresslevel=3)as out:
  while block:=source.read(1048576):out.write(block)
 h=hashlib.sha256();size=0
 with gzip.open(temporary,'rb')as source:
  while block:=source.read(1048576):h.update(block);size+=len(block)
 if h.hexdigest()!=expected or size!=before.st_size:raise ValueError('archive round trip differs; source retained')
 after=path.stat()
 if(before.st_mtime_ns,before.st_size)!=(after.st_mtime_ns,after.st_size)or digest(path)!=expected:
  raise ValueError('trace changed during archival; source retained')
 os.link(temporary,compressed);temporary.unlink()
 record=dict(original_path=str(path),original_sha256=expected,original_bytes=size,original_mtime_ns=before.st_mtime_ns,archive_path=str(compressed),archive_sha256=digest(compressed),archive_bytes=compressed.stat().st_size,representation='gzip',raw_bytes_preserved=True,timing_binary_changed=False)
 with index.open('x')as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 path.unlink();return record

def restore(path,check_only=False):
 path,compressed,index=paths(path);record=json.loads(index.read_text())
 if record['original_path']!=str(path)or record['archive_path']!=str(compressed)or digest(compressed)!=record['archive_sha256']:
  raise ValueError('trace archive binding differs')
 h=hashlib.sha256();size=0;temporary=None
 if not check_only and not path.exists():
  fd,name=tempfile.mkstemp(prefix='.trace_restore_',dir=path.parent);temporary=Path(name);output=os.fdopen(fd,'wb')
 else:output=None
 try:
  with gzip.open(compressed,'rb')as source:
   while block:=source.read(1048576):
    h.update(block);size+=len(block)
    if output:output.write(block)
  if size!=record['original_bytes']or h.hexdigest()!=record['original_sha256']:raise ValueError('restored trace bytes differ')
  if output:output.flush();os.fsync(output.fileno());output.close();output=None
  if path.exists()and digest(path)!=record['original_sha256']:raise ValueError('existing source differs; never overwrite it')
  if temporary:
   os.utime(temporary,ns=(record['original_mtime_ns'],record['original_mtime_ns']));os.link(temporary,path)
  return record
 finally:
  if output:output.close()
  if temporary:temporary.unlink(missing_ok=True)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('archive','restore','check'));parser.add_argument('path',type=Path);parser.add_argument('--expected-sha256');args=parser.parse_args()
 if args.action=='archive':
  if not args.expected_sha256:parser.error('archive requires the already-published raw trace SHA256')
  result=archive(args.path,args.expected_sha256)
 else:result=restore(args.path,check_only=args.action=='check')
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
