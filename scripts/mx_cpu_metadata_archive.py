#!/usr/bin/env python3
"""Archive closed, never-launched CPU rejection metadata with exact restoration."""
from pathlib import Path
import argparse,gzip,hashlib,io,json,lzma,os,tarfile,time

MARKER='CLOSED_CPU_METADATA_ARCHIVE.json'
def sha(data):return hashlib.sha256(data).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,value):
 with Path(p).open('x')as f:json.dump(value,f,indent=2);f.write('\n')
def eligible(folder):
 p=folder/'CPU_RESOURCE_RESULT.json'
 if not p.exists()or(folder/MARKER).exists()or(folder/'queries').exists()or(folder/'CANDIDATE_RESULT.json').exists():return False
 if not(folder/'LOSSLESS_CPU_SOLID_ARCHIVE.json').exists():return False
 if read(p)['status']!='resource_rejected':return False
 files=[p for p in folder.rglob('*')if p.is_file()]
 return bool(files)and all(not p.is_symlink()and p.stat().st_size<=262144 for p in files)
def pack(folders,repository,archive_root):
 assert folders and all(eligible(p)for p in folders)
 archive_root.mkdir(parents=True,exist_ok=True);stem='metadata_'+str(time.time_ns());target=archive_root/(stem+'.tar.xz');indexpath=archive_root/(stem+'.index.json.gz');rows=[]
 with lzma.open(target,'xb',format=lzma.FORMAT_XZ,check=lzma.CHECK_CRC64,preset=3)as stream:
  with tarfile.open(fileobj=stream,mode='w|')as tar:
   for folder in folders:
    for p in sorted(folder.rglob('*')):
     if not p.is_file():continue
     stat=p.stat();data=p.read_bytes();entry=str(p.relative_to(repository));record=dict(path=str(p),entry=entry,folder=str(folder),sha256=sha(data),bytes=len(data),allocated_bytes=stat.st_blocks*512,mtime_ns=stat.st_mtime_ns,mode=stat.st_mode&0o777)
     info=tarfile.TarInfo(entry);info.size=len(data);info.mode=record['mode'];info.mtime=int(stat.st_mtime);tar.addfile(info,io.BytesIO(data));rows.append(record)
 raw=json.dumps(dict(kind='closed_CPU_metadata_archive',schema_version=1,repository=str(repository),files=rows),sort_keys=True,separators=(',',':')).encode();compressed=gzip.compress(raw,mtime=0);indexpath.write_bytes(compressed);expected={r['entry']:r for r in rows};verified=set()
 with tarfile.open(target,'r|xz')as tar:
  for member in tar:
   assert member.isfile()and member.name in expected
   data=tar.extractfile(member).read();r=expected[member.name];assert len(data)==r['bytes']and sha(data)==r['sha256'];verified.add(member.name)
 assert verified==set(expected)
 # Refuse to remove metadata that changed since the verified archive was built.
 for r in rows:
  p=Path(r['path']);assert p.stat().st_mtime_ns==r['mtime_ns']and p.stat().st_size==r['bytes']and sha(p.read_bytes())==r['sha256']
 receipt=dict(kind='closed_CPU_metadata_archive',schema_version=1,archive=dict(path=str(target),sha256=sha(target.read_bytes())),index=dict(path=str(indexpath),sha256=sha(compressed)),reason='B11 storage; only closed CPU resource rejections, never queried or executed; all original bytes and earlier archive indices retained',files=len(rows),folders=len(folders),raw_bytes=sum(r['bytes']for r in rows),allocated_bytes_before=sum(r['allocated_bytes']for r in rows),archive_bytes=target.stat().st_size+indexpath.stat().st_size,roundtrip_verified=True)
 for folder in folders:save(folder/MARKER,dict(receipt,folder=str(folder)))
 for r in rows:Path(r['path']).unlink()
 save(archive_root/(stem+'.receipt.json'),receipt)
 print(json.dumps(receipt),flush=True);return receipt
def restore(folder,destination=None):
 marker=read(folder/MARKER);a=Path(marker['archive']['path']);i=Path(marker['index']['path']);assert sha(a.read_bytes())==marker['archive']['sha256']and sha(i.read_bytes())==marker['index']['sha256'];index=json.loads(gzip.decompress(i.read_bytes()));records={r['entry']:r for r in index['files']if r['folder']==str(folder)};seen=set()
 assert records
 with tarfile.open(a,'r|xz')as tar:
  for member in tar:
   if member.name not in records:continue
   r=records[member.name];data=tar.extractfile(member).read();assert len(data)==r['bytes']and sha(data)==r['sha256'];relative=Path(r['path']).relative_to(folder);assert '..'not in relative.parts
   p=(destination or folder)/relative;p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists():assert p.read_bytes()==data
   else:
    with p.open('xb')as f:f.write(data)
   p.chmod(r['mode']);os.utime(p,ns=(r['mtime_ns'],r['mtime_ns']));seen.add(member.name)
 assert seen==set(records);print('CPU_METADATA_RESTORE_VERIFIED',str(folder),len(seen),flush=True)
def main():
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
 p=sub.add_parser('pack');p.add_argument('--root',type=Path,required=True);p.add_argument('--repository',type=Path,required=True);p.add_argument('--archive-root',type=Path,required=True);p.add_argument('--batch-size',type=int,default=1000);p.add_argument('--max-folders',type=int,default=100000)
 p=sub.add_parser('restore');p.add_argument('folder',type=Path);p.add_argument('--destination',type=Path)
 args=parser.parse_args()
 if args.action=='restore':restore(args.folder.resolve(),args.destination.resolve()if args.destination else None);return
 repository=args.repository.resolve();root=args.root.resolve();assert root.is_relative_to(repository/'runs')and (root.parent/'CP4_REPORTED.json').exists()
 assert args.batch_size>0 and args.max_folders>0;folders=[];total=0
 for p in sorted(root.glob('*/B*/selection/rank*')):
  if not p.is_dir()or not eligible(p):continue
  folders.append(p)
  if len(folders)==args.batch_size:
   pack(folders,repository,args.archive_root.resolve());total+=len(folders);folders=[]
  if total+len(folders)>=args.max_folders:break
 if folders:pack(folders,repository,args.archive_root.resolve());total+=len(folders)
 print('CLOSED_CPU_METADATA_ARCHIVED_FOLDERS',total,flush=True)
if __name__=='__main__':main()
