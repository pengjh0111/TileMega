"""Build source-frozen AR and head exchange tests without launching a GPU."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
frozen=args.out/'frozen';shutil.copytree('include',frozen/'include')
for name in ('tp_allreduce_v2.cu','tp_head_exchange_v2.cu'):
 p=frozen/'test/unit/mx'/name;p.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(Path('test/unit/mx')/name,p)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={str(p.relative_to(frozen)):sha(p)for p in frozen.rglob('*')if p.is_file()}
source=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 files=files,date_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
source['source_digest']=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
(args.out/'SOURCE.json').write_text(json.dumps(source,indent=2)+'\n')
def one(key):
 kind,tp,b=key;out=args.out/f'{kind}_t{tp}_b{b}';out.mkdir()
 src=frozen/'test/unit/mx'/f'tp_{"allreduce"if kind=="ar"else"head_exchange"}_v2.cu'
 binary=out/'test'
 cmd=['/opt/maca/tools/cu-bridge/bin/cucc','-std=c++17','-O3','-I'+str(frozen/'include'),
      '-DTILEMEGA_WATCHDOG=0',f'-DMX7_TP={tp}',f'-DMX7_BATCH={b}',
      '-fno-strict-aliasing','-mllvm','--max-mtreg-number=255',str(src),'-o',str(binary)]
 with(out/'compile.log').open('w')as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 identity=dict(kind=kind,tp=tp,batch=b,command=cmd,returncode=r.returncode,source_digest=source['source_digest'],
               source=str(src),source_sha256=sha(src),binary=str(binary),binary_sha256=sha(binary)if binary.exists()else None)
 (out/'BUILD.json').write_text(json.dumps(identity,indent=2)+'\n');print(kind,tp,b,r.returncode,flush=True)
 return r.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
 assert all(code==0 for code in pool.map(one,[(kind,t,b)for kind in ('ar','head')for t in(2,4)for b in(1,16)]))
