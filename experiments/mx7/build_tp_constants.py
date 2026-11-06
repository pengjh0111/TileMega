"""Freeze and build the ten resident-grid TP communication comparisons."""
import argparse,concurrent.futures,hashlib,json,pathlib,shutil,subprocess,time
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--tp',type=int,choices=(2,4,8),action='append');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
f=a.out/'frozen';shutil.copytree('include',f/'include');(f/'experiments/mx7').mkdir(parents=True);src=f/'experiments/mx7/tp_constants.cu';shutil.copyfile('experiments/mx7/tp_constants.cu',src)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={str(p.relative_to(f)):sha(p)for p in f.rglob('*')if p.is_file()}
s=dict(files=files,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),date_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
s['source_digest']=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest();(a.out/'SOURCE.json').write_text(json.dumps(s,indent=2)+'\n')
def one(key):
 t,b=key;d=a.out/f'tp{t}_B{b}';d.mkdir();binary=d/'test'
 cmd=['/opt/maca/tools/cu-bridge/bin/cucc','-std=c++17','-O3','-I'+str(f/'include'),'-DTILEMEGA_WATCHDOG=0',f'-DMX7_TP={t}',f'-DMX7_BATCH={b}','-fno-strict-aliasing','-mllvm','--max-mtreg-number=255',str(src),'-o',str(binary)]
 with(d/'compile.log').open('w')as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 (d/'BUILD.json').write_text(json.dumps(dict(tp=t,b=b,command=cmd,returncode=r.returncode,source_digest=s['source_digest'],source=str(src),binary=str(binary),binary_sha256=sha(binary)if binary.exists()else None),indent=2)+'\n');print(t,b,r.returncode,flush=True);return r.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=10)as pool:assert all(x==0 for x in pool.map(one,[(t,b)for t in(a.tp or (2,4))for b in(1,2,4,8,16)]))
