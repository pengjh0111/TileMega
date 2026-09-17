#!/usr/bin/env python3
"""Exercise every integer point of the five-point solved interval in fresh processes."""
import fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];ROOT=Path(__file__).resolve().parent/'interval_closure'
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'));import measure

def main():
 free=shutil.disk_usage(ROOT).free//2**20;print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
 if free<8192:raise RuntimeError('disk budget')
 spec=dict(source=str(ROOT/'gqa2.cu'),kappa='1',residency='1',placement_macro='0')
 if measure.build(ROOT,'gqa2','interval',spec,'sm_89'):raise RuntimeError('compile failed')
 session=str(time.time_ns());binary=ROOT/'bin/interval'
 for seq in range(1,6):
  fixture=ROOT/'fixtures'/f's{seq}';fixture.mkdir(parents=True,exist_ok=True)
  command=['python3',str(REPO/'docs/experiments/E2E/prepare_e2e.py'),'--vh-raw',str(REPO/'docs/experiments/SEQSCAN/raw/export/gqa2'),'--out',str(fixture),'--seq',str(seq),'--past','3']
  with (fixture/'prepare.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
  folder=ROOT/'correctness'/f's{seq}';folder.mkdir(parents=True,exist_ok=True)
  for i in range(50):
   log=folder/f'r{i}.log'
   if log.exists():raise RuntimeError('refusing overwrite '+str(log))
   env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')};env.update(TILEMEGA_WARMUP='0',TILEMEGA_REPEAT='1')
   with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns();r=subprocess.run([str(binary),str(fixture)],env=env,capture_output=True,text=True,timeout=300)
   log.write_text(r.stdout+r.stderr);log.with_suffix('.json').write_text(json.dumps(dict(command=[str(binary),str(fixture)],seq=seq,round=i,session=session,started_ns=start,elapsed_ns=time.time_ns()-start,exit_code=r.returncode,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))+'\n')
   if r.returncode or 'RESULT status=PASS' not in r.stdout:raise RuntimeError(str(log))
  print('INTERVAL',seq,'50/50',flush=True)
if __name__=='__main__':main()
