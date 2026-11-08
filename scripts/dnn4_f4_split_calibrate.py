#!/usr/bin/env python3
"""Measure F4 fragment reductions for the models' legal split divisors."""
import argparse,concurrent.futures,hashlib,json,os,re,shutil,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--plans',type=Path,nargs='+');p.add_argument('--build',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
 if a.execute:
  if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or os.environ.get('TILEMEGA_DEVICE_INDEX')!='1':raise ValueError('split calibration requires reserved GPU1')
  b=json.loads(a.build.read_text());rows=[];sys.path.insert(0,str(ROOT/'scripts'));from dnn1_device import gpu
  for probe in b['probes']:
   if sha(probe['binary'])!=probe['binary_sha256']:raise ValueError('binary changed')
   processes=[]
   for i in range(5):
    start=gpu('1');r=subprocess.run([probe['binary']],capture_output=True,text=True,timeout=120);end=gpu('1');stem=f"{probe['name']}_p{i}";(a.out/(stem+'.stdout')).write_text(r.stdout);(a.out/(stem+'.stderr')).write_text(r.stderr)
    if r.returncode:raise RuntimeError(stem+' failed: '+r.stderr)
    processes.append(dict(measurement=json.loads(r.stdout),gpu=1,start_sm_clock_mhz=start['sm_clock_mhz'],end_sm_clock_mhz=end['sm_clock_mhz']))
   rows.append(dict(probe=probe,processes=processes,passes='5/5',median_ns=statistics.median(p['measurement']['ns'] for p in processes)))
   save(a.out/'result.json',dict(status='partial',evidence='verified',rows=rows,build=str(a.build)));print(probe['name'],rows[-1]['median_ns'],flush=True)
  if any(sha(p)!=v for p,v in b['inputs'].items()):raise ValueError('frozen inputs changed')
  save(a.out/'result.json',dict(status='passed',evidence='verified',rows=rows,build=str(a.build),qualification='provisional_container_pid_namespace'));return
 if not a.plans:raise ValueError('supply plan-derived K extents')
 stages=[s for p in a.plans for s in json.loads(p.read_text())['stage_rows'] if s['body'] in ['conv','gemm']]
 def splits(swap):
  return sorted({s for stage in stages if (stage['mode']=='swapab')==swap for tk in [64,128] for s in range(2,33 if swap else 17) if stage['k']%(tk*s)==0})
 specs=[dict(tm=m,tn=n,slices=s,swap=False) for m in [64,128,256] for n in [32,64,96,128,192,256] if m*n<=32768 for s in splits(False)]
 specs += [dict(tm=m,tn=128,slices=s,swap=True) for m in [8,16] for s in splits(True)]
 src=a.out/'source';src.mkdir();shutil.copytree(ROOT/'include',src/'include');shutil.copyfile(ROOT/'scripts/dnn4_probe_f4_split.cu',src/'probe.cu');frozen={str(p):sha(p) for p in src.rglob('*') if p.is_file()}
 def one(item):
  i,s=item;name=f'split_{s["tm"]}x{s["tn"]}_s{s["slices"]}';binary=a.out/name;log=a.out/(name+'.ptxas.log');cpu=136+2*(i%4)
  cmd=['taskset','-c',f'{cpu}-{cpu+1}','nvcc','-std=c++17','-O3','-gencode','arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','-Xptxas','-v','-I'+str(src/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),f'-DDNN4_SPLIT_TM={s["tm"]}',f'-DDNN4_SPLIT_TN={s["tn"]}',f'-DDNN4_SPLIT_S={s["slices"]}',f'-DDNN4_SPLIT_SWAP={int(s["swap"])}',str(src/'probe.cu'),'-o',str(binary)]
  with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  text=log.read_text();spill=list(map(int,re.findall(r'(\d+) bytes spill (?:stores|loads)',text)))
  if r.returncode or not spill or re.search(r'C750[47]',text):raise RuntimeError('split compile failure: '+str(log))
  sass=a.out/(name+'.sass')
  with sass.open('w') as f:subprocess.run(['cuobjdump','--dump-sass',str(binary)],stdout=f,check=True)
  print('built',name,'spill_bytes',max(spill),flush=True);return dict(**s,name=name,binary=str(binary),binary_sha256=sha(binary),sass_sha256=sha(sass),command=cmd,spill_bytes=max(spill),qualification='standalone_register_cap' if any(spill) else 'zero_spill')
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:probes=list(pool.map(one,enumerate(specs)))
 build=a.out/'build.json';save(build,dict(inputs=frozen,probes=probes,plans={str(p.resolve()):sha(p) for p in a.plans}))
 q=a.out/'queue';q.mkdir();env=dict(CUDA_VISIBLE_DEVICES='1',TILEMEGA_DEVICE_INDEX='1',TILEMEGA_GPU_LOCK='/tmp/tilemega_gpu1.lock',TILEMEGA_NUMA_NODE='0',TILEMEGA_GPU_PID_MODE='namespace_limited',TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',PYTHONPATH=str(ROOT/'python'))
 save(q/'queue_split.json',[dict(name='dnn4-f4-split-cal',gpu=True,timing=True,timeout_s=10800,needs_free_mib=8192,cwd=str(ROOT),env=env,command=['taskset','-c','104-111',str(ROOT/'scripts/dnn4_guard_python.py'),str(Path(__file__).resolve()),'--execute','--build',str(build),'--out','{attempt_dir}/cal'])])
 with (a.out/'dispatch.log').open('w') as f:subprocess.run([str(ROOT/'scripts/dnn4_guard_python.py'),'-m','tilemega.bench.scheduler','--queue-dir',str(q),'--out',str(a.out/'scheduler'),'--policy',str(ROOT/'configs/bench/sm90r2_guard_policy.json'),'--deadline-hours','6'],env=dict(os.environ,PYTHONPATH=str(ROOT/'python')),stdout=f,stderr=subprocess.STDOUT,check=True)
 state=json.loads((a.out/'scheduler/state.json').read_text());print(json.dumps(state))
 if any(r['status']!='done' for r in state.values()):raise RuntimeError('split calibration incomplete')
if __name__=='__main__':main()
