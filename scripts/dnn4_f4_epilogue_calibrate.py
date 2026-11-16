#!/usr/bin/env python3
"""Calibrate all admitted F4 epilogue tile/chain/store combinations in situ."""
import argparse,hashlib,json,os,re,shutil,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CLASSES=['plain','bias_act','bias_res_act','deferred_ln','deferred_ln_gelu','residual_ln_rowstats']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--build',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
 if a.execute:
  if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or os.environ.get('TILEMEGA_DEVICE_INDEX')!='1':raise ValueError('epilogue calibration requires reserved GPU1')
  b=json.loads(a.build.read_text());rows=[];sys.path.insert(0,str(ROOT/'scripts'));from dnn1_device import gpu
  if sha(b['binary'])!=b['binary_sha256']:raise ValueError('binary changed')
  for case in b['cases']:
   processes=[]
   for i in range(5):
    start=gpu('1');r=subprocess.run([b['binary'],str(case['index'])],capture_output=True,text=True,timeout=120);end=gpu('1');stem=f"case{case['index']}_p{i}";(a.out/(stem+'.stdout')).write_text(r.stdout);(a.out/(stem+'.stderr')).write_text(r.stderr)
    if r.returncode:raise RuntimeError(stem+' failed: '+r.stderr)
    processes.append(dict(measurement=json.loads(r.stdout),gpu=1,start_sm_clock_mhz=start['sm_clock_mhz'],end_sm_clock_mhz=end['sm_clock_mhz']))
   rows.append(dict(case=case,processes=processes,passes='5/5',median_ns=statistics.median(p['measurement']['ns'] for p in processes)))
   save(a.out/'result.json',dict(status='partial',evidence='verified',rows=rows,build=str(a.build)))
   print(case,rows[-1]['median_ns'],flush=True)
  if any(sha(p)!=v for p,v in b['inputs'].items()):raise ValueError('frozen inputs changed')
  save(a.out/'result.json',dict(status='passed',evidence='verified',rows=rows,build=str(a.build),sass_sha256=b['sass_sha256'],qualification='provisional_container_pid_namespace'));return
 src=a.out/'source';src.mkdir();shutil.copytree(ROOT/'include',src/'include');shutil.copyfile(ROOT/'scripts/dnn4_probe_f4_epilogue.cu',src/'probe.cu');cases=[];skipped=[]
 for tm in [64,128,256]:
  for tn in [32,64,96,128,192,256]:
   for kind,name in enumerate(CLASSES):
    if tm*tn>32768 or (kind==5 and (tn%64 or (tm==64 and tn==192) or (tm==256 and tn==128))):continue
    for store in [False,True] if tm>=128 and tn%64==0 else [False]:
     cases.append(dict(index=len(cases),tile_m=tm,tile_n=tn,chain=name,kind=kind,store='tma' if store else 'reg'))
 include=src/'dnn4_f4_epilogue_cases.inc';include.write_text(''.join(f'case {c["index"]}:return Run<{c["tile_m"]},{c["tile_n"]},{c["kind"]},{str(c["store"]=="tma").lower()}>();\n' for c in cases))
 frozen={str(p):sha(p) for p in src.rglob('*') if p.is_file()};binary=a.out/'probe';log=a.out/'ptxas.log'
 command=['taskset','-c','136-143','nvcc','-std=c++17','-O3','-gencode','arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','-Xptxas','-v','-I'+str(src/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),'-I'+str(src),str(src/'probe.cu'),'-o',str(binary)]
 with log.open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=1800,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 text=log.read_text();spills=list(map(int,re.findall(r'(\d+) bytes spill (?:stores|loads)',text)))
 if r.returncode or not spills or any(spills) or re.search(r'C750[47]',text):raise RuntimeError('epilogue compile/spill failure: '+str(log))
 sass=a.out/'probe.sass'
 with sass.open('w') as f:subprocess.run(['cuobjdump','--dump-sass',str(binary)],stdout=f,check=True)
 build=a.out/'build.json';save(build,dict(binary=str(binary),binary_sha256=sha(binary),sass_sha256=sha(sass),inputs=frozen,cases=cases,command=command))
 q=a.out/'queue';q.mkdir();env=dict(CUDA_VISIBLE_DEVICES='1',TILEMEGA_DEVICE_INDEX='1',TILEMEGA_GPU_LOCK='/tmp/tilemega_gpu1.lock',TILEMEGA_NUMA_NODE='0',TILEMEGA_GPU_PID_MODE='namespace_limited',TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',PYTHONPATH=str(ROOT/'python'))
 save(q/'queue_epi.json',[dict(name='dnn4-f4-epilogue-cal',gpu=True,timing=True,timeout_s=10800,needs_free_mib=8192,cwd=str(ROOT),env=env,command=['taskset','-c','104-111',str(ROOT/'scripts/dnn4_guard_python.py'),str(Path(__file__).resolve()),'--execute','--build',str(build),'--out','{attempt_dir}/cal'])])
 with (a.out/'dispatch.log').open('w') as f:subprocess.run([str(ROOT/'scripts/dnn4_guard_python.py'),'-m','tilemega.bench.scheduler','--queue-dir',str(q),'--out',str(a.out/'scheduler'),'--policy',str(ROOT/'configs/bench/sm90r2_guard_policy.json'),'--deadline-hours','6'],env=dict(os.environ,PYTHONPATH=str(ROOT/'python')),stdout=f,stderr=subprocess.STDOUT,check=True)
 state=json.loads((a.out/'scheduler/state.json').read_text());print(json.dumps(state))
 if any(r['status']!='done' for r in state.values()):raise RuntimeError('epilogue calibration incomplete')
if __name__=='__main__':main()
