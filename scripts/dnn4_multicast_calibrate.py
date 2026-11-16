#!/usr/bin/env python3
"""Compare half-page multicast with unicast in the132-CTA production loop."""
import argparse,hashlib,json,os,re,shutil,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--build',type=Path);p.add_argument('--unit-gate',type=Path);p.add_argument('--execute',action='store_true')
 a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
 if a.execute:
  if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or os.environ.get('TILEMEGA_DEVICE_INDEX')!='1':raise ValueError('calibration requires reserved GPU1')
  b=json.loads(a.build.read_text());rows=[]
  sys.path.insert(0,str(ROOT/'scripts'));from dnn1_device import gpu
  for probe in b['probes']:
   if sha(probe['binary'])!=probe['sha256']:raise ValueError('binary changed')
   for source in ['l2','dram']:
    processes=[]
    for i in range(5):
     start=gpu('1');r=subprocess.run([probe['binary'],source,'132'],capture_output=True,text=True,timeout=120);end=gpu('1')
     stem=f"{probe['name']}_{source}_p{i}";(a.out/(stem+'.stdout')).write_text(r.stdout);(a.out/(stem+'.stderr')).write_text(r.stderr)
     if r.returncode:raise RuntimeError(stem+' failed: '+r.stderr)
     processes.append(dict(measurement=json.loads(r.stdout),gpu=1,start_sm_clock_mhz=start['sm_clock_mhz'],end_sm_clock_mhz=end['sm_clock_mhz']))
    rows.append(dict(probe=probe,source=source,processes=processes,median_ns=statistics.median(r['measurement']['ns_per_k'] for r in processes),passes='5/5'))
    save(a.out/'result.json',dict(status='partial',evidence='verified',rows=rows))
  if any(sha(f)!=v for f,v in b['inputs'].items()):raise ValueError('frozen inputs changed')
  save(a.out/'result.json',dict(status='passed',evidence='verified',rows=rows,build=str(a.build),qualification='provisional_container_pid_namespace'))
  return
 if not a.unit_gate or not json.loads(a.unit_gate.read_text()).get('passed'):raise ValueError('multicast requires a passing P-UNIT/P-SAN cohort')
 src=a.out/'source';src.mkdir();shutil.copytree(ROOT/'include',src/'include');shutil.copyfile(ROOT/'scripts/dnn4_probe_multicast.cu',src/'probe.cu')
 frozen={str(f):sha(f) for f in src.rglob('*') if f.is_file()};probes=[]
 for tn,slots in [(128,6),(256,4)]:
  for mc in [0,1]:
   name=f'pipe_128x{tn}_s{slots}_mc{mc}';binary=a.out/name;log=a.out/(name+'.ptxas.log')
   command=['taskset','-c','136-143','nvcc','-std=c++17','-O3','-gencode','arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','-Xptxas','-v','-I'+str(src/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),'-DDNN4_TM=128',f'-DDNN4_TN={tn}',f'-DDNN4_SLOTS={slots}',f'-DDNN4_MULTICAST={mc}',str(src/'probe.cu'),'-o',str(binary)]
   with log.open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=600)
   text=log.read_text();spills=list(map(int,re.findall(r'(\d+) bytes spill (?:stores|loads)',text)))
   if r.returncode or not spills or any(spills) or re.search(r'C750[47]',text):raise RuntimeError('compile/spill gate failed: '+str(log))
   sass=a.out/(name+'.sass')
   with sass.open('w') as f:subprocess.run(['cuobjdump','--dump-sass',str(binary)],stdout=f,check=True)
   probes.append(dict(name=name,tile_m=128,tile_n=tn,slots=slots,mc=mc,binary=str(binary),sha256=sha(binary),sass_sha256=sha(sass),command=command))
 build=a.out/'build.json';save(build,dict(probes=probes,inputs=frozen,unit_gate=str(a.unit_gate.resolve()),unit_gate_sha256=sha(a.unit_gate)))
 queue=a.out/'queue';queue.mkdir();env=dict(CUDA_VISIBLE_DEVICES='1',TILEMEGA_DEVICE_INDEX='1',TILEMEGA_GPU_LOCK='/tmp/tilemega_gpu1.lock',TILEMEGA_NUMA_NODE='0',TILEMEGA_GPU_PID_MODE='namespace_limited',TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',PYTHONPATH=str(ROOT/'python'))
 save(queue/'queue_mc.json',[dict(name='dnn4-multicast-cal',gpu=True,timing=True,timeout_s=3600,needs_free_mib=8192,cwd=str(ROOT),env=env,command=['taskset','-c','104-111',str(ROOT/'scripts/dnn4_guard_python.py'),str(Path(__file__).resolve()),'--execute','--build',str(build),'--out','{attempt_dir}/probe'])])
 with (a.out/'dispatch.log').open('w') as f:subprocess.run([str(ROOT/'scripts/dnn4_guard_python.py'),'-m','tilemega.bench.scheduler','--queue-dir',str(queue),'--out',str(a.out/'scheduler'),'--policy',str(ROOT/'configs/bench/sm90r2_guard_policy.json'),'--deadline-hours','6'],env=dict(os.environ,PYTHONPATH=str(ROOT/'python')),stdout=f,stderr=subprocess.STDOUT,check=True)
 state=json.loads((a.out/'scheduler/state.json').read_text());print(json.dumps(state))
 if any(r['status']!='done' for r in state.values()):raise RuntimeError('multicast calibration incomplete')
if __name__=='__main__':main()
