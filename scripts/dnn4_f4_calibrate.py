#!/usr/bin/env python3
"""Collect F4 main-loop constants for every legal tile and both TK widths."""
import argparse,concurrent.futures,hashlib,json,os,re,shutil,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TILES=[(m,n) for m in [64,128,256] for n in [32,64,96,128,192,256] if m*n<=32768]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def build(out,bandwidth_only=False):
 src=out/'source';src.mkdir();shutil.copytree(ROOT/'include',src/'include')
 for kind in ['pipe','resident']:shutil.copyfile(ROOT/f'scripts/dnn4_probe_f4_{kind}.cu',src/(kind+'.cu'))
 frozen={str(p):sha(p) for p in src.rglob('*') if p.is_file()};specs=[]
 for tm,tn in TILES:
  for tk in [64,128]:
   specs.append(dict(kind='pipe',tm=tm,tn=tn,tk=tk,slots=2,mode='normal'))
   for mode in ['normal','gather']:specs.append(dict(kind='resident',tm=tm,tn=tn,tk=tk,slots=8,mode=mode))
 for tn in [128,256]:
  for tk in [64,128]:
   spec=dict(kind='pipe',tm=128,tn=tn,tk=tk,slots=min(8,230400//(2*tk*(128+tn))),mode='normal')
   if spec not in specs:specs.append(spec)
 if bandwidth_only:specs=[s for s in specs if s["kind"]=="pipe" and s["tm"]==128 and s["tn"] in [128,256] and s["slots"]==min(8,230400//(2*s["tk"]*(128+s["tn"])))]
 def one(item):
  i,s=item;name=f"{s['kind']}_{s['mode']}_{s['tm']}x{s['tn']}_k{s['tk']}_s{s['slots']}";binary=out/name;log=out/(name+'.ptxas.log');cpu=136+2*(i%4)
  prefix='DNN4' if s['kind']=='pipe' else 'F3_PROBE'
  cmd=['taskset','-c',f'{cpu}-{cpu+1}','nvcc','-std=c++17','-O3','-gencode','arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','-Xptxas','-v','-I'+str(src/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),f'-D{prefix}_TM={s["tm"]}',f'-D{prefix}_TN={s["tn"]}',f'-DDNN4_TK={s["tk"]}',f'-DDNN4_SLOTS={s["slots"]}',f'-DF3_PROBE_GATHER={int(s["mode"]=="gather")}',str(src/(s['kind']+'.cu')),'-o',str(binary)]
  with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  text=log.read_text();spills=list(map(int,re.findall(r'(\d+) bytes spill (?:stores|loads)',text)))
  if r.returncode or not spills or any(spills) or re.search(r'C750[47]',text):raise RuntimeError('probe compile/spill failure: '+str(log))
  sass=out/(name+'.sass')
  with sass.open('w') as f:subprocess.run(['cuobjdump','--dump-sass',str(binary)],stdout=f,check=True)
  result=dict(**s,name=name,binary=str(binary),binary_sha256=sha(binary),sass_sha256=sha(sass),command=cmd,spill_bytes=max(spills),dram_active_counts=([132] if "if(dram && active!=132)" in (src/"pipe.cu").read_text() else [33,66,99,132]));print('built',name,flush=True);return result
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:probes=list(pool.map(one,enumerate(specs)))
 if any(sha(p)!=v for p,v in frozen.items()):raise ValueError('frozen inputs changed')
 save(out/'build.json',dict(evidence='verified',inputs=frozen,probes=probes,nvcc=subprocess.check_output(['nvcc','--version'],text=True)))
def execute(a):
 if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or os.environ.get('TILEMEGA_DEVICE_INDEX')!='1':raise ValueError('F4 calibration requires reserved GPU1')
 b=json.loads(a.build.read_text());rows=json.loads(a.resume_result.read_text())['rows'] if a.resume_result else [];sys.path.insert(0,str(ROOT/'scripts'));from dnn1_device import gpu
 completed={(r['probe']['name'],tuple(r['args'])) for r in rows}
 for probe in b['probes']:
  if sha(probe['binary'])!=probe['binary_sha256']:raise ValueError('probe identity changed')
  specs=[([],probe['mode'])] if probe['kind']=='resident' else [([source,str(n)],source) for source in ['l2','dram'] for n in ([33,66,99,132] if probe['tm']==128 and probe['tn'] in [128,256] and probe['slots']==min(8,230400//(2*probe['tk']*(128+probe['tn']))) else [132])]
  specs=[(args,source) for args,source in specs if source!="dram" or int(args[1]) in probe.get("dram_active_counts",[132])]
  for args,source in specs:
   if (probe['name'],tuple(args)) in completed:continue
   processes=[];stem=probe['name']+'_'+'_'.join(args)
   for i in range(5):
    start=gpu('1');r=subprocess.run([probe['binary'],*args],capture_output=True,text=True,timeout=120);end=gpu('1');(a.out/f'{stem}_p{i}.stdout').write_text(r.stdout);(a.out/f'{stem}_p{i}.stderr').write_text(r.stderr)
    if r.returncode:raise RuntimeError(stem+' failed: '+r.stderr)
    processes.append(dict(measurement=json.loads(r.stdout),gpu=1,start_sm_clock_mhz=start['sm_clock_mhz'],end_sm_clock_mhz=end['sm_clock_mhz']))
   keys=['ns'] if probe['kind']=='resident' else ['ns_per_k','first_ns','per_cta_gbps','total_gbps']
   rows.append(dict(probe=probe,args=args,source=source,processes=processes,passes='5/5',median={k:statistics.median(p['measurement'][k] for p in processes) for k in keys}))
   save(a.out/'result.json',dict(evidence='verified',status='partial',rows=rows,build=str(a.build)))
   print(stem,rows[-1]['median'],flush=True)
 if any(sha(p)!=v for p,v in b['inputs'].items()):raise ValueError('frozen inputs changed')
 save(a.out/'result.json',dict(evidence='verified',status='passed',rows=rows,build=str(a.build),qualification='provisional_container_pid_namespace'))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--build',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--bandwidth-only',action='store_true');p.add_argument('--reuse-build',type=Path);p.add_argument('--resume-result',type=Path);a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
 if a.execute:execute(a);return
 if a.reuse_build:
  shutil.copyfile(a.reuse_build,a.out/'build.json')
 else:build(a.out,a.bandwidth_only)
 q=a.out/'queue';q.mkdir();env=dict(CUDA_VISIBLE_DEVICES='1',TILEMEGA_DEVICE_INDEX='1',TILEMEGA_GPU_LOCK='/tmp/tilemega_gpu1.lock',TILEMEGA_NUMA_NODE='0',TILEMEGA_GPU_PID_MODE='namespace_limited',TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',PYTHONPATH=str(ROOT/'python'))
 command=['taskset','-c','104-111',str(ROOT/'scripts/dnn4_guard_python.py'),str(Path(__file__).resolve()),'--execute','--build',str(a.out/'build.json'),'--out','{attempt_dir}/cal']
 if a.resume_result:command+=['--resume-result',str(a.resume_result.resolve())]
 save(q/'queue_f4.json',[dict(name='dnn4-f4-mainloop-cal',gpu=True,timing=True,timeout_s=10800,needs_free_mib=8192,cwd=str(ROOT),env=env,command=command)])
 with (a.out/'dispatch.log').open('w') as f:subprocess.run([str(ROOT/'scripts/dnn4_guard_python.py'),'-m','tilemega.bench.scheduler','--queue-dir',str(q),'--out',str(a.out/'scheduler'),'--policy',str(ROOT/'configs/bench/sm90r2_guard_policy.json'),'--deadline-hours','6'],env=dict(os.environ,PYTHONPATH=str(ROOT/'python')),stdout=f,stderr=subprocess.STDOUT,check=True)
 state=json.loads((a.out/'scheduler/state.json').read_text());print(json.dumps(state))
 if any(r['status']!='done' for r in state.values()):raise RuntimeError('F4 calibration incomplete')
if __name__=='__main__':main()
