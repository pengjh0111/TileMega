#!/usr/bin/env python3
"""Recheck the worst BERT B1 constant with its actual96-task attention grid."""
import argparse,json,os,re,shutil,subprocess,sys
from pathlib import Path
from dnn4_f4_tasks_calibrate import ROOT,sha,save

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 src=out/'source';src.mkdir();shutil.copytree(ROOT/'include',src/'include');shutil.copyfile(ROOT/'scripts/dnn4_probe_f4_tasks.cu',src/'probe.cu')
 inputs={str(p):sha(p) for p in src.rglob('*') if p.is_file()};binary=out/'attention_rows16_b1';log=out/'ptxas.log'
 cmd=['taskset','-c','136-143','nvcc','-std=c++17','-O3','-gencode','arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','-Xptxas','-v','-I'+str(src/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),'-DDNN4_TASK_KIND=6','-DDNN4_ATTN_ROWS=16','-DDNN4_TASK_BATCH=1',str(src/'probe.cu'),'-o',str(binary)]
 with log.open('w') as f:r=subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,timeout=600)
 text=log.read_text();spills=list(map(int,re.findall(r'(\d+) bytes spill (?:stores|loads)',text)))
 if r.returncode or not spills or any(spills) or re.search(r'C750[47]',text):raise RuntimeError('attention recheck compile gate failed')
 sass=out/'probe.sass'
 with sass.open('w') as f:subprocess.run(['cuobjdump','--dump-sass',str(binary)],stdout=f,check=True)
 build=out/'build.json';save(build,dict(inputs=inputs,probes=[dict(name='attention_rows16_b1',kind=6,rows=16,batch=1,binary=str(binary),binary_sha256=sha(binary),sass_sha256=sha(sass),command=cmd,spill_bytes=0)]))
 queue=out/'queue';queue.mkdir();environment=dict(CUDA_VISIBLE_DEVICES='1',TILEMEGA_DEVICE_INDEX='1',TILEMEGA_GPU_LOCK='/tmp/tilemega_gpu1.lock',TILEMEGA_NUMA_NODE='0',TILEMEGA_GPU_PID_MODE='namespace_limited',TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',PYTHONPATH=str(ROOT/'python'))
 save(queue/'queue_attention_recheck.json',[dict(name='dnn4-attention-constant-recheck',gpu=True,timing=True,timeout_s=1800,needs_free_mib=4096,cwd=str(ROOT),env=environment,command=['taskset','-c','104-111',str(ROOT/'scripts/dnn4_guard_python.py'),str(ROOT/'scripts/dnn4_f4_tasks_calibrate.py'),'--execute','--build',str(build),'--out','{attempt_dir}/cal'])])
 with (out/'dispatch.log').open('w') as f:subprocess.run([str(ROOT/'scripts/dnn4_guard_python.py'),'-m','tilemega.bench.scheduler','--queue-dir',str(queue),'--out',str(out/'scheduler'),'--policy',str(ROOT/'configs/bench/sm90r2_guard_policy.json'),'--deadline-hours','2'],env=dict(os.environ,PYTHONPATH=str(ROOT/'python')),stdout=f,stderr=subprocess.STDOUT,check=True)
 state=json.loads((out/'scheduler/state.json').read_text());save(out/'result.json',dict(evidence='verified',status=state,reason='P-MODEL worst BERT B1 constant;actual batch1 rather than16 at Rows16'))
 if any(r['status']!='done' for r in state.values()):raise RuntimeError('attention recheck incomplete')
if __name__=='__main__':main()
