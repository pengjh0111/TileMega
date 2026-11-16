#!/usr/bin/env python3
"""Apply DT-ST2 only to fitting seed stages with at least two tasks per CTA."""
import argparse,json,os,subprocess
from pathlib import Path
from dnn4_seed import HOST_QUERY,ROOT

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();a.library=a.library.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
 manifest=json.loads(Path(str(a.library)+'.plan.json').read_text());c=manifest['native_forward']
 if c['schema']!=4:raise ValueError('DT-ST2 requires schema4 seed')
 eligible=[i for i,s in enumerate(c['stages']) if s['body'] in ('conv','gemm') and s['mode']!='swapab' and s['tile_m']>=128 and s['tile_n']%64==0]
 source=Path(str(a.library)+'.cu').read_text().split('#include <tilemega/Codegen/tasks/Sm90F3Runtime.cuh>')[0]
 source+=HOST_QUERY+'\ntemplate<class B> void Store(int index){auto l=LayoutOf<B>::Get(B::kSlots,true);std::printf("%d %d %d %d\\n",index,B::kTasks,l.end,int(Sm90F3Budget::Fits(l.end)));}\nint main(){\n'
 source+=''.join(f'Store<Body_{i}>({i});\n' for i in eligible)+'}\n'
 cu=a.out/'layout.cu';cu.write_text(source);binary=a.out/'layout_probe'
 command=['taskset','-c','136-143','nvcc','-std=c++17','-O3','-gencode','arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','-I'+str(ROOT/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),str(cu),'-o',str(binary)]
 with (a.out/'build.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
 text=subprocess.check_output([str(binary)],text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));(a.out/'layout.txt').write_text(text);rows=[]
 for line in text.splitlines():
  i,tasks,bytes_,fits=map(int,line.split());accepted=bool(fits and tasks>=264)
  if accepted:c['stages'][i]['store']='tma'
  rows.append(dict(stage=i,tasks=tasks,payload_bytes=bytes_,fits=bool(fits),store='tma' if accepted else 'reg'))
 (a.out/'store.tmexec.json').write_text(json.dumps(c,indent=2)+'\n')
 (a.out/'result.json').write_text(json.dumps(dict(evidence='verified canonical body/resource function;no timing',source=str(a.library),rows=rows),indent=2)+'\n')
 print(json.dumps(dict(eligible=len(rows),tma=sum(r['store']=='tma' for r in rows))))
if __name__=='__main__':main()
