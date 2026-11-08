#!/usr/bin/env python3
"""Compile the prescribed384-thread tile/chain register table; never use a GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tma-store',action='store_true');a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(ROOT/'include',out/'include');rows=[];skipped=[]
    text='''#include <tilemega/Backend/Sm90F3Gemm.cuh>
using namespace tilemega::backend;
using E=cutlass::bfloat16_t;
template<class B> __device__ __forceinline__ void Run(Sm90F3GemmOperands p,
    CUtensorMap const* activation,CUtensorMap const* residual) {
  extern __shared__ __align__(1024) char shared[];
  auto* h=reinterpret_cast<Sm90F3Header*>(shared);
  if(threadIdx.x==0){h->rings[0].Initialize();Sm90F3Async::Base::InitFence();}
  __syncthreads();
  if(threadIdx.x<256 && (threadIdx.x&31)==0)h->rings[0].Prime();
  __syncthreads();
  p.activation=activation;p.residual=residual;Sm90F3Phase phase{};
  auto* arena=reinterpret_cast<E*>(shared+Sm90F3Budget::kArena0);
  auto* side=reinterpret_cast<char*>(arena)+B::kSideOffset;
  if(threadIdx.x>=256){Sm90F3Async::RegDealloc<56>();B::Produce(p,h->rings[0],arena,side,phase,0);}
  else {Sm90F3Async::RegAlloc<224>();B::Consume(p,h->rings[0],arena,side,phase,h,int(threadIdx.x)/128);}
  __syncthreads();if(threadIdx.x==0)h->rings[0].Invalidate();
}
'''
    kinds={
        'plain':(False,0,False,''),
        'bias_act':(False,2,False,'Sm90F3Step<Sm90F3Op::kScale,0>,Sm90F3Step<Sm90F3Op::kBias,1>,Sm90F3Step<Sm90F3Op::kActivation>'),
        'bias_res_act':(True,2,False,'Sm90F3Step<Sm90F3Op::kScale,0>,Sm90F3Step<Sm90F3Op::kBias,1>,Sm90F3Step<Sm90F3Op::kResidual,4294967295u>,Sm90F3Step<Sm90F3Op::kActivation>'),
        'deferred_ln':(False,2,True,'Sm90F3Step<Sm90F3Op::kDeferredLayerNorm,0,tilemega::codegen::DmActivation::kRelu,1>'),
        'deferred_ln_gelu':(False,2,True,'Sm90F3Step<Sm90F3Op::kDeferredLayerNorm,0,tilemega::codegen::DmActivation::kRelu,1>,Sm90F3Step<Sm90F3Op::kActivation,0,tilemega::codegen::DmActivation::kGeluErf>'),
        'residual_ln_rowstats':(True,3,True,'Sm90F3Step<Sm90F3Op::kBias,0>,Sm90F3Step<Sm90F3Op::kResidualLN,1,tilemega::codegen::DmActivation::kRelu,2>')}
    for tm in (64,128,256):
        for tn in (32,64,96,128,192,256):
            for kind,(residual,vectors,stats,steps) in kinds.items():
                name=f'scan_m{tm}_n{tn}_{kind}'
                if a.tma_store and (tm<128 or tn%64):
                    skipped.append(dict(tile_m=tm,tile_n=tn,chain=kind,reason='TMA store needs whole64-column WG tiles'));continue
                if tm*tn>32768 or (kind=='residual_ln_rowstats' and (tn%64 or (tm==64 and tn==192))):
                    skipped.append(dict(tile_m=tm,tile_n=tn,chain=kind,reason='accumulator or whole64-column statistics block'));continue
                index=len(rows);side=f'Sm90F3SideLayout<{tm},{tn},{str(residual).lower()},{vectors},{str(stats).lower()},12>'
                chain=f'Sm90F3Chain<{side}'+(','+steps if steps else '')+'>'
                if kind=='residual_ln_rowstats':chain='Sm90F3WriteRowStats<'+chain+'>'
                text+=f'using B{index}=Sm90F3Gemm<{tm},{tn},2,1,Sm90F3Mode::kNormal,{512 if tm==256 else 128},768,768,768,{chain},Sm90F3NoConv,Sm90F3NoPool,64,{str(a.tma_store).lower()}>;\n'
                text+=f'extern "C" __global__ __launch_bounds__(384,1) void {name}(Sm90F3GemmOperands p,const __grid_constant__ CUtensorMap a,const __grid_constant__ CUtensorMap r){{Run<B{index}>(p,&a,&r);}}\n'
                rows.append(dict(name=name,tile_m=tm,tile_n=tn,chain=kind,store='tma' if a.tma_store else 'reg'))
    (out/'scan.cu').write_text(text);log=out/'ptxas.log';binary=out/'scan.cubin'
    command=['taskset','-c','136-143','nvcc','-std=c++17','-O3','-gencode','arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','--cubin','-Xptxas','-v','-I'+str(out/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),str(out/'scan.cu'),'-o',str(binary)]
    with log.open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=1800)
    if r.returncode or re.search(r'C750[47]',log.read_text()):raise RuntimeError('register scan compilation failed: '+str(log))
    sass=subprocess.check_output(['cuobjdump','--dump-sass',str(binary)],text=True);(out/'scan.sass').write_text(sass)
    code={block.splitlines()[0].strip():block for block in sass.split('Function : ')[1:]}
    sections={}
    for block in log.read_text().split('Compiling entry function ')[1:]:
        name=block.split("'")[1];sections[name]=block
    for row in rows:
        section=sections[row['name']];block=code[row['name']]
        spills=list(map(int,re.findall(r'(\d+) bytes spill (?:stores|loads)',section)))
        row.update(spill_bytes=max(spills),max_consumer_register=max(map(int,re.findall(r'\bR(\d+)\b',block.split('USETMAXREG.TRY_ALLOC.CTAPOOL',1)[1]))),feasible=not any(spills))
    result=dict(evidence='verified',status='compiled',threads=384,producer_registers=56,consumer_registers=224,rows=rows,skipped=skipped,sass_sha256=sha(out/'scan.sass'),command=command,
                inputs={str(f):sha(f) for f in (out/'include').rglob('*') if f.is_file()})
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(cases=len(rows),zero_spill=sum(r['feasible'] for r in rows),failed=[r['name'] for r in rows if not r['feasible']])))
if __name__=='__main__':main()
