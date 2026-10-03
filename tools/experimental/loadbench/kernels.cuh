// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/PageRing.cuh>
#include <tilemega/Codegen/tasks/EventSync.cuh>
#include <type_traits>
namespace loadbench {
namespace ex=tilemega::codegen::executor;
using Arch=std::conditional_t<std::is_void_v<tilemega::arch::CurrentArch>,
                             tilemega::arch::Sm80,tilemega::arch::CurrentArch>;
using Copy=ex::Async<Arch>;
__device__ inline unsigned long long Clock() {
  unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;
}
__device__ inline unsigned long long Acquire(unsigned long long const* p) {
  unsigned long long v;asm volatile("ld.acquire.gpu.global.u64 %0,[%1];":"=l"(v):"l"(__cvta_generic_to_global(p)):"memory");return v;
}
__device__ inline void Release(unsigned long long* p,unsigned long long n=1) {
  asm volatile("red.release.gpu.global.add.u64 [%0],%1;"::"l"(__cvta_generic_to_global(p)),"l"(n):"memory");
}
__device__ inline void Wait(unsigned long long const* p,unsigned long long n) {while(Acquire(p)<n) {}}
// Copied from Calibration.cu, including rotate and the observable escape.
__global__ __launch_bounds__(256) void StreamReadKernel(
    float4 const* data,std::size_t elements,int passes,int rotate,float* sink) {
  std::size_t stride=std::size_t(gridDim.x)*blockDim.x;
  float4 acc=make_float4(0,0,0,0);
  for(int pass=0;pass<passes;++pass) {
    std::size_t shift=(std::size_t(pass)*rotate*blockDim.x)%elements;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=stride) {
      std::size_t j=i+shift;if(j>=elements)j-=elements;
      float4 value;
      asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
        :"=f"(value.x),"=f"(value.y),"=f"(value.z),"=f"(value.w):"l"(data+j):"memory");
      acc.x+=value.x;acc.y+=value.y;acc.z+=value.z;acc.w+=value.w;
    }
  }
  float sum=acc.x+acc.y+acc.z+acc.w;if(sum==1.0e30f)sink[blockIdx.x]=sum;
}
__global__ void Fill(char* p,std::size_t bytes) {
  for(std::size_t v=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;v<bytes/16;v+=std::size_t(gridDim.x)*blockDim.x)
    reinterpret_cast<uint4*>(p)[v]=make_uint4(1,2,3,4);
}
__device__ inline uint4 ReadCg(void const* p) {
  uint4 v;asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];"
    :"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");return v;
}
__device__ inline uint4 ReadNc(void const* p) {
  uint4 v;asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4];"
    :"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");return v;
}
// Shape mapping is shared by the two layouts, isolating address locality.
__device__ inline std::size_t ShapeOffset(std::size_t vector,int tn,int tk,int k,bool row) {
  if(!tn || !row)return vector*16;
  int stage_vectors=tn*tk/8;
  auto stage=vector/stage_vectors;auto inner=vector%stage_vectors;
  int kt=k/tk;auto nt=stage/kt;auto ki=stage%kt;
  return ((nt*tn+inner/(tk/8))*k+ki*tk+(inner%(tk/8))*8)*2;
}
template<int ILP>
__global__ void NcStream(char const* src,std::size_t vectors,int passes,unsigned* sink,
                        int tn=0,int tk=0,int k=0,bool row=false) {
  unsigned acc[ILP]={};auto stride=std::size_t(gridDim.x)*blockDim.x;
  for(int pass=0;pass<passes;++pass)
    for(auto base=(std::size_t(blockIdx.x)*blockDim.x+threadIdx.x);base<vectors;base+=stride*ILP) {
      #pragma unroll
      for(int i=0;i<ILP;++i)if(base+i*stride<vectors) {
        auto v=ReadNc(src+ShapeOffset(base+i*stride,tn,tk,k,row));acc[i]^=v.x^v.y^v.z^v.w;
      }
    }
  unsigned value=0;
  #pragma unroll
  for(int i=0;i<ILP;++i)value^=acc[i];
  if(threadIdx.x==0)sink[blockIdx.x]=value;
}
// Policy 0=cg, 1=128B prefetch, 2=256B prefetch, 3=evict_first.
template<int Policy>
__device__ inline void Cp(void* dst,void const* src,unsigned valid,std::uint64_t policy) {
  if constexpr(Policy==0)Copy::Copy16Bytes(dst,src,valid);
  else if constexpr(Policy==1)
    asm volatile("cp.async.cg.shared.global.L2::128B [%0],[%1],16,%2;"::"r"(Copy::Shared(dst)),"l"(src),"r"(valid):"memory");
  else if constexpr(Policy==2)
    asm volatile("cp.async.cg.shared.global.L2::256B [%0],[%1],16,%2;"::"r"(Copy::Shared(dst)),"l"(src),"r"(valid):"memory");
  else {
    if(valid)Copy::Copy16Hint(dst,src,policy);
    else Copy::Copy16Bytes(dst,src,0);
  }
}
template<int Depth,int Policy>
__global__ void CpStream(char const* src,std::size_t vectors,int passes) {
  extern __shared__ __align__(16) char smem[];
  auto stride=std::size_t(gridDim.x)*blockDim.x;
  auto rounds=(vectors+stride-1)/stride;
  auto policy=Copy::EvictFirst();
  for(int pass=0;pass<passes;++pass) {
    for(int r=0;r<Depth-1;++r) {
      auto v=std::size_t(r)*stride+std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
      if(v<vectors)Cp<Policy>(smem+(r*blockDim.x+threadIdx.x)*16,src+v*16,16,policy);
      asm volatile("cp.async.commit_group;":::"memory");
    }
    for(std::size_t r=0;r<rounds;++r) {
      asm volatile("cp.async.wait_group %0;"::"n"(Depth-2):"memory");
      auto next=r+Depth-1;auto v=next*stride+std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
      if(v<vectors)Cp<Policy>(smem+((next%Depth)*blockDim.x+threadIdx.x)*16,src+v*16,16,policy);
      asm volatile("cp.async.commit_group;":::"memory");
    }
    asm volatile("cp.async.wait_group 0;":::"memory");
  }
}
template<int TN,int TK,int Stages,int Method>
__global__ void GemvStream(char const* src,std::size_t tiles,int k,bool row,unsigned* sink) {
  extern __shared__ __align__(16) char smem[];
  constexpr int bytes=TN*TK*2;constexpr int vectors=bytes/16;
  unsigned acc=0;auto policy=Copy::EvictFirst();int kt=k/TK;
  for(std::size_t tile=blockIdx.x;tile<tiles;tile+=gridDim.x) {
    for(int r=0;r<kt+Stages-1;++r) {
      int slot=r%Stages;
      if(r>=Stages-1) {
        if constexpr(Method>=2)asm volatile("cp.async.wait_group %0;"::"n"(Stages-2):"memory");
        __syncthreads();
      }
      if(r<kt) {
        #pragma unroll
        for(int v=threadIdx.x;v<vectors;v+=128) {
          auto index=(tile*kt+r)*vectors+v;auto offset=ShapeOffset(index,TN,TK,k,row);
          if constexpr(Method==0) {auto a=ReadCg(src+offset);acc^=a.x^a.y^a.z^a.w;}
          else if constexpr(Method==1) {auto a=ReadNc(src+offset);acc^=a.x^a.y^a.z^a.w;}
          else Cp<Method-2>(smem+slot*bytes+v*16,src+offset,16,policy);
        }
      }
      if constexpr(Method>=2)asm volatile("cp.async.commit_group;":::"memory");
    }
    if constexpr(Method>=2)asm volatile("cp.async.wait_group 0;":::"memory");
  }
  if(threadIdx.x==0)sink[blockIdx.x]=acc;
}
// Exact original loader anchor remains separate from wider-loader probes.
__global__ void LoaderAnchor(char const* source,int pages_per_cta) {
  using Ring=ex::PageRing<16384,5,Arch,false>;
  extern __shared__ __align__(1024) char storage[];
  Ring ring{reinterpret_cast<Ring::Slot*>(storage),storage+1024};
  ring.Initialize();std::uint64_t sequence=0;
  for(int page=0;page<pages_per_cta;++page,++sequence) {
    if(ex::IsCompute()) {ring.AwaitFull(sequence);ring.Release(sequence);}
    else {
      ring.AcquireEmpty(sequence);auto tile=std::size_t(blockIdx.x)*pages_per_cta+page;
      auto* dst=ring.Page(sequence);
      for(int v=ex::LoaderLane();v<16384/16;v+=32)Copy::Copy16(dst+v*16,source+tile*16384+v*16);
      ring.PublishCopies(sequence);
    }
  }
}
__global__ void PageLoader(char const* src,int per_cta,int page_bytes,int pages,
                           int loader_warps,bool consume,bool bulk,unsigned* sink) {
  extern __shared__ __align__(1024) char memory[];
  struct Slot {std::uint64_t full,empty,generation;};
  auto* slots=reinterpret_cast<Slot*>(memory);auto* data=memory+1024;
  int lt=loader_warps*32,tid=threadIdx.x;bool loader=tid<lt;unsigned acc=0;
  if(tid==0)for(int s=0;s<pages;++s) {
    Copy::Init(&slots[s].full,lt);Copy::Init(&slots[s].empty,128);slots[s].generation=~std::uint64_t(0);
  }
  if(tid==0)Copy::InitFence();__syncthreads();
  if(!loader)for(int s=0;s<pages;++s)Copy::Arrive(&slots[s].empty);
  for(int i=0;i<per_cta;++i) {
    auto& slot=slots[i%pages];int phase=(i/pages)&1;auto* dst=data+(i%pages)*page_bytes;
    if(loader) {
      Copy::Wait(&slot.empty,phase);
      if(tid==0)*reinterpret_cast<volatile std::uint64_t*>(&slot.generation)=i;
      auto* from=src+(std::size_t(blockIdx.x)*per_cta+i)*page_bytes;
      if(bulk) {
        if(tid==0) {Copy::ExpectTx(&slot.full,page_bytes);Copy::Bulk(dst,from,page_bytes,&slot.full);}
        else Copy::Arrive(&slot.full);
      }else {
        for(int v=tid;v<page_bytes/16;v+=lt)Copy::Copy16(dst+v*16,from+v*16);
        Copy::CompleteCopies(&slot.full);
      }
    }else {
      while(*reinterpret_cast<volatile std::uint64_t*>(&slot.generation)!=std::uint64_t(i)) {}
      Copy::Wait(&slot.full,phase);
      if(consume)for(int v=tid-lt;v<page_bytes/4;v+=128)acc^=reinterpret_cast<volatile unsigned*>(dst)[v];
      Copy::Arrive(&slot.empty);
    }
  }
  if(!loader && tid==lt)sink[blockIdx.x]=acc;
}
__device__ inline void ReadBlock(char const* src,std::size_t bytes,unsigned& acc) {
  for(std::size_t v=threadIdx.x;v<bytes/16;v+=blockDim.x) {
    auto a=ReadCg(src+v*16);acc^=a.x^a.y^a.z^a.w;
  }
}
// Foreground synchronization and background reads share one fully resident grid.
__global__ void Hop(char const* src,std::size_t bytes,bool background,int mode,int kappa,
                    unsigned long long* counters,unsigned long long* times,unsigned* sink) {
  int foreground=background?gridDim.x/2:gridDim.x;
  unsigned acc=0;
  if(int(blockIdx.x)>=foreground) {
    do {ReadBlock(src+std::size_t(blockIdx.x)*bytes,bytes,acc);}while(Acquire(counters)==0);
    if(threadIdx.x==0)sink[blockIdx.x]=acc;return;
  }
  ReadBlock(src+std::size_t(blockIdx.x)*bytes,bytes,acc);__syncthreads();
  if(threadIdx.x==0)times[blockIdx.x*2]=Clock();
  if(mode==0) {
    if(threadIdx.x==0){Release(counters+1);Wait(counters+1,foreground);}
  }else {
    if(threadIdx.x==0)Release(counters+1+blockIdx.x/kappa);
    for(int g=threadIdx.x;g<(foreground+kappa-1)/kappa;g+=blockDim.x)
      Wait(counters+1+g,min(kappa,foreground-g*kappa));
  }
  __syncthreads();
  if(threadIdx.x==0) {times[blockIdx.x*2+1]=Clock();Release(counters);sink[blockIdx.x]=acc;}
}
__global__ void ClusterHop(char const* src,std::size_t bytes,int cluster_size,
                          unsigned long long* counters,unsigned long long* times,unsigned* sink) {
  if constexpr(Copy::Caps::kCluster) {
    unsigned acc=0;
    ReadBlock(src+std::size_t(blockIdx.x)*bytes,bytes,acc);__syncthreads();
    if(threadIdx.x==0)times[blockIdx.x*2]=Clock();
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    if(threadIdx.x==0 && blockIdx.x%cluster_size==0)Release(counters);
    if(threadIdx.x==0)Wait(counters,gridDim.x/cluster_size);
    __syncthreads();
    if(threadIdx.x==0){times[blockIdx.x*2+1]=Clock();sink[blockIdx.x]=acc;}
  }
}
__global__ void StepStream(char const* src,std::size_t bytes,unsigned* sink,
                          unsigned long long* counter,int steps,bool pdl,int trigger,bool early) {
  if constexpr(Copy::Caps::kPdl)if(pdl) {
    if(early)for(int line=threadIdx.x;line<1024*1024/128;line+=blockDim.x)
      asm volatile("prefetch.global.L2 [%0];"::"l"(src+line*128):"memory");
    Copy::WaitPreviousGrid();if(trigger==1)Copy::LaunchDependents();
  }
  unsigned acc=0;
  for(int step=0;step<steps;++step) {
    std::size_t first=std::size_t(blockIdx.x)*bytes/gridDim.x;
    std::size_t end=std::size_t(blockIdx.x+1)*bytes/gridDim.x;
    first=(first/16)*16;end=(end/16)*16;
    ReadBlock(src+first,end-first,acc);__syncthreads();
    if(steps>1) {
      if(threadIdx.x==0){Release(counter);Wait(counter,std::size_t(step+1)*gridDim.x);}
      __syncthreads();
    }
  }
  if(threadIdx.x==0)sink[blockIdx.x]=acc;
  if constexpr(Copy::Caps::kPdl)if(pdl && trigger==0)Copy::LaunchDependents();
}
} // namespace loadbench
