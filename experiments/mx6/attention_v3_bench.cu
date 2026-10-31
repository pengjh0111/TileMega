// SPDX-License-Identifier: BSD-3-Clause
// Derived from MX-5 attention_stage_bench.cu; rotation counts addressed KV bytes.
#include <tilemega/Backend/Mx/DecodeAttentionV3.h>
#include <cuda_runtime.h>
#ifndef MX6_B
#define MX6_B 1
#define MX6_HKV 4
#define MX6_DIM 128
#define MX6_Q 8
#define MX6_NORM 1
#define MX6_ROW 0
#define MX6_PREFETCH 0
#define MX6_SPLIT 1
#endif
using namespace tilemega;
using namespace tilemega::backend::mx;
using Sync=tilemega::codegen::mx::StageSyncV3<512>;
using Body=MxAttentionV3<MX6_DIM,MX6_Q,bool(MX6_NORM),MX6_PREFETCH,1088,MX6_ROW,MX6_SPLIT>;
struct BenchShape : MxStaticAttentionShapeV2<MX6_B,MX6_HKV,512> {using Sync=::Sync;};
struct AttentionBenchArgs {
  MxAttentionOperandsV2 attention;
  unsigned long long *counter,*stamps;
  int* resources;
  int iterations,copies;
};
__device__ __forceinline__ MxAttentionOperandsV2 Operand(AttentionBenchArgs const& a,int copy) {
  auto p=(&a.attention)[UniformOpaque(0)];
  std::size_t const offset=std::size_t(copy)*MX6_B*MX6_HKV*1088*MX6_DIM;
  p.k_cache+=offset;p.v_cache+=offset;
  return p;
}
__global__ void __launch_bounds__(512) Mx6AttentionBench(AttentionBenchArgs a) {
  __builtin_assume(gridDim.x==104);
  extern __shared__ __align__(16) char storage[];
  auto* generation=(unsigned*)(storage+max(32784,Body::kSharedBytes));
  if(Sync::IsRole())*generation=0;
  Sync::CtaSync();
  auto pf=Body::Prefetch(Operand(a,0),BenchShape{});
#pragma unroll 1
  for(int i=0;i<a.iterations;++i) {
    pf=Body::Run(pf,Operand(a,i%a.copies),BenchShape{},storage,[&] {
      Sync::Arrive(a.counter);
      return Body::Prefetch(Operand(a,(i+1)%a.copies),BenchShape{});
    });
    Sync::Wait(a.counter,(i+1ull)*104,generation,i+1,[]{},{});
    if(blockIdx.x==0&&Sync::IsRole())a.stamps[i]=maca::GlobalTimerNs();
  }
}
extern "C" int mx6_attention_bench(AttentionBenchArgs const* a,void* stream) {
  if(!a||a->copies<1||a->iterations<0||a->attention.capacity!=1088||a->attention.past<0||a->attention.past>=1088)return 4;
  if(MX6_SPLIT>1&&MX6_B*MX6_HKV*MX6_SPLIT>104)return 4;
  int const bytes=max(32784,Body::kSharedBytes)+16;
  if(bytes>65536)return 4;
  if(cudaFuncSetAttribute(Mx6AttentionBench,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)!=cudaSuccess)return 5;
  cudaFuncAttributes p{};if(cudaFuncGetAttributes(&p,Mx6AttentionBench)!=cudaSuccess)return 6;
  int resident=0;
  if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Mx6AttentionBench,512,bytes)!=cudaSuccess)return 7;
  if(a->resources){a->resources[0]=p.numRegs;a->resources[1]=p.localSizeBytes;a->resources[2]=p.sharedSizeBytes;a->resources[3]=resident;}
  if(p.localSizeBytes||p.sharedSizeBytes||p.numRegs>255||resident!=1)return 8;
  if(!a->iterations)return 0;
  Mx6AttentionBench<<<104,512,bytes,static_cast<cudaStream_t>(stream)>>>(*a);
  return int(cudaGetLastError());
}
extern "C" unsigned mx6_attention_bench_args_size(){return sizeof(AttentionBenchArgs);}
