// SPDX-License-Identifier: BSD-3-Clause
// Derived from MX-5 dense GEMM/head benches; geometry is static as in a plan.
#include <tilemega/Backend/Mx/DecodeHeadV2.h>
#include <tilemega/Codegen/tasks/mx/StageProgram.cuh>
#include <cuda_runtime.h>
#ifndef MX7_B
#define MX7_B 1
#define MX7_N 2048
#define MX7_K 2048
#define MX7_TN 16
#define MX7_WN 1
#define MX7_D 1
#define MX7_PW 1
#define MX7_PART 1
#define MX7_STAGE 0
#endif
using namespace tilemega;
using namespace tilemega::backend::mx;
using Sync=tilemega::codegen::mx::StageSyncV2<512>;
struct BenchShape : MxStaticShapeV2<MX7_N,MX7_K,MX7_B,512> { using Sync=::Sync; };
constexpr MxPart kPart=MX7_PART?MxPart::kTileRange:MxPart::kStreamK;
#if MX7_STAGE == 2
using Body=MxHead<MX7_TN,MX7_WN,MX7_D,MX7_PW>;
using Operands=MxHeadOperandsV2;
#elif MX7_STAGE == 4
using Body=MxGemm<MX7_TN,MX7_WN,MX7_D,MxOpV2::kSwiGLU,true,kPart,MX7_PW>;
using Operands=MxGemmOperandsV2;
#else
using Body=MxGemm<MX7_TN,MX7_WN,MX7_D,MX7_STAGE==1?MxOpV2::kResidual:MxOpV2::kStore,
    MX7_STAGE==0,kPart,MX7_PW>;
using Operands=MxGemmOperandsV2;
#endif
struct BenchArgs {
  MxGemmOperandsV2 gemm;
  MxHeadPartialV2* head_partial;
  unsigned* head_tickets;
  int* tokens;
  unsigned long long *counter,*stamps;
  int* resources;
  int iterations,copies;
};
__device__ __forceinline__ Operands Operand(BenchArgs const& a,int copy) {
  auto p=a.gemm;p.w+=std::size_t(copy)*MX7_N*MX7_K;
#if MX7_STAGE == 2
  return {p,a.head_partial,a.head_tickets,a.tokens,1088,64};
#else
  return p;
#endif
}
__global__ void __launch_bounds__(512) Mx7DenseBench(BenchArgs a) {
  __builtin_assume(gridDim.x==104);
  extern __shared__ __align__(16) char storage[];
#if MX7_STAGE == 2
  int const core=max(32784,Body::SmemBytes(MX7_B));
#else
  int const core=max(32784,MxGemmSmemV2(MX7_B,MX7_TN,MX7_WN).bytes+16);
#endif
  auto* generation=(unsigned*)(storage+core);
  if(Sync::IsRole())*generation=0;
  Sync::CtaSync();
  auto pf=Body::Prefetch(Operand(a,0),BenchShape{});
#pragma unroll 1
  for(int i=0;i<a.iterations;++i) {
    pf=Body::Run(pf,Operand(a,i%a.copies),BenchShape{},storage,[&] {
      return Body::Prefetch(Operand(a,(i+1)%a.copies),BenchShape{});
    });
    tilemega::codegen::mx::StageBarrierV2<Sync>(a.counter,(i+1ull)*104,generation,i+1,[]{},{});
    if(blockIdx.x==0&&Sync::IsRole())a.stamps[i]=maca::GlobalTimerNs();
  }
}
extern "C" int mx7_stage(BenchArgs const* a,void* stream) {
  if(!a||a->copies<1||a->iterations<0)return 4;
  if(!MxGemmLegalV2(MX7_N,MX7_K,MX7_B,MX7_TN,MX7_WN,MX7_D,
      MX7_STAGE==4?MxOpV2::kSwiGLU:MX7_STAGE==1?MxOpV2::kResidual:MxOpV2::kStore,kPart,104))return 4;
#if MX7_STAGE == 2
  int const bytes=max(32784,Body::SmemBytes(MX7_B))+16;
#else
  int const bytes=max(32784,MxGemmSmemV2(MX7_B,MX7_TN,MX7_WN).bytes+16)+16;
#endif
  if(bytes>65536)return 4;
  if(cudaFuncSetAttribute(Mx7DenseBench,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)!=cudaSuccess)return 5;
  cudaFuncAttributes p{};if(cudaFuncGetAttributes(&p,Mx7DenseBench)!=cudaSuccess)return 6;
  int resident=0;
  if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Mx7DenseBench,512,bytes)!=cudaSuccess)return 7;
  if(a->resources){a->resources[0]=p.numRegs;a->resources[1]=p.localSizeBytes;a->resources[2]=p.sharedSizeBytes;a->resources[3]=resident;}
  if(p.localSizeBytes||p.sharedSizeBytes||p.numRegs>255||resident!=1)return 8;
  if(!a->iterations)return 0;
  Mx7DenseBench<<<104,512,bytes,static_cast<cudaStream_t>(stream)>>>(*a);
  return int(cudaGetLastError());
}
extern "C" unsigned mx7_stage_args_size(){return sizeof(BenchArgs);}
