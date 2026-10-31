// SPDX-License-Identifier: BSD-3-Clause
// Derived from MX-5 stage/head/router benches; geometry is static as in a plan.
#include <tilemega/Backend/Mx/DecodeHeadV2.h>
#include <tilemega/Backend/Mx/RouterLogitsV3.h>
#include <cuda_runtime.h>
#ifndef MX6_B
#define MX6_B 1
#define MX6_N 2048
#define MX6_K 2048
#define MX6_TN 16
#define MX6_WN 1
#define MX6_D 1
#define MX6_PW 1
#define MX6_PART 1
#define MX6_STAGE 0
#endif
using namespace tilemega;
using namespace tilemega::backend::mx;
using Sync=tilemega::codegen::mx::StageSyncV3<512>;
struct BenchShape : MxStaticShapeV2<MX6_N,MX6_K,MX6_B,512> { using Sync=::Sync; };
constexpr MxPart kPart=MX6_PART?MxPart::kTileRange:MxPart::kStreamK;
#if MX6_STAGE == 2
using Body=MxHead<MX6_TN,MX6_WN,MX6_D,MX6_PW>;
using Operands=MxHeadOperandsV2;
#elif MX6_STAGE == 3
using Body=MxRouterLogitsV3<16,MX6_D,MX6_PW>;
using Operands=MxRouterLogitsOperandsV3;
#else
using Body=MxGemm<MX6_TN,MX6_WN,MX6_D,MX6_STAGE==1?MxOpV2::kResidual:MxOpV2::kStore,
    MX6_STAGE==0,kPart,MX6_PW>;
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
  auto p=a.gemm;p.w+=std::size_t(copy)*MX6_N*MX6_K;
#if MX6_STAGE == 2
  return {p,a.head_partial,a.head_tickets,a.tokens,1088,64};
#elif MX6_STAGE == 3
  return {p.a,p.w,p.residual,p.out,nullptr,nullptr,MX6_K,MX6_N,p.epsilon};
#else
  return p;
#endif
}
__global__ void __launch_bounds__(512) Mx6RegularBench(BenchArgs a) {
  __builtin_assume(gridDim.x==104);
  extern __shared__ __align__(16) char storage[];
#if MX6_STAGE == 2
  int const core=max(32784,Body::SmemBytes(MX6_B));
#else
  int const core=max(32784,MxGemmSmemV2(MX6_B,MX6_TN,MX6_WN).bytes+16);
#endif
  auto* generation=(unsigned*)(storage+core);
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
extern "C" int mx6_stage(BenchArgs const* a,void* stream) {
  if(!a||a->copies<1||a->iterations<0)return 4;
  if(!MxGemmLegalV3(MX6_N,MX6_K,MX6_B,MX6_TN,MX6_WN,MX6_D,
      MX6_STAGE==1?MxOpV2::kResidual:MxOpV2::kStore,kPart,false))return 4;
#if MX6_STAGE == 2
  int const bytes=max(32784,Body::SmemBytes(MX6_B))+16;
#else
  int const bytes=max(32784,MxGemmSmemV2(MX6_B,MX6_TN,MX6_WN).bytes+16)+16;
#endif
  if(bytes>65536)return 4;
  if(cudaFuncSetAttribute(Mx6RegularBench,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)!=cudaSuccess)return 5;
  cudaFuncAttributes p{};if(cudaFuncGetAttributes(&p,Mx6RegularBench)!=cudaSuccess)return 6;
  int resident=0;
  if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Mx6RegularBench,512,bytes)!=cudaSuccess)return 7;
  if(a->resources){a->resources[0]=p.numRegs;a->resources[1]=p.localSizeBytes;a->resources[2]=p.sharedSizeBytes;a->resources[3]=resident;}
  if(p.localSizeBytes||p.sharedSizeBytes||p.numRegs>255||resident!=1)return 8;
  if(!a->iterations)return 0;
  Mx6RegularBench<<<104,512,bytes,static_cast<cudaStream_t>(stream)>>>(*a);
  return int(cudaGetLastError());
}
extern "C" unsigned mx6_stage_args_size(){return sizeof(BenchArgs);}
