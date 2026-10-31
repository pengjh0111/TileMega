// SPDX-License-Identifier: BSD-3-Clause
// Derived from MX-5 grouped_stage_bench; replay uses captured router logits.
#include <tilemega/Backend/Mx/GroupedGemmV3.h>
#include <tilemega/Backend/Mx/RouteTableV3.h>
#include <cuda_runtime.h>
#ifndef MX6_B
#define MX6_B 1
#define MX6_E 128
#define MX6_H 2048
#define MX6_I 768
#define MX6_TN 64
#define MX6_WN 1
#define MX6_D 1
#define MX6_PW 0
#define MX6_GU 1
#define MX6_COMBINE 0
#endif
using namespace tilemega;
using namespace tilemega::backend::mx;
using namespace tilemega::codegen;
using Sync=mx::StageSyncV3<512>;
using Rms=DmEpilogueStep<DmEpilogueKind::kDeferredRMSNorm,DmActivation::kRelu,
    DmGatePair::kSimpleGate,1,DmRounding::kFP32,DmRounding::kBF16,DmWriteKind::kDense,1>;
using Gate=DmEpilogueStep<DmEpilogueKind::kGatePair,DmActivation::kRelu,
    DmGatePair::kSwiGLU,16,DmRounding::kBF16,DmRounding::kBF16,DmWriteKind::kDense,1>;
constexpr int kN=MX6_GU?2*MX6_I:MX6_H,kK=MX6_GU?MX6_H:MX6_I;
using Body=MxGroupedGemmV3<MX6_TN,MX6_WN,MX6_D,MX6_GU?MxOpV2::kSwiGLU:MxOpV2::kStore,
    bool(MX6_GU),MxMoeGateChainV3<Rms,Gate>,MX6_PW,false,MxCombineModeV3(MX6_COMBINE)>;
using Builder=MxBuildRouteTableV3<MX6_B,MX6_H,MX6_I,MX6_GU?MX6_TN:32,
    MX6_GU?16:MX6_TN,MX6_E,MX6_E==128>;
struct BenchShape : MxStaticShapeV2<kN,kK,MX6_B,512> {using Sync=::Sync;};
struct GroupedBenchArgs {
  Bf16 const *x,*w,*logits;
  Bf16 *h,*partials,*residual,*out;
  unsigned* tickets;
  unsigned long long *counter,*starts,*ends;
  int* resources;
  int iterations,copies;
  float epsilon;
};
__global__ void __launch_bounds__(512) Mx6GroupedBench(GroupedBenchArgs a) {
  __builtin_assume(gridDim.x==104);
  extern __shared__ __align__(16) char storage[];
  int const core=max(max(32784,MxGemmSmemV2(MX6_B,MX6_TN,MX6_WN).bytes),Builder::kSharedBytes);
  auto* table=(MxSharedRouteTableV3<false>*)(storage+core);
  auto* state=(MxSharedCombineStateV3*)(storage+core+sizeof(MxRouteTableV3));
  auto* generation=(unsigned*)(storage+core+sizeof(MxRouteTableV3)+sizeof(MxCombineStateV3));
  MxMoeCombineTileV3<MX6_TN,false,MxCombineModeV3(MX6_COMBINE)>::Initialize(state,MX6_H);
  if(Sync::IsRole())*generation=0;
  Sync::CtaSync();
  unsigned barriers=0;
#pragma unroll 1
  for(int i=0;i<a.iterations;++i) {
    unsigned long long start=0;
    if constexpr(MX6_GU)if(blockIdx.x==0&&Sync::IsRole())start=maca::GlobalTimerNs();
    Builder::Run(a.logits+std::size_t(i%64)*MX6_B*MX6_E,table,storage);
    auto const* weights=a.w+std::size_t(i%a.copies)*MX6_E*kN*kK;
    MxMoeCombineOperandsV3<false> combine{a.partials,table,a.residual,a.out,
        (MxMoeGlobalTicketV2*)a.tickets,state,unsigned(i+1),{}};
    MxGroupedGemmOperandsV3<false> p{MX6_GU?a.x:a.h,weights,MX6_GU?a.h:a.partials,
        table,unsigned(kN*kK),a.epsilon,{},combine,nullptr};
    typename Body::Prefetched pf{};
    if constexpr(!MX6_GU) {
      // RouteTable belongs to gate/up. Prepare it outside the down interval.
      Sync::CtaSync();Sync::Arrive(a.counter);
      pf=Body::Prefetch(p,BenchShape{});
      ++barriers;Sync::Wait(a.counter,std::uint64_t(barriers)*104,generation,barriers,[]{},{});
      if(blockIdx.x==0&&Sync::IsRole())start=maca::GlobalTimerNs();
    }
    Body::Run(pf,p,BenchShape{},storage,[&] {
      Sync::Arrive(a.counter);
      return typename Body::Prefetched{};
    });
    ++barriers;Sync::Wait(a.counter,std::uint64_t(barriers)*104,generation,barriers,[]{},{});
    if(blockIdx.x==0&&Sync::IsRole()) {
      auto const end=maca::GlobalTimerNs();a.starts[i]=start;a.ends[i]=end;
    }
  }
}
extern "C" int mx6_grouped_bench(GroupedBenchArgs const* a,void* stream) {
  if(!a||a->copies<1||a->iterations<0)return 4;
  if(!MxGemmLegalV3((MX6_B*8<MX6_E?MX6_B*8:MX6_E)*kN,kK,MX6_B,
      MX6_TN,MX6_WN,MX6_D,MX6_GU?MxOpV2::kSwiGLU:MxOpV2::kStore,MxPart::kTileRange,true))return 4;
  int const core=max(max(32784,MxGemmSmemV2(MX6_B,MX6_TN,MX6_WN).bytes),Builder::kSharedBytes);
  int const bytes=core+sizeof(MxRouteTableV3)+sizeof(MxCombineStateV3)+16;
  if(bytes>65536)return 4;
  if(cudaFuncSetAttribute(Mx6GroupedBench,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)!=cudaSuccess)return 5;
  cudaFuncAttributes p{};if(cudaFuncGetAttributes(&p,Mx6GroupedBench)!=cudaSuccess)return 6;
  int resident=0;
  if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Mx6GroupedBench,512,bytes)!=cudaSuccess)return 7;
  if(a->resources){a->resources[0]=p.numRegs;a->resources[1]=p.localSizeBytes;a->resources[2]=p.sharedSizeBytes;a->resources[3]=resident;}
  if(p.localSizeBytes||p.sharedSizeBytes||p.numRegs>255||resident!=1)return 8;
  if(!a->iterations)return 0;
  Mx6GroupedBench<<<104,512,bytes,static_cast<cudaStream_t>(stream)>>>(*a);
  return int(cudaGetLastError());
}
extern "C" unsigned mx6_grouped_bench_args_size(){return sizeof(GroupedBenchArgs);}
