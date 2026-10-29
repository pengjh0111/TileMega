// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/GroupedGemmV2.h>
#include <cuda_runtime.h>
using namespace tilemega;
using namespace tilemega::backend::mx;
using namespace tilemega::codegen;
using Rms=DmEpilogueStep<DmEpilogueKind::kDeferredRMSNorm,DmActivation::kRelu,DmGatePair::kSimpleGate,1,DmRounding::kFP32,DmRounding::kBF16,DmWriteKind::kDense,1>;
using Gate=DmEpilogueStep<DmEpilogueKind::kGatePair,DmActivation::kRelu,DmGatePair::kSwiGLU,16,DmRounding::kBF16,DmRounding::kBF16,DmWriteKind::kDense,1>;
using Chain=MxMoeGateChainV2<Rms,Gate>;
template<int B,int WN,int Depth,int Threads,bool GateUp,int Pw>
__global__ void __launch_bounds__(Threads) MoeGroupedKernel(MxGroupedGemmOperandsV2 operands) {
  unsigned const grid=gridDim.x;__builtin_assume(grid==104);
  extern __shared__ __align__(16) char storage[];
  using Shape=std::conditional_t<GateUp,MxStaticShapeV2<1536,2048,B,Threads>,MxStaticShapeV2<2048,768,B,Threads>>;
  using Complete=std::conditional_t<GateUp,MxNoUnitCompletionV2,MxMoeDownCompletionV2<true>>;
  using Body=MxGroupedGemm<64,WN,Depth,GateUp?MxOpV2::kSwiGLU:MxOpV2::kStore,GateUp,Chain,Complete,Pw>;
  auto pf=Body::Prefetch(operands,Shape{});
  Body::Run(pf,operands,Shape{},storage,[]{return MxNoPrefetchedV2{};});
}
template<int B,int WN,int Depth,int Threads,bool GateUp,int Pw>
int Launch(MxGroupedGemmOperandsV2 p,cudaStream_t stream,int* resources,bool query_only) {
  int const bytes=max(32784,MxGemmSmemV2(B,64,WN).bytes);
  if(bytes>65536)return 4;
  auto kernel=MoeGroupedKernel<B,WN,Depth,Threads,GateUp,Pw>;
  if(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)!=cudaSuccess)return 5;
  cudaFuncAttributes a{};if(cudaFuncGetAttributes(&a,kernel)!=cudaSuccess)return 6;
  int resident=0;
  if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,Threads,bytes)!=cudaSuccess||resident!=1)return 7;
  if(a.localSizeBytes||a.sharedSizeBytes||a.numRegs>(Threads==576?227:255))return 8;
  if(resources){resources[0]=a.numRegs;resources[1]=int(a.localSizeBytes);resources[2]=int(a.sharedSizeBytes);resources[3]=resident;}
  if(query_only)return 0;
  unsigned const stride=GateUp?1536*2048:2048*768;
  if(WN>int((p.units+103)/104))return 4;
  if(!p.units||p.units>(GateUp?24u:32u)*128u||p.expert_stride!=stride)return 4;
  MoeGroupedKernel<B,WN,Depth,Threads,GateUp,Pw><<<104,Threads,bytes,stream>>>(p);
  return int(cudaGetLastError());
}
template<int B,int WN,int Depth,int Threads,bool GateUp,int Pw>
int Select(MxGroupedGemmOperandsV2 p,cudaStream_t stream,int* resources,bool query_only) {
  return Launch<B,WN,Depth,Threads,GateUp,Pw>(p,stream,resources,query_only);
}
extern "C" int mx5_moe_grouped_probe(MxGroupedGemmOperandsV2 p,int batch,int wn,int depth,int threads,int gate_up,int pw,cudaStream_t stream,int* resources,int query_only) {
  if((query_only!=0&&query_only!=1)||(gate_up!=0&&gate_up!=1))return 4;
#define CASE(B,W,D,T,G,P) if(batch==B&&wn==W&&depth==D&&threads==T&&gate_up==G&&pw==P)return Select<B,W,D,T,bool(G),P>(p,stream,resources,query_only)
  CASE(1,1,1,512,1,0);CASE(16,1,1,512,1,0);
  CASE(1,2,2,576,1,0);CASE(16,4,2,576,1,0);
  CASE(16,8,1,576,1,0);CASE(16,8,1,512,1,0);
  CASE(1,1,1,512,0,0);CASE(16,1,1,512,0,0);
  CASE(1,2,2,576,0,2);CASE(16,4,2,576,0,2);
  CASE(16,8,1,576,0,4);CASE(16,8,1,512,0,4);
  return 4;
#undef CASE
}
extern "C" unsigned mx5_moe_grouped_operands_size(){return sizeof(MxGroupedGemmOperandsV2);}
