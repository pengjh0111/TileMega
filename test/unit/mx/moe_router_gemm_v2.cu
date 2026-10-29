// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/RouterGemmV2.h>
#include <cuda_runtime.h>
using namespace tilemega::backend::mx;
template<int B,int TN,int D,int Threads,bool Stream,bool Fp32>
__global__ void __launch_bounds__(Threads) MoeRouterGemmKernel(MxMoeRouterOperandsV2 p) {
  unsigned const grid=gridDim.x;__builtin_assume(grid==104);
  extern __shared__ __align__(16) char storage[];
  using Shape=MxStaticShapeV2<128,2048,B,Threads>;
  using Dispatch=MxMoeDispatchV2<B,2048,768,64,32,Fp32,Threads>;
  using Body=MxRouterGemmV2<TN,D,Stream?MxPart::kStreamK:MxPart::kTileRange,Dispatch,D>;
  auto pf=Body::Prefetch(p,Shape{});
  Body::Run(pf,p,Shape{},storage,[]{return MxNoPrefetchedV2{};});
}
template<int B,int TN,int D,int Threads,bool Stream,bool Fp32>
int Launch(MxMoeRouterOperandsV2 p,cudaStream_t stream,int* resources,bool query_only) {
  int const bytes=max(32784,MxGemmSmemV2(B,TN,1).bytes+16);
  if(bytes>65536)return 4;
  auto kernel=MoeRouterGemmKernel<B,TN,D,Threads,Stream,Fp32>;
  if(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)!=cudaSuccess)return 5;
  cudaFuncAttributes a{};if(cudaFuncGetAttributes(&a,kernel)!=cudaSuccess)return 6;
  int resident=0;
  if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,Threads,bytes)!=cudaSuccess||resident!=1)return 7;
  if(a.localSizeBytes||a.sharedSizeBytes||a.numRegs>(Threads==576?227:255))return 8;
  if(resources){resources[0]=a.numRegs;resources[1]=int(a.localSizeBytes);resources[2]=int(a.sharedSizeBytes);resources[3]=resident;}
  if(query_only)return 0;
  MoeRouterGemmKernel<B,TN,D,Threads,Stream,Fp32><<<104,Threads,bytes,stream>>>(p);
  return int(cudaGetLastError());
}
extern "C" int mx5_moe_router_gemm_probe(MxMoeRouterOperandsV2 p,int batch,int tn,int depth,int threads,int stream_k,int fp32,cudaStream_t stream,int* resources,int query_only) {
  if(query_only!=0&&query_only!=1)return 4;
#define CASE(B,N,D,T,S,F) if(batch==B&&tn==N&&depth==D&&threads==T&&stream_k==S&&fp32==F)return Launch<B,N,D,T,bool(S),bool(F)>(p,stream,resources,query_only)
  CASE(1,16,1,512,0,1);CASE(16,16,1,576,0,1);
  CASE(1,32,2,576,1,1);CASE(16,32,2,512,1,1);
  CASE(1,64,1,512,1,0);CASE(16,64,1,576,1,0);
  CASE(1,128,1,576,0,0);CASE(16,128,1,512,0,0);
  return 4;
#undef CASE
}
extern "C" unsigned mx5_moe_router_gemm_operands_size(){return sizeof(MxMoeRouterOperandsV2);}
