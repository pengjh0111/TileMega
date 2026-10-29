// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/MoeDispatchV2.h>
#include <cuda_runtime.h>
using namespace tilemega::backend::mx;
template<int B,bool Fp32,int Threads>
__global__ void __launch_bounds__(Threads) MoeDispatchKernel(MxMoeDispatchOperands operands) {
  extern __shared__ __align__(16) char storage[];
  MxMoeDispatchV2<B,2048,768,64,32,Fp32,Threads>::Run(operands,storage);
}
template<int B,bool Fp32,int Threads>
int Launch(MxMoeDispatchOperands p,cudaStream_t s,int* resources,bool query_only) {
  constexpr int body=MxMoeDispatchV2<B,2048,768,64,32,Fp32,Threads>::kSharedBytes;
  constexpr int bytes=body>32784?body:32784;
  if(cudaFuncSetAttribute(MoeDispatchKernel<B,Fp32,Threads>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)!=cudaSuccess)return 5;
  cudaFuncAttributes a{};
  if(cudaFuncGetAttributes(&a,MoeDispatchKernel<B,Fp32,Threads>)!=cudaSuccess)return 6;
  int resident=0;
  if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,MoeDispatchKernel<B,Fp32,Threads>,Threads,bytes)!=cudaSuccess||resident!=1)return 7;
  if(a.localSizeBytes||a.sharedSizeBytes||a.numRegs>(Threads==576?227:255))return 8;
  if(resources){resources[0]=a.numRegs;resources[1]=int(a.localSizeBytes);resources[2]=int(a.sharedSizeBytes);resources[3]=resident;}
  if(query_only)return 0;
  MoeDispatchKernel<B,Fp32,Threads><<<1,Threads,bytes,s>>>(p);
  return int(cudaGetLastError());
}
extern "C" int mx5_moe_dispatch_probe(MxMoeDispatchOperands p,int batch,int fp32,int threads,cudaStream_t stream,int* resources,int query_only) {
  if((threads!=512&&threads!=576)||(fp32!=0&&fp32!=1)||(query_only!=0&&query_only!=1))return 4;
#define CASE(B) case B:if(threads==512){if(fp32)return Launch<B,true,512>(p,stream,resources,query_only);return Launch<B,false,512>(p,stream,resources,query_only);}else{if(fp32)return Launch<B,true,576>(p,stream,resources,query_only);return Launch<B,false,576>(p,stream,resources,query_only);}
  switch(batch){CASE(1);CASE(2);CASE(4);CASE(8);CASE(16);default:return 4;}
#undef CASE
}
