// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/MoeCombineV2.h>
#include <cuda_runtime.h>
using namespace tilemega::backend::mx;
template<int B,int TN,bool Fp32,int Threads>
__global__ void __launch_bounds__(Threads) MoeCombineKernel(MxMoeCombineOperandsV2 p,unsigned active) {
  extern __shared__ __align__(16) char storage[];
  MxMoeCombineV2<B,2048,TN,Fp32,Threads>::Arrive(p,0,active,reinterpret_cast<unsigned*>(storage));
}
template<int B,int TN,bool Fp32,int Threads>
int Launch(MxMoeCombineOperandsV2 p,unsigned active,cudaStream_t stream,int* resources,bool query_only) {
  constexpr int bytes=32784;
  if(cudaFuncSetAttribute(MoeCombineKernel<B,TN,Fp32,Threads>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)!=cudaSuccess)return 5;
  cudaFuncAttributes a{};
  if(cudaFuncGetAttributes(&a,MoeCombineKernel<B,TN,Fp32,Threads>)!=cudaSuccess)return 6;
  int resident=0;
  if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,MoeCombineKernel<B,TN,Fp32,Threads>,Threads,bytes)!=cudaSuccess||resident!=1)return 7;
  if(a.localSizeBytes||a.sharedSizeBytes||a.numRegs>(Threads==576?227:255))return 8;
  if(resources){resources[0]=a.numRegs;resources[1]=int(a.localSizeBytes);resources[2]=int(a.sharedSizeBytes);resources[3]=resident;}
  if(query_only)return 0;
  if(active<1||active>128)return 4;
  MoeCombineKernel<B,TN,Fp32,Threads><<<active,Threads,bytes,stream>>>(p,active);
  return int(cudaGetLastError());
}
template<int B,int TN> int Dtype(MxMoeCombineOperandsV2 p,unsigned active,int fp32,int threads,cudaStream_t s,int* r,bool q) {
  if(threads==512){if(fp32)return Launch<B,TN,true,512>(p,active,s,r,q);return Launch<B,TN,false,512>(p,active,s,r,q);}
  if(fp32)return Launch<B,TN,true,576>(p,active,s,r,q);return Launch<B,TN,false,576>(p,active,s,r,q);
}
extern "C" int mx5_moe_combine_probe(MxMoeCombineOperandsV2 p,unsigned active,int batch,int tn,int fp32,int threads,cudaStream_t s,int* r,int query_only) {
  if((threads!=512&&threads!=576)||(fp32!=0&&fp32!=1)||(query_only!=0&&query_only!=1))return 4;
#define CASE(B) case B:if(tn==16)return Dtype<B,16>(p,active,fp32,threads,s,r,query_only);if(tn==128)return Dtype<B,128>(p,active,fp32,threads,s,r,query_only);return 4
  switch(batch){CASE(1);CASE(16);default:return 4;}
#undef CASE
}
