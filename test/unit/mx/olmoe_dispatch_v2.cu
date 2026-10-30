// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/MoeDispatchV2.h>
#include <cuda_runtime.h>
using namespace tilemega::backend::mx;
template<int B>
__global__ void __launch_bounds__(512) OlmoeDispatch(MxMoeDispatchOperands p) {
  extern __shared__ __align__(16) char smem[];
  MxMoeDispatchV2<B,2048,1024,64,64,false,512,64,false>::Run(p,smem);
}
template<int B>
int LaunchOlmoe(MxMoeDispatchOperands p,cudaStream_t s) {
  cudaFuncAttributes a{};
  if(cudaFuncGetAttributes(&a,OlmoeDispatch<B>)!=cudaSuccess)return 5;
  if(a.localSizeBytes||a.sharedSizeBytes||a.numRegs>255)return 6;
  OlmoeDispatch<B><<<1,512,32784,s>>>(p);
  return int(cudaGetLastError());
}
extern "C" int mx5_olmoe_dispatch(MxMoeDispatchOperands p,int batch,void* stream) {
  auto s=static_cast<cudaStream_t>(stream);
  if(batch==1)return LaunchOlmoe<1>(p,s);
  if(batch==16)return LaunchOlmoe<16>(p,s);
  return 4;
}
