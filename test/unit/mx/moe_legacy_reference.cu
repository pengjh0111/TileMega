// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/MoETopKTaskBody.h>
#include <tilemega/Codegen/tasks/MoECombineTaskBody.h>
#include <cuda_runtime.h>
using namespace tilemega::codegen;
using Arch=tilemega::arch::Xcore1000;
__global__ void LegacyTopKKernel(MoeTopKOperands p) {
  MoETopKTaskBody<Arch,8,128>::Run(p,blockIdx.x);
}
__global__ void LegacyCombineKernel(MoeCombineOperands p) {
  extern __shared__ __align__(16) char storage[];
  using Body=MoECombineTaskBody<Arch,8,1,128>;
  Body::Run(p,blockIdx.x,*reinterpret_cast<Body::SharedStorage*>(storage));
}
extern "C" int mx5_legacy_topk(MoeTopKOperands p,cudaStream_t stream) {
  LegacyTopKKernel<<<(p.tokens+127)/128,128,32784,stream>>>(p);
  return int(cudaGetLastError());
}
extern "C" int mx5_legacy_combine(MoeCombineOperands p,cudaStream_t stream) {
  LegacyCombineKernel<<<p.tokens*((p.channels+127)/128),128,32784,stream>>>(p);
  return int(cudaGetLastError());
}
extern "C" unsigned mx5_legacy_topk_operands_size(){return sizeof(MoeTopKOperands);}
extern "C" unsigned mx5_legacy_combine_operands_size(){return sizeof(MoeCombineOperands);}
