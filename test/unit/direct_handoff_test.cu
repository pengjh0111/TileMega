// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/executor/DirectHandoff.cuh>
#include <tilemega/Solver/HandoffRuntimeProjection.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/QuasiPolynomial.h>
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <vector>
using namespace tilemega::codegen::executor;
__global__ void HandoffKernel(int const* input,int* output) {
  __shared__ alignas(16) int page[128];
  DirectHandoff(page,[&](void* memory) {
    auto* p=static_cast<int*>(memory);
    p[ComputeThread()]=input[ComputeThread()]*2;
  },[&](void* memory) {
    auto* p=static_cast<int*>(memory);
    output[ComputeThread()]=p[ComputeThread()]+1;
  });
}
int main() {
  // The same synthetic 1:1 CG is projected onto the fused runtime task
  // before the device page handoff is exercised below. This catches a page
  // exchange that works in isolation but was removed from the schedule.
  tilemega::analysis::IslContext isl;
  using Relation=tilemega::analysis::CouplingRelation;
  auto tasks=Relation::FromIslText("{ [] -> [s,t] : 0<=s<2 and 0<=t<4 }");
  auto dependencies=Relation::FromIslText("{ [1,t] -> [0,t] : 0<=t<4 }");
  auto identity=Relation::FromIslText("{ [t] -> [u] : 0<=t<4 and u=t }");
  auto projection=tilemega::solver::ProjectHandoffRuntime(
      tasks,dependencies,2,{{0,1,"smem_direct",identity}});
  assert((projection.surviving_stages==std::vector<int>{1}));
  assert(projection.dependencies.Card().IsZero());
  int input[128],result[128];
  for(int i=0;i<128;++i)input[i]=i;
  int *device_input=nullptr,*device_output=nullptr;
  assert(cudaMalloc(&device_input,sizeof input)==cudaSuccess);
  assert(cudaMalloc(&device_output,sizeof result)==cudaSuccess);
  assert(cudaMemcpy(device_input,input,sizeof input,cudaMemcpyHostToDevice)==cudaSuccess);
  HandoffKernel<<<1,160>>>(device_input,device_output);
  assert(cudaDeviceSynchronize()==cudaSuccess);
  assert(cudaMemcpy(result,device_output,sizeof result,cudaMemcpyDeviceToHost)==cudaSuccess);
  for(int i=0;i<128;++i)assert(result[i]==i*2+1);
  assert(cudaFree(device_input)==cudaSuccess);
  assert(cudaFree(device_output)==cudaSuccess);
  std::cout<<"PASS synthetic 1:1 projected page handoff with an idle loader warp\n";
}
