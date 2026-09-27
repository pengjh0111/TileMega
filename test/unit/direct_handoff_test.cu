// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/executor/DirectHandoff.cuh>
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
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
  int input[128],result[128];
  for(int i=0;i<128;++i)input[i]=i;
  int *device_input=nullptr,*device_output=nullptr;
  assert(cudaMalloc(&device_input,sizeof input)==cudaSuccess);
  assert(cudaMalloc(&device_output,sizeof result)==cudaSuccess);
  assert(cudaMemcpy(device_input,input,sizeof input,cudaMemcpyHostToDevice)==cudaSuccess);
  HandoffKernel<<<1,128>>>(device_input,device_output);
  assert(cudaDeviceSynchronize()==cudaSuccess);
  assert(cudaMemcpy(result,device_output,sizeof result,cudaMemcpyDeviceToHost)==cudaSuccess);
  for(int i=0;i<128;++i)assert(result[i]==i*2+1);
  assert(cudaFree(device_input)==cudaSuccess);
  assert(cudaFree(device_output)==cudaSuccess);
  std::cout<<"PASS 128 direct page handoffs\n";
}
