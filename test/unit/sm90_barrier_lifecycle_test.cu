// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/executor/Async.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace {
void Check(cudaError_t error) {
  if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(2);}
}
__global__ void Reuse(int const* source,int* failures) {
  using A=tilemega::codegen::executor::Async<tilemega::arch::Sm90>;
  __shared__ __align__(16) unsigned char storage[1024];
  auto* full=reinterpret_cast<std::uint64_t*>(storage);
  auto* tile=reinterpret_cast<int*>(storage+512);
  for(int task=0;task<64;++task) {
    __syncthreads();
    if(threadIdx.x==0){A::Init(full,1);A::InitFence();}
    __syncthreads();
    if(threadIdx.x==0){A::ExpectTx(full,512);A::Bulk(tile,source+task*128,512,full);}
    A::Wait(full,0);
    if(tile[threadIdx.x]!=task*128+int(threadIdx.x))atomicAdd(failures,1);
    __syncthreads();
    if(threadIdx.x==0)A::Invalidate(full);
    __syncthreads();
    // An epilogue aliases the header before the next task creates a barrier.
    auto* output=reinterpret_cast<int*>(storage);
    output[threadIdx.x]=tile[threadIdx.x]+17;
    __syncthreads();
    if(output[threadIdx.x]!=task*128+int(threadIdx.x)+17)atomicAdd(failures,1);
  }
}
}
int main() {
  int *source,*failures;Check(cudaMallocManaged(&source,64*128*sizeof(int)));
  Check(cudaMallocManaged(&failures,sizeof(int)));*failures=0;
  for(int i=0;i<64*128;++i)source[i]=i;
  Reuse<<<4,128>>>(source,failures);Check(cudaDeviceSynchronize());
  int result=*failures;Check(cudaFree(failures));Check(cudaFree(source));
  std::printf("barrier invalidation and aliased header reuse: %s (%d errors)\n",result?"FAIL":"PASS",result);
  return result?1:0;
}
