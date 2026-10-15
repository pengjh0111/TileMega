// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_ARCH_ID 900
#include <tilemega/Codegen/executor/ServingLaunch.cuh>
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>

__global__ void ClusterKernel(unsigned* result) {
  __shared__ unsigned mailbox;
  auto group=cooperative_groups::this_cluster();
  if(threadIdx.x==0)mailbox=blockIdx.x+1;
  group.sync();
  if(threadIdx.x==0) {
    unsigned sum=0;
    for(unsigned rank=0;rank<group.num_blocks();++rank)
      sum+=*group.map_shared_rank(&mailbox,rank);
    result[blockIdx.x]=sum;
  }
  group.sync();
}
int main() {
  using namespace tilemega::codegen::executor;
  cudaDeviceProp properties{};
  if(cudaGetDeviceProperties(&properties,0)!=cudaSuccess)return 2;
  int const c=TILEMEGA_SERVING_CLUSTER_DIM;
  int grid=(properties.multiProcessorCount/c)*c,clusters=0;
  if(ServingActiveClusters(&clusters,ClusterKernel,128,0,grid)!=cudaSuccess || clusters<=0)return 3;
  grid=clusters*c;
  unsigned* output=nullptr;
  if(cudaMalloc(&output,grid*sizeof(unsigned))!=cudaSuccess)return 4;
  std::vector<unsigned> host(grid);
  for(bool pdl:{false,true}) {
    auto status=LaunchServing(ClusterKernel,grid,128,0,nullptr,pdl,output);
    if(status!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess)return 5;
    if(cudaMemcpy(host.data(),output,grid*sizeof(unsigned),cudaMemcpyDeviceToHost)!=cudaSuccess)return 6;
    for(int worker=0;worker<grid;++worker) {
      unsigned base=(worker/c)*c,expected=c*(base+1)+c*(c-1)/2;
      if(host[worker]!=expected) {
        std::fprintf(stderr,"cluster%d worker%d got%u expected%u\n",c,worker,host[worker],expected);return 7;
      }
    }
  }
  if constexpr(TILEMEGA_SERVING_CLUSTER_DIM>1)
    if(LaunchServing(ClusterKernel,grid-1,128,0,nullptr,false,output)!=cudaErrorInvalidConfiguration)return 8;
  cudaFree(output);
  std::printf("PASS serving cluster launch c=%d resident_grid=%d PDL=0,1\n",c,grid);
}
