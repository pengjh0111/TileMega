// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/ClusterPlacement.h>
#ifndef TILEMEGA_PDL
#define TILEMEGA_PDL 0
#endif
#ifndef TILEMEGA_ARCH_PATH_SM80
#define TILEMEGA_ARCH_PATH_SM80 0
#endif

namespace tilemega::codegen::executor {
template<class Arch>
struct GridDependency {
  using Copy=Async<Arch,TILEMEGA_ARCH_PATH_SM80!=0>;
  __device__ static void Wait() {
    if constexpr(TILEMEGA_PDL!=0)Copy::WaitPreviousGrid();
  }
  __device__ static void Release() {
    if constexpr(TILEMEGA_PDL!=0)Copy::LaunchDependents();
  }
};
template<class Kernel>
inline cudaError_t ServingActiveClusters(int* clusters,Kernel kernel,int threads,
    std::size_t shared,int grid) {
  if(!clusters || grid<=0 || grid%TILEMEGA_SERVING_CLUSTER_DIM)
    return cudaErrorInvalidConfiguration;
  cudaLaunchAttribute attribute{};
  attribute.id=cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim={TILEMEGA_SERVING_CLUSTER_DIM,1,1};
  cudaLaunchConfig_t config{};
  config.gridDim=dim3(grid);config.blockDim=dim3(threads);
  config.dynamicSmemBytes=shared;
  config.attrs=&attribute;config.numAttrs=1;
  return cudaOccupancyMaxActiveClusters(clusters,kernel,&config);
}

// The launch attribute and device waits form one protocol; the caller only
// enables it for a serving decode kernel compiled with TILEMEGA_PDL.
template<class Kernel,class... Args>
inline cudaError_t LaunchServing(Kernel kernel,int grid,int threads,
    std::size_t shared,cudaStream_t stream,bool pdl,Args... args) {
  if(pdl || TILEMEGA_SERVING_CLUSTER_DIM>1) {
    if(grid<=0 || grid%TILEMEGA_SERVING_CLUSTER_DIM)return cudaErrorInvalidConfiguration;
    cudaLaunchAttribute attributes[2]{};
    unsigned count=0;
    if constexpr(TILEMEGA_SERVING_CLUSTER_DIM>1) {
      attributes[count].id=cudaLaunchAttributeClusterDimension;
      attributes[count++].val.clusterDim={TILEMEGA_SERVING_CLUSTER_DIM,1,1};
    }
    if(pdl) {
      attributes[count].id=cudaLaunchAttributeProgrammaticStreamSerialization;
      attributes[count++].val.programmaticStreamSerializationAllowed=1;
    }
    cudaLaunchConfig_t config{};
    config.gridDim=dim3(grid);config.blockDim=dim3(threads);
    config.dynamicSmemBytes=shared;config.stream=stream;
    config.attrs=attributes;config.numAttrs=count;
    return cudaLaunchKernelEx(&config,kernel,args...);
  }
  kernel<<<grid,threads,shared,stream>>>(args...);
  return cudaGetLastError();
}
} // namespace tilemega::codegen::executor
