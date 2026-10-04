// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Target/ArchDispatch.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

__global__ void QueryClusterKernel() {}

int main() {
  auto check=[](cudaError_t error) {
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
  };
  auto attr=[&](cudaDeviceAttr key) {
    int value=0;check(cudaDeviceGetAttribute(&value,key,0));return value;
  };
  using Caps=tilemega::arch::Caps<tilemega::arch::Sm120>;
  cudaLaunchConfig_t config{};
  config.gridDim=dim3(160);config.blockDim=dim3(128);
  int max_cluster=0;
  check(cudaOccupancyMaxPotentialClusterSize(&max_cluster,QueryClusterKernel,&config));
  std::cout<<"{\"architecture\":\"sm_120\",\"query_evidence\":\"verified CUDA attributes\","
      <<"\"num_sms\":"<<attr(cudaDevAttrMultiProcessorCount)
      <<",\"shared_memory_per_sm\":"<<attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor)
      <<",\"shared_memory_per_block\":"<<attr(cudaDevAttrMaxSharedMemoryPerBlock)
      <<",\"shared_memory_per_block_optin\":"<<attr(cudaDevAttrMaxSharedMemoryPerBlockOptin)
      <<",\"l2_bytes\":"<<attr(cudaDevAttrL2CacheSize)
      <<",\"cluster_launch\":"<<attr(cudaDevAttrClusterLaunch)
      <<",\"max_portable_cluster_query\":"<<max_cluster
      <<",\"caps\":{\"kCpAsync\":"<<Caps::kCpAsync
      <<",\"kMbarrier\":"<<Caps::kMbarrier
      <<",\"kMbarrierTx\":"<<Caps::kMbarrierTx
      <<",\"kMbarrierTryWait\":"<<Caps::kMbarrierTryWait
      <<",\"kBulkCopy\":"<<Caps::kBulkCopy
      <<",\"kTma\":"<<Caps::kTma
      <<",\"kBulkPrefetch\":"<<Caps::kBulkPrefetch
      <<",\"kPdl\":"<<Caps::kPdl
      <<",\"kCluster\":"<<Caps::kCluster
      <<",\"kMaxClusterSize\":"<<Caps::kMaxClusterSize
      <<",\"kTcgen05\":"<<Caps::kTcgen05
      <<",\"kBf16CollectiveBuilder\":"<<Caps::kBf16CollectiveBuilder
      <<"},\"serving_path_status\":\"pending generated-source/PTX inspection; caps alone do not prove execution\"}\n";
}
