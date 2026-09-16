// SPDX-License-Identifier: BSD-3-Clause
// Compile-only on sm_120: no claim of execution on the sm_89 development GPU.
#include <tilemega/Codegen/tasks/ClusterSync.cuh>

__global__ void sync_v3_cluster_arrive(unsigned long long* observed) {
  using CS = tilemega::codegen::ClusterSync<tilemega::arch::CurrentArch>;
  extern __shared__ unsigned long long local[];
  auto* counter = CS::CounterPeer(local, 0);
  if (threadIdx.x == 0) observed[blockIdx.x] = CS::ArriveCounter(counter);
}
