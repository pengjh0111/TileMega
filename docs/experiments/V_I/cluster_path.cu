// V-I advanced path: compile the complete cluster/DSMEM synchronization form
// only for targets whose ArchDispatch capability table enables clusters.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <tilemega/Target/ArchDispatch.h>

namespace cg = cooperative_groups;
using Arch = tilemega::arch::CurrentArch;
static_assert(tilemega::arch::Caps<Arch>::kCluster);
static_assert(tilemega::arch::Caps<Arch>::kTma);
static_assert(tilemega::arch::Caps<Arch>::kWarpSpecialized);
static_assert(!tilemega::arch::Caps<Arch>::kTcgen05);

extern "C" __global__ __cluster_dims__(2, 1, 1)
void tilemega_cluster_path(int* output) {
  __shared__ int local;
  auto cluster = cg::this_cluster();
  if (threadIdx.x == 0) local = static_cast<int>(blockIdx.x);
  cluster.sync();
  int peer_rank = (cluster.block_rank() + 1) % cluster.num_blocks();
  int* peer = cluster.map_shared_rank(&local, peer_rank);
  if (threadIdx.x == 0) output[blockIdx.x] = *peer;
  cluster.sync();
}
