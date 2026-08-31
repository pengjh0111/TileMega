// V-C: complete cluster/DSMEM communication kernel. This translation unit is
// cross-compiled for capable targets and intentionally not run on sm_89.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <tilemega/Target/ArchDispatch.h>

namespace cg = cooperative_groups;
using Arch = tilemega::arch::CurrentArch;
static_assert(tilemega::arch::Caps<Arch>::kCluster,
              "V-C must only be built for a cluster-capable target");

extern "C" __global__ __launch_bounds__(256) __cluster_dims__(2, 1, 1)
void cluster_dsmem_exchange(int* output) {
  __shared__ int tile[256];
  auto cluster = cg::this_cluster();
  tile[threadIdx.x] = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  cluster.sync();

  int peer_rank = (cluster.block_rank() + 1) % cluster.num_blocks();
  int* peer_tile = cluster.map_shared_rank(tile, peer_rank);
  output[blockIdx.x * blockDim.x + threadIdx.x] = peer_tile[threadIdx.x];
  cluster.sync();
}
