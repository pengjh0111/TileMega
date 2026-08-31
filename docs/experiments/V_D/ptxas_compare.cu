// Materialize three candidates with static SHM so ptxas can independently
// report the allocation and be compared with sizeof(Collective::SharedStorage).
#include "traits_common.cuh"

__device__ void touch_storage(unsigned char* bytes, int size, int* output) {
  int index = (threadIdx.x * 257 + blockIdx.x) % size;
  bytes[index] = static_cast<unsigned char>(threadIdx.x);
  if (threadIdx.x == 0) output[blockIdx.x] = bytes[index];
}

#define DEFINE_PROBE(M, N) \
extern "C" __global__ void probe_##M##x##N(int* output) { \
  using C = v_d::Collective<M, N>; \
  __shared__ typename C::SharedStorage storage; \
  touch_storage(reinterpret_cast<unsigned char*>(&storage), static_cast<int>(sizeof(storage)), output); \
}

DEFINE_PROBE(64, 64)
DEFINE_PROBE(64, 96)
DEFINE_PROBE(96, 96)
