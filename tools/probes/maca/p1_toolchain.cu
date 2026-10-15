// P-1: dynamic smem, opt-in attribute, occupancy query, templates and
// if constexpr in a shared library driven from a torch process via ctypes.
#include <cuda_runtime.h>
#include <cstdint>

template <int kMode>
__global__ void Scale(const float* in, float* out, int n) {
  extern __shared__ float smem[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  smem[threadIdx.x] = i < n ? in[i] : 0.f;
  __syncthreads();
  if constexpr (kMode == 0) {
    if (i < n) out[i] = smem[threadIdx.x] * 2.f;
  } else {
    if (i < n) out[i] = smem[blockDim.x - 1 - threadIdx.x] + 1.f;
  }
}

extern "C" int probe_run(const float* in, float* out, int n, int mode, void* stream, int* info) {
  const int threads = 256;
  const int smem = 65536;
  cudaError_t e = mode == 0
      ? cudaFuncSetAttribute(Scale<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)
      : cudaFuncSetAttribute(Scale<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  info[0] = static_cast<int>(e);
  int blocks = -1;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, Scale<0>, threads, smem);
  info[1] = static_cast<int>(e);
  info[2] = blocks;
  cudaFuncAttributes attr{};
  info[3] = static_cast<int>(cudaFuncGetAttributes(&attr, Scale<0>));
  info[4] = attr.numRegs;
  info[5] = static_cast<int>(attr.localSizeBytes);
  info[6] = attr.maxThreadsPerBlock;
  auto s = static_cast<cudaStream_t>(stream);
  if (mode == 0) Scale<0><<<(n + threads - 1) / threads, threads, smem, s>>>(in, out, n);
  else Scale<1><<<(n + threads - 1) / threads, threads, threads * 4, s>>>(in, out, n);
  info[7] = static_cast<int>(cudaGetLastError());
  return 0;
}
