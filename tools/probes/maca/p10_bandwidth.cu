// P-10: DRAM / L2 read bandwidth vs waves per CTA, CTAs per AP, load method and
// in-flight depth; plus a functional check of arrive_gvmcnt counting.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

typedef __NATIVE_VECTOR__(4, unsigned int) b128;

// Register path: each thread keeps kUnroll 16 B loads in flight.
template <int kUnroll>
__global__ void ReadReg(uint4 const* __restrict__ src, size_t n16, unsigned* out) {
  extern __shared__ char smem[];
  if (threadIdx.x == 0) smem[0] = 0;
  size_t const stride = size_t(gridDim.x) * blockDim.x;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned acc = 0;
  for (; i + (kUnroll - 1) * stride < n16; i += kUnroll * stride) {
    uint4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) v[u] = src[i + u * stride];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x12345678u) out[0] = acc;
}

// Async path: ldg_b128_bsm into a kStages-deep smem ring, wait with arrive_gvmcnt.
template <int kStages>
__global__ void ReadBsm(uint4 const* __restrict__ src, size_t n16, unsigned* out) {
  extern __shared__ uint4 ring[];
  size_t const per_iter = size_t(gridDim.x) * blockDim.x;
  size_t base = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t iters = n16 / per_iter;
  unsigned acc = 0;
  for (int s = 0; s < kStages - 1 && s < int(iters); ++s)
    __builtin_mxc_ldg_b128_bsm(&ring[s * blockDim.x + threadIdx.x], (void*)(src + base + s * per_iter), 0, -1,
                               false, true, false, false);
  for (size_t it = 0; it < iters; ++it) {
    size_t nxt = it + kStages - 1;
    if (nxt < iters)
      __builtin_mxc_ldg_b128_bsm(&ring[(nxt % kStages) * blockDim.x + threadIdx.x],
                                 (void*)(src + base + nxt * per_iter), 0, -1, false, true, false, false);
    // Loads retire in order, so leaving kStages-1 outstanding completes `it`.
    if (nxt < iters) __builtin_mxc_arrive_gvmcnt(kStages - 1);
    else __builtin_mxc_arrive_gvmcnt(0);
    uint4 v = ring[(it % kStages) * blockDim.x + threadIdx.x];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) out[0] = acc;
}

// arrive_gvmcnt functional check: with n loads issued, waiting for n-1 must
// make the first one visible in smem (each thread checks its own slot).
template <int depth>
__global__ void GvmCheck(uint4 const* src, unsigned* bad) {
  extern __shared__ uint4 ring[];
  int t = threadIdx.x;
  for (int s = 0; s < 8; ++s) ring[s * blockDim.x + t] = uint4{0, 0, 0, 0};
  __syncthreads();
  for (int s = 0; s < depth; ++s)
    __builtin_mxc_ldg_b128_bsm(&ring[s * blockDim.x + t], (void*)(src + (size_t(blockIdx.x) * 8 + s) * blockDim.x + t),
                               0, -1, false, true, false, false);
  __builtin_mxc_arrive_gvmcnt(depth - 1);
  uint4 v = ring[t];
  uint4 e = src[size_t(blockIdx.x) * 8 * blockDim.x + t];
  if (v.x != e.x || v.y != e.y || v.z != e.z || v.w != e.w) atomicAdd(bad, 1u);
}

__global__ void Fill(uint4* p, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = uint4{unsigned(i), unsigned(i * 3), unsigned(i * 7), unsigned(i ^ 0x5a5a5a5a)};
}

int main() {
  size_t const bytes = size_t(2) << 30;
  size_t const n16 = bytes / 16;
  uint4* buf; cudaMalloc(&buf, bytes);
  unsigned* out; cudaMalloc(&out, 64);
  Fill<<<1024, 256>>>(buf, n16);
  cudaDeviceSynchronize();
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  auto time = [&](auto kernel, int grid, int threads, int smem, size_t n, char const* name, int param, int reps) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, threads, smem>>>(buf, n, out);
    cudaError_t e0 = cudaGetLastError();
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < reps; ++r) kernel<<<grid, threads, smem>>>(buf, n, out);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    std::printf("P10 %s param=%d grid=%d threads=%d smem=%d bytes=%zu GB/s=%.1f launch=%s\n", name, param, grid, threads,
                smem, n * 16, double(n) * 16 * reps / (ms * 1e6), cudaGetErrorString(e0));
  };
  int const ap = 104;
  // DRAM, register path.
  for (int threads : {128, 256, 512}) {
    time(ReadReg<1>, ap, threads, 65536, n16, "dram_reg_1cta", 1, 3);
    time(ReadReg<2>, ap, threads, 65536, n16, "dram_reg_1cta", 2, 3);
    time(ReadReg<4>, ap, threads, 65536, n16, "dram_reg_1cta", 4, 3);
    time(ReadReg<8>, ap, threads, 65536, n16, "dram_reg_1cta", 8, 3);
  }
  for (int u : {1, 2, 4, 8}) {
    if (u == 1) time(ReadReg<1>, 2 * ap, 128, 32768, n16, "dram_reg_2cta", 1, 3);
    if (u == 2) time(ReadReg<2>, 2 * ap, 128, 32768, n16, "dram_reg_2cta", 2, 3);
    if (u == 4) time(ReadReg<4>, 2 * ap, 128, 32768, n16, "dram_reg_2cta", 4, 3);
    if (u == 8) time(ReadReg<8>, 2 * ap, 128, 32768, n16, "dram_reg_2cta", 8, 3);
  }
  // DRAM, async smem path (ring = stages * threads * 16 B).
  for (int threads : {128, 256, 512}) {
    time(ReadBsm<2>, ap, threads, 2 * threads * 16, n16, "dram_bsm_1cta", 2, 3);
    time(ReadBsm<4>, ap, threads, 4 * threads * 16, n16, "dram_bsm_1cta", 4, 3);
    time(ReadBsm<8>, ap, threads, 8 * threads * 16, n16, "dram_bsm_1cta", 8, 3);
  }
  time(ReadBsm<4>, 2 * ap, 128, 4 * 128 * 16, n16, "dram_bsm_2cta", 4, 3);
  time(ReadBsm<8>, 2 * ap, 128, 8 * 128 * 16, n16, "dram_bsm_2cta", 8, 3);
  // L2: a 4 MiB working set re-read many times.
  size_t const l2n = (size_t(4) << 20) / 16;
  time(ReadReg<4>, ap, 256, 65536, l2n, "l2_reg_1cta", 4, 200);
  time(ReadReg<8>, ap, 512, 65536, l2n, "l2_reg_1cta", 8, 200);
  time(ReadBsm<4>, ap, 256, 4 * 256 * 16, l2n, "l2_bsm_1cta", 4, 200);
  // arrive_gvmcnt semantics.
  unsigned* bad; cudaMalloc(&bad, 4);
  auto gvm = [&](auto kernel, int depth) {
    cudaMemset(bad, 0, 4);
    kernel<<<ap, 128, 8 * 128 * 16>>>(buf, bad);
    unsigned h; cudaMemcpy(&h, bad, 4, cudaMemcpyDeviceToHost);
    std::printf("P10 gvmcnt depth=%d wait=%d mismatching_threads=%u err=%s\n", depth, depth - 1, h,
                cudaGetErrorString(cudaDeviceSynchronize()));
  };
  for (int rep = 0; rep < 20; ++rep) { gvm(GvmCheck<1>, 1); gvm(GvmCheck<2>, 2); gvm(GvmCheck<4>, 4); gvm(GvmCheck<8>, 8); }
  return 0;
}
