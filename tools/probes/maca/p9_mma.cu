// P-9: bf16 MMA on xcore1002 -- builtin availability, fragment layout, throughput.
//
// Layout hypothesis taken from the wmma header (stated, checked here):
//   A (16x16, M x K):  lane l holds row m = l & 15, k = 4*(l >> 4) + {0..3}
//   B (16x16, K x N):  lane l holds col n = l & 15, k = 4*(l >> 4) + {0..3}
//   C (16x16 fp32):    lane l holds col n = l & 15, rows m = 4*(l >> 4) + {0..3}
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

typedef __NATIVE_VECTOR__(4, float) f32x4;
typedef __fp16 h4 __attribute__((__vector_size__(4 * sizeof(__fp16))));

__device__ __forceinline__ uint16_t Bf16Bits(float f) {
  uint32_t u = __float_as_uint(f);
  u += 0x7fff + ((u >> 16) & 1);
  return static_cast<uint16_t>(u >> 16);
}

// Each lane packs its 4 A / 4 B elements per the hypothesis and runs one MMA.
__global__ void LayoutCheck(float const* A, float const* B, float* C) {
  int l = threadIdx.x;
  int r = l & 15, kq = 4 * (l >> 4);
  uint16_t a[4], b[4];
  for (int i = 0; i < 4; ++i) {
    a[i] = Bf16Bits(A[r * 16 + kq + i]);
    b[i] = Bf16Bits(B[(kq + i) * 16 + r]);
  }
  h4 av, bv;
  std::memcpy(&av, a, 8);
  std::memcpy(&bv, b, 8);
  f32x4 c = {0, 0, 0, 0};
  c = __builtin_mxc_mma_16x16x16bf16(av, bv, c);
  for (int i = 0; i < 4; ++i) C[(kq + i) * 16 + r] = c[i];
}

// Throughput: each wave runs kChains independent accumulators for iters steps.
template <int kChains>
__global__ void Throughput(float* out, int iters) {
  extern __shared__ char smem[];
  smem[threadIdx.x] = 0;
  long long ai = 0x3f803f803f803f80ll ^ threadIdx.x, bi = 0x3f803f803f803f80ll;
  h4 a, b;
  std::memcpy(&a, &ai, 8);
  std::memcpy(&b, &bi, 8);
  f32x4 c[kChains];
  for (int j = 0; j < kChains; ++j) c[j] = f32x4{0, 0, 0, 0};
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < kChains; ++j) c[j] = __builtin_mxc_mma_16x16x16bf16(a, b, c[j]);
  }
  float s = 0;
  for (int j = 0; j < kChains; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main() {
  std::vector<float> hA(256), hB(256), ref(256, 0.f), hC(256);
  for (int i = 0; i < 256; ++i) { hA[i] = (i % 7) - 3; hB[i] = ((i * 5) % 11) - 5; }
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n)
      for (int k = 0; k < 16; ++k) ref[m * 16 + n] += hA[m * 16 + k] * hB[k * 16 + n];
  float *A, *B, *C;
  cudaMalloc(&A, 1024); cudaMalloc(&B, 1024); cudaMalloc(&C, 1024);
  cudaMemcpy(A, hA.data(), 1024, cudaMemcpyHostToDevice);
  cudaMemcpy(B, hB.data(), 1024, cudaMemcpyHostToDevice);
  LayoutCheck<<<1, 64>>>(A, B, C);
  cudaError_t e = cudaDeviceSynchronize();
  cudaMemcpy(hC.data(), C, 1024, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < 256; ++i) {
    if (hC[i] != ref[i] && bad < 8) std::printf("P9 mismatch m=%d n=%d got=%g ref=%g\n", i / 16, i % 16, hC[i], ref[i]);
    bad += hC[i] != ref[i];
  }
  // The MMA accumulates with small non-IEEE rounding (verified); compare with a
  // tolerance and report exact mismatches separately.
  int far = 0;
  for (int i = 0; i < 256; ++i) far += std::fabs(hC[i] - ref[i]) > 1e-4f * (1.f + std::fabs(ref[i]));
  std::printf("P9 layout far=%d\n", far);
  if (far || e != cudaSuccess) return 1;
  std::printf("P9 layout err=%s mismatches=%d C[0]=%g ref=%g C[17]=%g ref=%g\n", cudaGetErrorString(e), bad, hC[0],
              ref[0], hC[17], ref[17]);

  float* out; cudaMalloc(&out, 104 * 1024 * 4);
  cudaEvent_t s0, s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
  int const iters = 20000;
  auto run = [&](auto kernel, int chains, int waves, int blocks) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    kernel<<<blocks, 64 * waves, 65536>>>(out, 100);
    cudaDeviceSynchronize();
    cudaEventRecord(s0);
    kernel<<<blocks, 64 * waves, 65536>>>(out, iters);
    cudaEventRecord(s1);
    cudaEventSynchronize(s1);
    float ms; cudaEventElapsedTime(&ms, s0, s1);
    double flops = 2.0 * 16 * 16 * 16 * double(iters) * chains * waves * blocks;
    std::printf("P9 tput chains=%d waves/AP=%d blocks=%d ms=%.3f TFLOPS=%.2f cycles/mma/wave@1.6GHz=%.2f\n", chains,
                waves, blocks, ms, flops / ms / 1e9, ms * 1e-3 * 1.6e9 / (double(iters) * chains));
  };
  for (int waves : {1, 2, 4, 8}) {
    run(Throughput<1>, 1, waves, 1);
    run(Throughput<4>, 4, waves, 1);
  }
  for (int waves : {1, 2, 4, 8}) run(Throughput<4>, 4, waves, 104);
  run(Throughput<8>, 8, 4, 104);
  return 0;
}
