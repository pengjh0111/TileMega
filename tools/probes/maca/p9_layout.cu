// P-9: recover the bf16 16x16x16 MMA fragment layout from raw lane values.
// Experiment E: lane l supplies a[i] = 1 iff (l, i) == (la, ia), b[j] = 1 + idx(l, j);
// the nonzero outputs show which (lane, j) of B pairs with (la, ia) of A, and
// where the products land in C.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
typedef __NATIVE_VECTOR__(4, float) f32x4;
typedef __fp16 h4 __attribute__((__vector_size__(4 * sizeof(__fp16))));
__device__ uint16_t B16(float f) { return static_cast<uint16_t>(__float_as_uint(f) >> 16); }
__global__ void K(int la, int ia, int mode, float* out) {
  int l = threadIdx.x;
  uint16_t a[4], b[4];
  for (int i = 0; i < 4; ++i) {
    if (mode == 0) { a[i] = B16((l == la && i == ia) ? 1.f : 0.f); b[i] = B16(float(l * 4 + i)); }
    else           { b[i] = B16((l == la && i == ia) ? 1.f : 0.f); a[i] = B16(float(l * 4 + i)); }
  }
  h4 av, bv; std::memcpy(&av, a, 8); std::memcpy(&bv, b, 8);
  f32x4 c = {0, 0, 0, 0};
  c = __builtin_mxc_mma_16x16x16bf16(av, bv, c);
  for (int i = 0; i < 4; ++i) out[l * 4 + i] = c[i];
}
int main() {
  float* d; cudaMalloc(&d, 256 * 4); float h[256];
  for (int mode = 0; mode < 2; ++mode) {
    for (int la : {0, 1, 15, 16, 17, 32, 48, 63}) for (int ia = 0; ia < 4; ++ia) {
      K<<<1, 64>>>(la, ia, mode, d); cudaMemcpy(h, d, 1024, cudaMemcpyDeviceToHost);
      std::printf("%s one-hot lane=%d i=%d ->", mode ? "B" : "A", la, ia);
      for (int p = 0; p < 256; ++p) if (h[p] != 0) std::printf(" C(l%d,i%d)=%g", p / 4, p % 4, h[p]);
      std::printf("\n");
    }
  }
}
