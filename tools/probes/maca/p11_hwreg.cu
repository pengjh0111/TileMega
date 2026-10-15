// P-11: scan hardware-register ids for an AP identifier (one block per AP via 64 KiB smem).
#include <cuda_runtime.h>
#include <cstdio>
#include <set>
#ifndef REG
#define REG 0
#endif
__global__ void Read(unsigned* out) {
  extern __shared__ char s[];
  s[threadIdx.x] = 0;
  if (threadIdx.x == 0) out[blockIdx.x] = __builtin_mxc_get_hwreg(REG);
}
int main() {
  unsigned* d; cudaMalloc(&d, 104 * 4);
  cudaFuncSetAttribute(Read, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
  Read<<<104, 64, 65536>>>(d);
  cudaError_t e = cudaDeviceSynchronize();
  unsigned h[104]; cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
  std::set<unsigned> u(h, h + 104);
  std::printf("reg=%d err=%s distinct=%zu sample=%x %x %x %x\n", REG, cudaGetErrorString(e), u.size(), h[0], h[1], h[2], h[103]);
}
