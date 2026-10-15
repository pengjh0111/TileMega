// P-1: a host-only C++ translation unit using the CUDA runtime and driver
// APIs through cu-bridge headers, compiled by g++.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdio>
int main() {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  cudaDeviceProp p{};
  if (n) cudaGetDeviceProperties(&p, 0);
  int l2 = 0;
  if (n) cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);
  CUresult r = cuInit(0);
  int dv = 0, rv = 0;
  cudaDriverGetVersion(&dv); cudaRuntimeGetVersion(&rv);
  std::printf("err=%d count=%d name=%s sm=%d cc=%d.%d l2=%d cuInit=%d driver=%d runtime=%d\n", (int)e, n, p.name,
              p.multiProcessorCount, p.major, p.minor, l2, (int)r, dv, rv);
  return 0;
}
