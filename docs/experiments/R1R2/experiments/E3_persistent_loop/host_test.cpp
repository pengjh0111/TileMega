#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    CUresult err = call;                                                     \
    if (err != CUDA_SUCCESS) {                                               \
      const char *errStr;                                                    \
      cuGetErrorString(err, &errStr);                                        \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              errStr);                                                       \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

static const int N = 1000000;

int main() {
  CUdevice dev;
  CUcontext ctx;
  CUDA_CHECK(cuInit(0));
  CUDA_CHECK(cuDeviceGet(&dev, 0));
  CUDA_CHECK(cuCtxCreate(&ctx, NULL, 0, dev));

  int smCount = 0;
  cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
  printf("SM count = %d\n", smCount);

  CUmodule modP, modNP;
  CUfunction fnP, fnNP;
  CUDA_CHECK(cuModuleLoad(&modP, "persistent_loop.cubin"));
  CUDA_CHECK(cuModuleGetFunction(&fnP, modP, "persistent_loop"));
  CUDA_CHECK(cuModuleLoad(&modNP, "non_persistent.cubin"));
  CUDA_CHECK(cuModuleGetFunction(&fnNP, modNP, "non_persistent"));

  CUdeviceptr d_out;
  CUDA_CHECK(cuMemAlloc(&d_out, sizeof(float) * N));

  std::vector<float> h_out(N);

  // --- Correctness check: persistent version, grid = SM count ---
  CUDA_CHECK(cuMemsetD32(d_out, 0, N));
  int total_tiles = N;
  void *argsP[] = {&total_tiles, &d_out};
  CUDA_CHECK(cuLaunchKernel(fnP, smCount, 1, 1, 1, 1, 1, 0, 0, argsP, NULL));
  CUDA_CHECK(cuCtxSynchronize());
  CUDA_CHECK(cuMemcpyDtoH(h_out.data(), d_out, sizeof(float) * N));
  int mismatches = 0;
  for (int i = 0; i < N; ++i) {
    float expect = (float)i * 2.0f;
    if (h_out[i] != expect) {
      if (mismatches < 5)
        printf("[persistent] MISMATCH at %d: got %f expect %f\n", i, h_out[i], expect);
      mismatches++;
    }
  }
  printf("[persistent, grid=%d, total_tiles=%d] mismatches=%d / %d\n", smCount, total_tiles, mismatches, N);

  // --- Correctness check: non-persistent version, grid = N ---
  CUDA_CHECK(cuMemsetD32(d_out, 0, N));
  void *argsNP[] = {&d_out};
  CUDA_CHECK(cuLaunchKernel(fnNP, N, 1, 1, 1, 1, 1, 0, 0, argsNP, NULL));
  CUDA_CHECK(cuCtxSynchronize());
  CUDA_CHECK(cuMemcpyDtoH(h_out.data(), d_out, sizeof(float) * N));
  mismatches = 0;
  for (int i = 0; i < N; ++i) {
    float expect = (float)i * 2.0f;
    if (h_out[i] != expect) {
      if (mismatches < 5)
        printf("[non-persistent] MISMATCH at %d: got %f expect %f\n", i, h_out[i], expect);
      mismatches++;
    }
  }
  printf("[non-persistent, grid=%d] mismatches=%d / %d\n", N, mismatches, N);

  // --- Timing comparison ---
  CUevent start, stop;
  cuEventCreate(&start, 0);
  cuEventCreate(&stop, 0);
  float ms;

  const int TRIALS = 10;

  // Persistent, grid=SM count
  cuEventRecord(start, 0);
  for (int t = 0; t < TRIALS; ++t) {
    CUDA_CHECK(cuLaunchKernel(fnP, smCount, 1, 1, 1, 1, 1, 0, 0, argsP, NULL));
  }
  cuEventRecord(stop, 0);
  cuEventSynchronize(stop);
  cuEventElapsedTime(&ms, start, stop);
  printf("[timing] persistent (grid=%d, total_tiles=%d): %.4f ms / launch\n", smCount, total_tiles, ms / TRIALS);

  // Non-persistent, grid=N
  cuEventRecord(start, 0);
  for (int t = 0; t < TRIALS; ++t) {
    CUDA_CHECK(cuLaunchKernel(fnNP, N, 1, 1, 1, 1, 1, 0, 0, argsNP, NULL));
  }
  cuEventRecord(stop, 0);
  cuEventSynchronize(stop);
  cuEventElapsedTime(&ms, start, stop);
  printf("[timing] non-persistent (grid=%d): %.4f ms / launch\n", N, ms / TRIALS);

  // Persistent, grid=10000 (matching the spec's suggested comparison point),
  // each block does N/10000=100 tiles.
  int grid_10k = 10000;
  void *argsP2[] = {&total_tiles, &d_out};
  cuEventRecord(start, 0);
  for (int t = 0; t < TRIALS; ++t) {
    CUDA_CHECK(cuLaunchKernel(fnP, grid_10k, 1, 1, 1, 1, 1, 0, 0, argsP2, NULL));
  }
  cuEventRecord(stop, 0);
  cuEventSynchronize(stop);
  cuEventElapsedTime(&ms, start, stop);
  printf("[timing] persistent (grid=%d, total_tiles=%d, ~%d tiles/block): %.4f ms / launch\n",
         grid_10k, total_tiles, total_tiles / grid_10k, ms / TRIALS);

  return 0;
}
