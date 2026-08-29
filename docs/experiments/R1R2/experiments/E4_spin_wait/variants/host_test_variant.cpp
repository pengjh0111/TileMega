#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static const int N = 262144;

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <cubin> <entry_name> [iterations]\n", argv[0]);
    return 1;
  }
  const char *cubinPath = argv[1];
  const char *entryName = argv[2];
  int iterations = (argc > 3) ? atoi(argv[3]) : 200;

  CUdevice cuDevice;
  CUcontext cuContext;
  CUDA_CHECK(cuInit(0));
  CUDA_CHECK(cuDeviceGet(&cuDevice, 0));
  CUDA_CHECK(cuCtxCreate(&cuContext, NULL, 0, cuDevice));

  CUmodule cuModule;
  CUfunction kernel;
  CUDA_CHECK(cuModuleLoad(&cuModule, cubinPath));
  CUDA_CHECK(cuModuleGetFunction(&kernel, cuModule, entryName));

  double ref_sum_d = 0.0;
  for (int i = 0; i < N; ++i) ref_sum_d += (double)(float)i;
  float ref_sum = (float)ref_sum_d;

  CUdeviceptr d_data, d_flag, d_checksum;
  CUDA_CHECK(cuMemAlloc(&d_data, sizeof(float) * N));
  CUDA_CHECK(cuMemAlloc(&d_flag, sizeof(int32_t)));
  CUDA_CHECK(cuMemAlloc(&d_checksum, sizeof(float)));

  CUstream stream;
  CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  int pass_count = 0, fail_count = 0;
  float min_checksum = 1e30f, max_checksum = -1e30f;

  for (int it = 0; it < iterations; ++it) {
    CUDA_CHECK(cuMemsetD32(d_data, 0xdeadbeef, N));
    CUDA_CHECK(cuMemsetD32(d_flag, 0, 1));
    CUDA_CHECK(cuMemsetD32(d_checksum, 0, 1));

    void *args[] = {&d_data, &d_flag, &d_checksum};
    CUDA_CHECK(
        cuLaunchKernel(kernel, 2, 1, 1, 1, 1, 1, 0, stream, args, NULL));
    CUDA_CHECK(cuCtxSynchronize());

    float checksum = 0.0f;
    CUDA_CHECK(cuMemcpyDtoH(&checksum, d_checksum, sizeof(float)));
    if (checksum < min_checksum) min_checksum = checksum;
    if (checksum > max_checksum) max_checksum = checksum;

    double err = fabs((double)checksum - (double)ref_sum);
    if (err < 1.0) pass_count++; else fail_count++;
  }

  printf("[%s] over %d iterations: PASS=%d FAIL=%d  observed range=[%f, %f] ref=%f\n",
         entryName, iterations, pass_count, fail_count, min_checksum, max_checksum, ref_sum);

  CUDA_CHECK(cuModuleUnload(cuModule));
  CUDA_CHECK(cuCtxDestroy(cuContext));
  return 0;
}
