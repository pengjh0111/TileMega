#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static const int N = 262144;

int main(int argc, char **argv) {
  int iterations = (argc > 1) ? atoi(argv[1]) : 1000;

  CUdevice cuDevice;
  CUcontext cuContext;
  CUDA_CHECK(cuInit(0));
  CUDA_CHECK(cuDeviceGet(&cuDevice, 0));
  CUDA_CHECK(cuCtxCreate(&cuContext, NULL, 0, cuDevice));

  CUmodule cuModule;
  CUfunction kernel;
  CUDA_CHECK(cuModuleLoad(&cuModule, "spin_wait_test.cubin"));
  CUDA_CHECK(cuModuleGetFunction(&kernel, cuModule, "spin_wait_test"));

  int regs = 0, shared = 0, local = 0, maxThreads = 0;
  cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);
  cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel);
  cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);
  cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                      kernel);
  printf("[cuFuncGetAttribute] spin_wait_test: REGS=%d SHARED_BYTES=%d "
         "LOCAL_BYTES=%d MAX_THREADS_PER_BLOCK=%d\n",
         regs, shared, local, maxThreads);

  // Reference: sum_{i=0}^{N-1} float(i)
  double ref_sum_d = 0.0;
  for (int i = 0; i < N; ++i) ref_sum_d += (double)(float)i;
  float ref_sum = (float)ref_sum_d;
  printf("Reference checksum = %f (double-precision accum = %f)\n", ref_sum,
         ref_sum_d);

  CUdeviceptr d_data, d_flag, d_checksum;
  CUDA_CHECK(cuMemAlloc(&d_data, sizeof(float) * N));
  CUDA_CHECK(cuMemAlloc(&d_flag, sizeof(int32_t)));
  CUDA_CHECK(cuMemAlloc(&d_checksum, sizeof(float)));

  CUstream stream;
  CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  int pass_count = 0, fail_count = 0;
  float min_checksum = 1e30f, max_checksum = -1e30f;

  for (int it = 0; it < iterations; ++it) {
    // Poison data so a "consumer read garbage before producer wrote" failure
    // is visibly distinguishable from "happened to read zeros".
    CUDA_CHECK(cuMemsetD32(d_data, 0xdeadbeef, N));
    CUDA_CHECK(cuMemsetD32(d_flag, 0, 1));
    CUDA_CHECK(cuMemsetD32(d_checksum, 0, 1));

    void *args[] = {&d_data, &d_flag, &d_checksum};
    // Grid = 2 tile blocks: block 0 = producer, block 1 = consumer.
    CUDA_CHECK(
        cuLaunchKernel(kernel, 2, 1, 1, 1, 1, 1, 0, stream, args, NULL));
    CUDA_CHECK(cuCtxSynchronize());

    float checksum = 0.0f;
    CUDA_CHECK(cuMemcpyDtoH(&checksum, d_checksum, sizeof(float)));

    if (checksum < min_checksum) min_checksum = checksum;
    if (checksum > max_checksum) max_checksum = checksum;

    double err = fabs((double)checksum - (double)ref_sum);
    bool ok = err < 1.0;  // generous float32 accumulation tolerance
    if (ok) {
      pass_count++;
    } else {
      fail_count++;
      if (fail_count <= 10 || it == iterations - 1) {
        printf("[iter %d] MISMATCH: checksum=%f ref=%f err=%g\n", it,
               checksum, ref_sum, err);
      }
    }
  }

  printf("\n=== SUMMARY over %d iterations ===\n", iterations);
  printf("PASS=%d FAIL=%d\n", pass_count, fail_count);
  printf("observed checksum range: [%f, %f]  (reference=%f)\n", min_checksum,
         max_checksum, ref_sum);

  CUDA_CHECK(cuModuleUnload(cuModule));
  CUDA_CHECK(cuCtxDestroy(cuContext));
  return 0;
}
