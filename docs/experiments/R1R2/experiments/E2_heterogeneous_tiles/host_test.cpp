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

static void printFuncAttrs(CUfunction f, const char *label) {
  int regs = 0, shared = 0, local = 0, maxThreads = 0, constMem = 0;
  cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, f);
  cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f);
  cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f);
  cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f);
  cuFuncGetAttribute(&constMem, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, f);
  printf("[cuFuncGetAttribute] %s: REGS=%d SHARED_BYTES=%d LOCAL_BYTES=%d "
         "MAX_THREADS_PER_BLOCK=%d CONST_BYTES=%d\n",
         label, regs, shared, local, maxThreads, constMem);
}

int main() {
  CUdevice cuDevice;
  CUcontext cuContext;
  CUDA_CHECK(cuInit(0));
  CUDA_CHECK(cuDeviceGet(&cuDevice, 0));
  CUDA_CHECK(cuCtxCreate(&cuContext, NULL, 0, cuDevice));

  // ---- Load and inspect isolated single-branch kernels ----
  {
    CUmodule mod;
    CUfunction fn;
    CUDA_CHECK(cuModuleLoad(&mod, "branch_a_only.cubin"));
    CUDA_CHECK(cuModuleGetFunction(&fn, mod, "branch_a_only"));
    printFuncAttrs(fn, "branch_a_only (isolated)");
    cuModuleUnload(mod);
  }
  {
    CUmodule mod;
    CUfunction fn;
    CUDA_CHECK(cuModuleLoad(&mod, "branch_b_only.cubin"));
    CUDA_CHECK(cuModuleGetFunction(&fn, mod, "branch_b_only"));
    printFuncAttrs(fn, "branch_b_only (isolated)");
    cuModuleUnload(mod);
  }

  // ---- Load combined heterogeneous kernel ----
  CUmodule cuModule;
  CUfunction kernel;
  CUDA_CHECK(cuModuleLoad(&cuModule, "heterogeneous_dispatch.cubin"));
  CUDA_CHECK(
      cuModuleGetFunction(&kernel, cuModule, "heterogeneous_dispatch"));
  printFuncAttrs(kernel, "heterogeneous_dispatch (combined, task_type=0/1)");

  // ---- Prepare test data ----
  const int N = 128 * 128;
  std::vector<float> h_a(N), h_b(N), h_out(N, -999.0f);
  for (int i = 0; i < N; ++i) {
    h_a[i] = (float)(i % 7) * 0.1f;
    h_b[i] = (float)(i % 5) * 0.2f;
  }
  // reference for branch A: C = A * B  (128x128 matmul, both loaded from
  // offset [0,0], row-major)
  std::vector<float> ref_c(N, 0.0f);
  for (int r = 0; r < 128; ++r) {
    for (int c = 0; c < 128; ++c) {
      float acc = 0.0f;
      for (int k = 0; k < 128; ++k) {
        acc += h_a[r * 128 + k] * h_b[k * 128 + c];
      }
      ref_c[r * 128 + c] = acc;
    }
  }
  // reference for branch B: sum of first 256 elements of in_a
  float ref_sum = 0.0f;
  for (int i = 0; i < 256; ++i) {
    ref_sum += h_a[i];
  }

  CUdeviceptr d_a, d_b, d_out;
  CUDA_CHECK(cuMemAlloc(&d_a, sizeof(float) * N));
  CUDA_CHECK(cuMemAlloc(&d_b, sizeof(float) * N));
  CUDA_CHECK(cuMemAlloc(&d_out, sizeof(float) * N));
  CUDA_CHECK(cuMemcpyHtoD(d_a, h_a.data(), sizeof(float) * N));
  CUDA_CHECK(cuMemcpyHtoD(d_b, h_b.data(), sizeof(float) * N));

  CUstream stream;
  CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  // ---- task_type = 0 : branch A (matmul) ----
  {
    CUDA_CHECK(cuMemsetD32(d_out, 0xdeadbeef, N));
    int32_t task_type = 0;
    void *args[] = {&task_type, &d_a, &d_b, &d_out};
    CUDA_CHECK(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream, args, NULL));
    CUDA_CHECK(cuCtxSynchronize());
    CUDA_CHECK(cuMemcpyDtoH(h_out.data(), d_out, sizeof(float) * N));
    double maxAbsErr = 0.0;
    for (int i = 0; i < N; ++i) {
      double e = fabs((double)h_out[i] - (double)ref_c[i]);
      if (e > maxAbsErr) maxAbsErr = e;
    }
    printf("[task_type=0, branch A / mmaf 128x128] max_abs_err=%g  sample "
           "out[0]=%f ref[0]=%f  out[16383]=%f ref[16383]=%f\n",
           maxAbsErr, h_out[0], ref_c[0], h_out[N - 1], ref_c[N - 1]);
    printf("[task_type=0] RESULT: %s\n", maxAbsErr < 1e-2 ? "PASS" : "FAIL");
  }

  // ---- task_type = 1 : branch B (reduce) ----
  {
    CUDA_CHECK(cuMemsetD32(d_out, 0xdeadbeef, N));
    int32_t task_type = 1;
    void *args[] = {&task_type, &d_a, &d_b, &d_out};
    CUDA_CHECK(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream, args, NULL));
    CUDA_CHECK(cuCtxSynchronize());
    float out0;
    CUDA_CHECK(cuMemcpyDtoH(&out0, d_out, sizeof(float)));
    double err = fabs((double)out0 - (double)ref_sum);
    printf("[task_type=1, branch B / reduce sum256] out[0]=%f ref_sum=%f "
           "abs_err=%g\n",
           out0, ref_sum, err);
    printf("[task_type=1] RESULT: %s\n", err < 1e-2 ? "PASS" : "FAIL");
  }

  CUDA_CHECK(cuModuleUnload(cuModule));
  CUDA_CHECK(cuCtxDestroy(cuContext));
  return 0;
}
