// v4 = v2 + a plain host-side usleep(50ms) before the kernel launch (no GPU memset),
// to test whether pure elapsed time (vs. actual GPU-side warm-up work) is what matters.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#define CUDA_CHECK(x) do { CUresult _e = (x); if (_e != CUDA_SUCCESS) { \
  const char *s=nullptr; cuGetErrorString(_e,&s); \
  fprintf(stderr, "%s:%d CUDA error %d: %s\n", __FILE__, __LINE__, _e, s?s:"?"); \
  exit(1);} } while(0)
int main(int argc, char** argv) {
  const char* cubinPath = argv[1];
  const char* entryName = argv[2];
  int gridSize = atoi(argv[3]);
  CUDA_CHECK(cuInit(0));
  CUdevice dev; CUDA_CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx; CUDA_CHECK(cuCtxCreate(&ctx, NULL, 0, dev));
  CUmodule mod; CUDA_CHECK(cuModuleLoad(&mod, cubinPath));
  CUfunction fn; CUDA_CHECK(cuModuleGetFunction(&fn, mod, entryName));
  const int N = 262144;
  CUdeviceptr d_data, d_flag, d_checksum;
  CUDA_CHECK(cuMemAlloc(&d_data, N*sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_flag, sizeof(int)));
  CUDA_CHECK(cuMemAlloc(&d_checksum, sizeof(float)));
  CUDA_CHECK(cuMemsetD32(d_flag, 0, 1));
  usleep(50000); // 50ms pure host-side sleep, no GPU work
  void* args[] = { &d_data, &d_flag, &d_checksum };
  printf("launching grid=%d ...\n", gridSize); fflush(stdout);
  CUDA_CHECK(cuLaunchKernel(fn, gridSize,1,1, 1,1,1, 0,0,args,NULL));
  CUresult syncErr = cuCtxSynchronize();
  if (syncErr != CUDA_SUCCESS) { printf("sync FAILED\n"); return 2; }
  printf("kernel completed without hang.\n");
  return 0;
}
