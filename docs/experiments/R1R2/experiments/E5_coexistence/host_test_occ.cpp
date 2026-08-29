// E5: does the entry-level `occupancy` optimization_hints field change the
// driver-reported occupancy and/or the grid-oversubscription deadlock threshold
// found in E4 section 10?
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(x) do { CUresult _e = (x); if (_e != CUDA_SUCCESS) { \
  const char *s=nullptr; cuGetErrorString(_e,&s); \
  fprintf(stderr, "%s:%d CUDA error %d: %s\n", __FILE__, __LINE__, _e, s?s:"?"); \
  exit(1);} } while(0)

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s cubin entry gridSize [timeoutMode:0/1]\n", argv[0]); return 1; }
  const char* cubinPath = argv[1];
  const char* entryName = argv[2];
  int gridSize = atoi(argv[3]);

  CUDA_CHECK(cuInit(0));
  CUdevice dev; CUDA_CHECK(cuDeviceGet(&dev, 0));
  int smCount; CUDA_CHECK(cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  CUcontext ctx; CUDA_CHECK(cuCtxCreate(&ctx, NULL, 0, dev));

  CUmodule mod; CUDA_CHECK(cuModuleLoad(&mod, cubinPath));
  CUfunction fn; CUDA_CHECK(cuModuleGetFunction(&fn, mod, entryName));

  int regs=0, shared=0;
  cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);
  cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn);

  int numBlocksPerSm = -1;
  CUresult occErr = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, fn, 1, 0);
  printf("[%s] SM=%d REGS=%d SHARED=%d cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=1)-> err=%d numBlocksPerSm=%d (theoretical max coop grid=%d)\n",
         entryName, smCount, regs, shared, occErr, numBlocksPerSm, numBlocksPerSm*smCount);

  const int N = 262144;
  CUdeviceptr d_data, d_flag, d_checksum;
  CUDA_CHECK(cuMemAlloc(&d_data, N*sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_flag, sizeof(int)));
  CUDA_CHECK(cuMemAlloc(&d_checksum, sizeof(float)*gridSize)); // room enough, only [1] used really but harmless
  CUDA_CHECK(cuMemsetD32(d_flag, 0, 1));

  void* args[] = { &d_data, &d_flag, &d_checksum };
  printf("launching grid=%d ...\n", gridSize); fflush(stdout);
  CUDA_CHECK(cuLaunchKernel(fn, gridSize,1,1, 1,1,1, 0,0,args,NULL));
  CUresult syncErr = cuCtxSynchronize();
  if (syncErr != CUDA_SUCCESS) {
    const char* s=nullptr; cuGetErrorString(syncErr,&s);
    printf("cuCtxSynchronize FAILED: %s\n", s?s:"?");
    return 2;
  }
  printf("kernel completed without hang.\n");
  return 0;
}
