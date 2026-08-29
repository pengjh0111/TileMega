#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#define CUDA_CHECK(call) do { CUresult err = call; if (err != CUDA_SUCCESS) { const char *e; cuGetErrorString(err,&e); fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,e); exit(1);} } while(0)
// R2-D follow-up: same as host_test_coop.cpp but using the CORRECT blockDim=(256,1,1)
// (matching the cubin's EIATTR_REQNTID / MAX_THREADS_PER_BLOCK) for BOTH the occupancy
// query and the actual cuLaunchCooperativeKernel call, instead of the (1,1,1) mismatch
// used throughout round 1 (E4/E5) and in host_test_coop.cpp.
int main(int argc, char**argv){
  int gridSize = atoi(argv[1]);
  const char* cubinPath = argv[2];
  const char* entryName = argv[3];
  CUdevice dev; CUcontext ctx;
  CUDA_CHECK(cuInit(0)); CUDA_CHECK(cuDeviceGet(&dev,0)); CUDA_CHECK(cuCtxCreate(&ctx,NULL,0,dev));
  CUmodule mod; CUfunction fn;
  CUDA_CHECK(cuModuleLoad(&mod,cubinPath));
  CUDA_CHECK(cuModuleGetFunction(&fn,mod,entryName));

  int maxThreads=0;
  CUDA_CHECK(cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn));
  printf("MAX_THREADS_PER_BLOCK (from cubin) = %d -- using this as blockDim.x\n", maxThreads);

  int numBlocksPerSm = 0;
  CUresult occErr = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, fn, maxThreads, 0);
  int smCount = 0;
  cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
  printf("cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=%d) -> err=%d numBlocksPerSm=%d  (SM count=%d, so max coop grid = %d)\n",
         maxThreads, occErr, numBlocksPerSm, smCount, numBlocksPerSm*smCount);

  CUdeviceptr d_data,d_flag,d_checksum;
  CUDA_CHECK(cuMemAlloc(&d_data,sizeof(float)*262144));
  CUDA_CHECK(cuMemAlloc(&d_flag,sizeof(int)));
  CUDA_CHECK(cuMemAlloc(&d_checksum,sizeof(float)));
  CUDA_CHECK(cuMemsetD32(d_data,0xdeadbeef,262144));
  CUDA_CHECK(cuMemsetD32(d_flag,0,1));
  CUDA_CHECK(cuMemsetD32(d_checksum,0,1));
  void*args[]={&d_data,&d_flag,&d_checksum};

  printf("attempting cuLaunchCooperativeKernel with grid=%d, blockDim=(%d,1,1) ...\n", gridSize, maxThreads); fflush(stdout);
  CUresult err = cuLaunchCooperativeKernel(fn, gridSize,1,1, maxThreads,1,1, 0, 0, args);
  if (err != CUDA_SUCCESS) {
    const char* estr; cuGetErrorString(err,&estr);
    printf("cuLaunchCooperativeKernel REJECTED at launch time: %s\n", estr);
    return 0;
  }
  printf("cuLaunchCooperativeKernel ACCEPTED the launch, now syncing...\n"); fflush(stdout);
  CUDA_CHECK(cuCtxSynchronize());
  printf("kernel completed without hang.\n");
  return 0;
}
