#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#define CUDA_CHECK(call) do { CUresult err = call; if (err != CUDA_SUCCESS) { const char *e; cuGetErrorString(err,&e); fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,e); exit(1);} } while(0)
int main(int argc, char**argv){
  int gridSize = atoi(argv[1]);
  CUdevice dev; CUcontext ctx;
  CUDA_CHECK(cuInit(0)); CUDA_CHECK(cuDeviceGet(&dev,0)); CUDA_CHECK(cuCtxCreate(&ctx,NULL,0,dev));
  CUmodule mod; CUfunction fn;
  CUDA_CHECK(cuModuleLoad(&mod,"spin_wait_tokenchain.cubin"));
  CUDA_CHECK(cuModuleGetFunction(&fn,mod,"spin_wait_tokenchain"));

  // Query the driver's own occupancy calculation for this exact kernel/blockDim.
  int numBlocksPerSm = 0;
  CUresult occErr = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, fn, 1, 0);
  int smCount = 0;
  cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
  printf("cuOccupancyMaxActiveBlocksPerMultiprocessor -> err=%d numBlocksPerSm=%d  (SM count=%d, so max coop grid = %d)\n",
         occErr, numBlocksPerSm, smCount, numBlocksPerSm*smCount);

  CUdeviceptr d_data,d_flag,d_checksum;
  CUDA_CHECK(cuMemAlloc(&d_data,sizeof(float)*262144));
  CUDA_CHECK(cuMemAlloc(&d_flag,sizeof(int)));
  CUDA_CHECK(cuMemAlloc(&d_checksum,sizeof(float)));
  CUDA_CHECK(cuMemsetD32(d_data,0xdeadbeef,262144));
  CUDA_CHECK(cuMemsetD32(d_flag,0,1));
  CUDA_CHECK(cuMemsetD32(d_checksum,0,1));
  void*args[]={&d_data,&d_flag,&d_checksum};

  printf("attempting cuLaunchCooperativeKernel with grid=%d ...\n", gridSize); fflush(stdout);
  CUresult err = cuLaunchCooperativeKernel(fn, gridSize,1,1, 1,1,1, 0, 0, args);
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
