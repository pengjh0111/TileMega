#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#define CUDA_CHECK(call) do { CUresult err = call; if (err != CUDA_SUCCESS) { const char *e; cuGetErrorString(err,&e); fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,e); exit(1);} } while(0)
static const int N = 262144;
int main(int argc, char**argv){
  int gridSize = atoi(argv[1]);
  CUdevice dev; CUcontext ctx;
  CUDA_CHECK(cuInit(0)); CUDA_CHECK(cuDeviceGet(&dev,0)); CUDA_CHECK(cuCtxCreate(&ctx,NULL,0,dev));
  CUmodule mod; CUfunction fn;
  CUDA_CHECK(cuModuleLoad(&mod,"spin_wait_tokenchain.cubin"));
  CUDA_CHECK(cuModuleGetFunction(&fn,mod,"spin_wait_tokenchain"));
  CUdeviceptr d_data,d_flag,d_checksum;
  CUDA_CHECK(cuMemAlloc(&d_data,sizeof(float)*N));
  CUDA_CHECK(cuMemAlloc(&d_flag,sizeof(int)));
  CUDA_CHECK(cuMemAlloc(&d_checksum,sizeof(float)));
  CUDA_CHECK(cuMemsetD32(d_data,0xdeadbeef,N));
  CUDA_CHECK(cuMemsetD32(d_flag,0,1));
  CUDA_CHECK(cuMemsetD32(d_checksum,0,1));
  void*args[]={&d_data,&d_flag,&d_checksum};
  printf("launching grid=%d ...\n", gridSize); fflush(stdout);
  CUDA_CHECK(cuLaunchKernel(fn,gridSize,1,1,1,1,1,0,0,args,NULL));
  CUDA_CHECK(cuCtxSynchronize());
  printf("kernel completed without hang.\n");
  float checksum; CUDA_CHECK(cuMemcpyDtoH(&checksum,d_checksum,sizeof(float)));
  printf("checksum(from block-1, other consumer blocks not checked)=%f\n", checksum);
  return 0;
}
