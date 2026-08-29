#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#define CK(x) do{ CUresult _e=(x); if(_e!=CUDA_SUCCESS){const char*s;cuGetErrorString(_e,&s);fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,s);exit(1);} }while(0)
int main(int argc, char**argv){
  const char* cubinPath = argv[1];
  const char* entryName = argv[2];
  CK(cuInit(0));
  CUdevice dev; CK(cuDeviceGet(&dev,0));
  CUcontext ctx; CK(cuCtxCreate(&ctx,NULL,0,dev));
  CUmodule mod; CK(cuModuleLoad(&mod,cubinPath));
  CUfunction fn; CK(cuModuleGetFunction(&fn,mod,entryName));
  int val;
  CK(cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn));
  printf("MAX_THREADS_PER_BLOCK = %d\n", val);
  CK(cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
  printf("NUM_REGS = %d\n", val);
  CK(cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn));
  printf("SHARED_SIZE_BYTES = %d\n", val);
  CK(cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn));
  printf("LOCAL_SIZE_BYTES = %d\n", val);
  int sm_count; CK(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  printf("SM_COUNT = %d\n", sm_count);

  for (int bs : {1, 32, 128, 256}) {
    int numBlocks=-1;
    CUresult e = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, fn, bs, 0);
    printf("cuOccupancyMaxActiveBlocksPerMultiprocessor(blockSize=%d) -> err=%d numBlocksPerSm=%d (theoretical max coop grid=%d)\n",
           bs, e, numBlocks, numBlocks*sm_count);
  }
  return 0;
}
