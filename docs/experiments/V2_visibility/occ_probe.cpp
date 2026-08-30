#include <initializer_list>
#include <cuda.h>
#include <cstdio>
int main(int argc,char**argv){
  cuInit(0); CUdevice d; cuDeviceGet(&d,0); CUcontext c; cuDevicePrimaryCtxRetain(&c,d); cuCtxSetCurrent(c);
  CUmodule m; if(cuModuleLoad(&m,argv[1])!=CUDA_SUCCESS){printf("load fail\n");return 1;}
  CUfunction f; if(cuModuleGetFunction(&f,m,argv[2])!=CUDA_SUCCESS){printf("no entry %s\n",argv[2]);return 1;}
  int nsm=0; cuDeviceGetAttribute(&nsm,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,d);
  int reg=0,shm=0; cuFuncGetAttribute(&reg,CU_FUNC_ATTRIBUTE_NUM_REGS,f);
  cuFuncGetAttribute(&shm,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f);
  printf("  %-28s REG=%-4d SHM=%-5d SMs=%d\n",argv[2],reg,shm,nsm);
  for(int bd : {1,128,256}){
    int occ=-1; cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ,f,bd,0);
    printf("    blockDim=%-4d occ/SM=%-4d  => coop grid limit = %d\n",bd,occ,occ*nsm);
  }
  return 0;
}
