// Section 3.5: R2-D launched a Tile IR kernel with cuLaunchCooperativeKernel and
// blockDim=256, but the cuda-tile README says Tile IR kernels require block dims
// (1,1,1) ("block dims: unused, must be (1,1,1)") and v1_host.cpp uses (1,1,1).
// What does the driver actually do with a non-(1,1,1) blockDim, and is R2-D's
// admission result (grid=170 accepted, 171 rejected) still valid?
#include <cuda.h>
#include <cstdio>
#include <vector>
#define CK(x) do{CUresult r=(x); if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);\
  printf("    %-46s -> %s\n", #x, s);} }while(0)
int main(int argc,char**argv){
  cuInit(0); CUdevice d; cuDeviceGet(&d,0); CUcontext c; cuDevicePrimaryCtxRetain(&c,d); cuCtxSetCurrent(c);
  CUmodule m; if(cuModuleLoad(&m,argv[1])!=CUDA_SUCCESS){printf("load fail\n");return 1;}
  CUfunction f; cuModuleGetFunction(&f,m,argv[2]);
  int nsm=0; cuDeviceGetAttribute(&nsm,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,d);
  int reg=0,shm=0,maxt=0;
  cuFuncGetAttribute(&reg,CU_FUNC_ATTRIBUTE_NUM_REGS,f);
  cuFuncGetAttribute(&shm,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f);
  cuFuncGetAttribute(&maxt,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,f);
  printf("  SMs=%d  REG=%d  SHM=%d  MAX_THREADS_PER_BLOCK=%d\n",nsm,reg,shm,maxt);
  printf("\n  cuOccupancyMaxActiveBlocksPerMultiprocessor as a function of the\n"
         "  blockDim the HOST passes (the cubin's own REQNTID is 128):\n");
  for(int bd : {1,32,64,128,256,512}){
    int occ=-1; CUresult r=cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ,f,bd,0);
    const char*s="ok"; if(r!=CUDA_SUCCESS) cuGetErrorString(r,&s);
    printf("    blockDim=%-5d occupancy/SM=%-4d total=%-6d %s\n",bd,occ,occ*nsm,s);
  }
  // Cooperative admission limit, probed by bisection, for each blockDim.
  printf("\n  cuLaunchCooperativeKernel admission limit (max grid accepted):\n");
  CUdeviceptr p; cuMemAlloc(&p, 1<<22); cuMemsetD32(p,0,(1<<22)/4);
  void* a[3]={&p,&p,&p}; void** ka=a;
  for(int bd : {1,128,256}){
    int lo=1,hi=4096,best=0;
    while(lo<=hi){int mid=(lo+hi)/2;
      CUresult r=cuLaunchCooperativeKernel(f,mid,1,1,bd,1,1,0,0,ka);
      if(r==CUDA_SUCCESS){cuCtxSynchronize();best=mid;lo=mid+1;} else {hi=mid-1;}
    }
    printf("    blockDim=%-5d max cooperative grid = %d\n",bd,best);
  }
  return 0;
}
