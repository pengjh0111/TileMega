#include <cuda.h>
#include <cstdio>
#include <initializer_list>
int main(int argc,char**argv){
  cuInit(0);CUdevice d;cuDeviceGet(&d,0);CUcontext c;cuDevicePrimaryCtxRetain(&c,d);cuCtxSetCurrent(c);
  CUmodule m;cuModuleLoad(&m,argv[1]);CUfunction f;cuModuleGetFunction(&f,m,argv[2]);
  CUdeviceptr p;cuMemAlloc(&p,1<<22);cuMemsetD32(p,0,(1<<22)/4);
  void*a[3]={&p,&p,&p};
  for(int bd:{1,2,128,256}){
    const char*s1;const char*s2;
    CUresult r1=cuLaunchKernel(f,8,1,1,bd,1,1,0,0,a,0); cuGetErrorString(r1,&s1);
    if(r1==CUDA_SUCCESS) cuCtxSynchronize();
    CUresult r2=cuLaunchCooperativeKernel(f,8,1,1,bd,1,1,0,0,a); cuGetErrorString(r2,&s2);
    if(r2==CUDA_SUCCESS) cuCtxSynchronize();
    printf("  blockDim=%-4d cuLaunchKernel: %-28s cuLaunchCooperativeKernel: %s\n",bd,s1,s2);
  }
  return 0;}
