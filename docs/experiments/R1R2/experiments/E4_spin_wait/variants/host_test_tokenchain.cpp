#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define CUDA_CHECK(call) do { CUresult err = call; if (err != CUDA_SUCCESS) { const char *e; cuGetErrorString(err,&e); fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,e); exit(1);} } while(0)
static const int N = 262144;
int main(int argc, char**argv){
  const char* cubinPath = argv[1];
  const char* entryName = argv[2];
  int iterations = argc>3? atoi(argv[3]) : 1000;
  CUdevice dev; CUcontext ctx;
  CUDA_CHECK(cuInit(0)); CUDA_CHECK(cuDeviceGet(&dev,0)); CUDA_CHECK(cuCtxCreate(&ctx,NULL,0,dev));
  CUmodule mod; CUfunction fn;
  CUDA_CHECK(cuModuleLoad(&mod,cubinPath));
  CUDA_CHECK(cuModuleGetFunction(&fn,mod,entryName));
  double ref_sum_d=0.0; for(int i=0;i<N;i++) ref_sum_d += (double)(float)i;
  float ref_sum=(float)ref_sum_d;
  CUdeviceptr d_data,d_flag,d_checksum;
  CUDA_CHECK(cuMemAlloc(&d_data,sizeof(float)*N));
  CUDA_CHECK(cuMemAlloc(&d_flag,sizeof(int)));
  CUDA_CHECK(cuMemAlloc(&d_checksum,sizeof(float)));
  CUstream stream; CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  int pass=0, fail=0, distinct_wrong=0;
  float min_c=1e30f, max_c=-1e30f;
  for(int it=0; it<iterations; it++){
    CUDA_CHECK(cuMemsetD32(d_data,0xdeadbeef,N));
    CUDA_CHECK(cuMemsetD32(d_flag,0,1));
    CUDA_CHECK(cuMemsetD32(d_checksum,0,1));
    void* args[]={&d_data,&d_flag,&d_checksum};
    CUDA_CHECK(cuLaunchKernel(fn,2,1,1,1,1,1,0,stream,args,NULL));
    CUDA_CHECK(cuCtxSynchronize());
    float checksum; CUDA_CHECK(cuMemcpyDtoH(&checksum,d_checksum,sizeof(float)));
    if(checksum<min_c) min_c=checksum;
    if(checksum>max_c) max_c=checksum;
    double relerr = fabs((double)checksum-(double)ref_sum)/fabs((double)ref_sum);
    // Generous tolerance for float32 parallel-reduction-order rounding (~16 ULP at this magnitude).
    if (relerr < 1e-5) pass++; else { fail++; if(fail<=10) printf("[iter %d] MISMATCH checksum=%f ref=%f relerr=%g\n", it, checksum, ref_sum, relerr);}
  }
  printf("[%s] over %d iterations: PASS=%d FAIL=%d observed_range=[%f,%f] ref=%f\n", entryName, iterations, pass, fail, min_c, max_c, ref_sum);
  return 0;
}
