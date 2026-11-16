#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <stdexcept>
#include <vector>

void Check(cudaError_t result) {
  if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
}
__global__ void moe_profile_probe(float const* input,float* output,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count)output[i]=input[i]*2.0f+1.0f;
}
int main()try {
  constexpr int n=262144;
  float *a,*b;
  Check(cudaMalloc(&a,n*sizeof(float)));Check(cudaMalloc(&b,n*sizeof(float)));
  std::vector<float> values(n,1.0f);
  Check(cudaMemcpy(a,values.data(),n*sizeof(float),cudaMemcpyHostToDevice));
  for(int i=0;i<5;++i)moe_profile_probe<<<n/256,256>>>(a,b,n);
  Check(cudaDeviceSynchronize());Check(cudaProfilerStart());
  for(int i=0;i<20;++i)moe_profile_probe<<<n/256,256>>>(a,b,n);
  Check(cudaDeviceSynchronize());Check(cudaProfilerStop());
  Check(cudaMemcpy(values.data(),b,n*sizeof(float),cudaMemcpyDeviceToHost));
  for(float value:values)if(value!=3.0f)throw std::runtime_error("canary output differs");
  Check(cudaFree(a));Check(cudaFree(b));
  std::puts("CUDA_PROFILING_CANARY_PASS");return 0;
}catch(std::exception const& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
