
#include <cstdio>
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define HD __host__ __device__
#else
#define HD
#endif
template<bool T> HD int Evaluate() {
  int state=0;
  auto issue=[&](int) {
    if constexpr(T) {state+=1;}
    else for(int j=0;j<1;++j) {state+=2;}
    state+=8;
  };
  issue(0);
  return state;
}
#if defined(__CUDACC__)
__global__ void Inspect(int* result) {result[0]=Evaluate<true>();result[1]=Evaluate<false>();}
#endif
int main() {
  int a=Evaluate<true>(),b=Evaluate<false>();
  std::printf("host=%d,%d\n",a,b);
  if(a!=9 || b!=10)return 2;
#if defined(__CUDACC__)
  int* values=nullptr;
  if(cudaMallocManaged(&values,2*sizeof(int))!=cudaSuccess)return 4;
  Inspect<<<1,1>>>(values);
  if(cudaGetLastError()!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess)return 5;
  std::printf("device=%d,%d\n",values[0],values[1]);
  bool correct=values[0]==9 && values[1]==10;
  cudaFree(values);
  if(!correct)return 3;
#endif
}
