// SPDX-License-Identifier: BSD-3-Clause
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Tick=unsigned long long;
constexpr int kGrid=132,kShared=232448;
void Check(cudaError_t c) {
  if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));
}
template<class T> struct Device {
  T* data=nullptr;
  explicit Device(std::size_t count) {Check(cudaMalloc(&data,count*sizeof(T)));}
  ~Device() {cudaFree(data);}
};
__device__ __forceinline__ Tick Clock() {
  Tick v;asm volatile("mov.u64 %0, %%clock64;":"=l"(v)::"memory");return v;
}
__device__ __forceinline__ Tick Timer() {
  Tick v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v)::"memory");return v;
}
__global__ void Calibrate(Tick* ticks) {
  Tick start=Timer(),clock=Clock();while(Timer()-start<20000000ull){}
  ticks[0]=Clock()-clock;ticks[1]=Timer()-start;
}
__global__ void Flush(uint4* dst,std::size_t count) {
  for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
    dst[i]=make_uint4(unsigned(i),17,29,43);
}

// Generated straight-line asm lives with its build, never in the repository.
template<int KB> __device__ __noinline__ float Work(float);
#include "dnn4_icache_ops.inc"

template<int KB>
__global__ __launch_bounds__(256,1) void CodeProbe(int passes,Tick* stamps,float* sink) {
  float value=float(threadIdx.x)*.001f;
  for(int pass=0;pass<passes;++pass) {
    __syncthreads();
    Tick start=Clock();value=Work<KB>(value);Tick end=Clock();
    if(threadIdx.x==0 && pass+1==passes)stamps[blockIdx.x]=end-start;
  }
  sink[blockIdx.x*256+threadIdx.x]=value;
}
__global__ __launch_bounds__(256,1) void Evict(float* sink) {
  sink[blockIdx.x*256+threadIdx.x]=Work<1024>(float(threadIdx.x)*.001f);
}

template<int KB>
void Run() {
  Device<Tick> ticks(2),stamps(kGrid);Device<float> sink(kGrid*256);
  Device<uint4> flush((128u<<20)/sizeof(uint4));
  Calibrate<<<1,1>>>(ticks.data);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  Tick calibration[2];Check(cudaMemcpy(calibration,ticks.data,sizeof(calibration),cudaMemcpyDeviceToHost));
  double cycles_per_ns=double(calibration[0])/calibration[1];
  Check(cudaFuncSetAttribute(CodeProbe<KB>,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
  Check(cudaFuncSetAttribute(Evict,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
  auto read=[&]() {
    Check(cudaGetLastError());Check(cudaDeviceSynchronize());
    std::vector<Tick> values(kGrid);
    Check(cudaMemcpy(values.data(),stamps.data,kGrid*sizeof(Tick),cudaMemcpyDeviceToHost));
    std::sort(values.begin(),values.end());
    return (double(values[65])+double(values[66]))/2/cycles_per_ns;
  };
  CodeProbe<KB><<<kGrid,256,kShared>>>(2,stamps.data,sink.data);double hot=read();
  Flush<<<kGrid,256>>>(flush.data,(128u<<20)/sizeof(uint4));Check(cudaGetLastError());
  Evict<<<kGrid,256,kShared>>>(sink.data);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  CodeProbe<KB><<<kGrid,256,kShared>>>(1,stamps.data,sink.data);double cold=read();
  Evict<<<kGrid,256,kShared>>>(sink.data);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  CodeProbe<KB><<<kGrid,256,kShared>>>(1,stamps.data,sink.data);double l2=read();
  std::cout<<std::setprecision(17)<<"{\"metric\":\"icache\",\"code_kb\":"<<KB
      <<",\"hot_ns\":"<<hot<<",\"cold_ns\":"<<cold<<",\"l2_hot_ns\":"<<l2
      <<",\"p_cold_us_per_kb\":"<<(cold-hot)/1000/KB
      <<",\"p_l2_us_per_kb\":"<<(l2-hot)/1000/KB
      <<",\"cycles_per_ns\":"<<cycles_per_ns<<",\"ctas\":132,\"threads\":256}\n";
}
int main(int argc,char** argv)try {
  if(!std::getenv("TILEMEGA_RUN_NONCE") || !std::getenv("TILEMEGA_GPU_LOCK_HELD"))
    throw std::runtime_error("probes require the scheduler and guard");
  if(argc!=2)throw std::runtime_error("usage: probe {8|32|128|512}");
  switch(std::atoi(argv[1])) {
    case 8:Run<8>();break;case 32:Run<32>();break;
    case 128:Run<128>();break;case 512:Run<512>();break;
    default:throw std::runtime_error("invalid code size");
  }
  return 0;
}catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
