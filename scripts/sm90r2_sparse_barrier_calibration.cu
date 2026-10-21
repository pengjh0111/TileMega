// SPDX-License-Identifier: BSD-3-Clause
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

using U64=unsigned long long;
void Check(cudaError_t error) {if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
__device__ U64 Now() {U64 value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;}

__global__ __launch_bounds__(160) void Sample(U64* counter,U64* stamps,
    int repeats,int active,int delay_ns) {
  for(int repeat=0;repeat<repeats;++repeat) {
    // Inactive CTAs arrive early, as in a native stage with fewer tasks than
    // its resident grid. Delay is an experimental coordinate, not a fit value.
    if(threadIdx.x==0 && int(blockIdx.x)<active) {
      U64 begin=Now();while(Now()-begin<U64(delay_ns)) {}
    }
    __syncthreads();
    if(threadIdx.x==0) {
      stamps[(repeat*gridDim.x+blockIdx.x)*2]=Now();
      asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(counter):"memory");
      U64 seen;
      do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(counter):"memory");}
      while(seen<U64(repeat+1)*gridDim.x);
    }
    __syncthreads();
    if(threadIdx.x==0)stamps[(repeat*gridDim.x+blockIdx.x)*2+1]=Now();
  }
}

void Print(char const* metric,int grid,int active,int delay,std::vector<U64> values) {
  std::sort(values.begin(),values.end());
  std::printf("%s,%d,%d,%d,%zu,%llu,%llu,ns\n",metric,grid,active,delay,values.size(),
      values[values.size()/2],values[values.size()*99/100]);
}
int main() {try {
  Check(cudaSetDevice(0));cudaDeviceProp p{};Check(cudaGetDeviceProperties(&p,0));
  if(p.major!=9 || p.minor!=0)throw std::runtime_error("sparse V3 calibration requires Hopper");
  constexpr int repeats=256;
  U64 *counter,*stamps;Check(cudaMalloc(&counter,8));
  Check(cudaMalloc(&stamps,2*repeats*p.multiProcessorCount*2*sizeof(U64)));
  std::printf("metric,grid,active_ctas,delay_ns,samples,median,p99,unit\n");
  for(int residency:{1,2}) {
    int shared=int(std::min(p.sharedMemPerBlockOptin,p.sharedMemPerMultiprocessor/residency)/256)*256;
    Check(cudaFuncSetAttribute(Sample,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    int occupancy=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,Sample,160,shared));
    if(occupancy!=residency)throw std::runtime_error("sparse V3 residency mismatch");
    int grid=p.multiProcessorCount*residency;
    for(int active:{1,16,64,128,grid})for(int delay:{0,2000,8000}) {
      Check(cudaMemset(counter,0,8));Sample<<<grid,160,shared>>>(counter,stamps,8,active,delay);
      Check(cudaDeviceSynchronize());Check(cudaMemset(counter,0,8));
      Sample<<<grid,160,shared>>>(counter,stamps,repeats,active,delay);
      std::vector<U64> data(2*repeats*grid);Check(cudaMemcpy(data.data(),stamps,data.size()*8,cudaMemcpyDeviceToHost));
      U64 count=0;Check(cudaMemcpy(&count,counter,8,cudaMemcpyDeviceToHost));
      if(count!=U64(repeats)*grid)throw std::runtime_error("incomplete V3 arrival count");
      std::vector<U64> tails,departures;
      for(int repeat=8;repeat<repeats;++repeat) {
        U64 arrive=0,leave=0;
        for(int cta=0;cta<grid;++cta) {
          arrive=std::max(arrive,data[(repeat*grid+cta)*2]);
          leave=std::max(leave,data[(repeat*grid+cta)*2+1]);
        }
        if(leave<arrive)throw std::runtime_error("inconsistent global timestamps");
        tails.push_back(leave-arrive);
        for(int cta=0;cta<grid;++cta) {
          U64 end=data[(repeat*grid+cta)*2+1];
          if(end<arrive)throw std::runtime_error("CTA left before final arrival");
          departures.push_back(end-arrive);
        }
      }
      Print("v3_sparse_last_arrival_to_last_departure",grid,active,delay,tails);
      Print("v3_sparse_last_arrival_to_cta_departure",grid,active,delay,departures);
    }
  }
  Check(cudaFree(stamps));Check(cudaFree(counter));return 0;
}catch(std::exception const& error) {std::fprintf(stderr,"%s\n",error.what());return 2;}}
