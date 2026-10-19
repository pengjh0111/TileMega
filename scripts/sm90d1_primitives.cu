// SPDX-License-Identifier: BSD-3-Clause
// Standalone diagnostic; no serving default or target coefficient is changed.
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using U64 = unsigned long long;
void Check(cudaError_t status) {
  if(status != cudaSuccess) {std::fprintf(stderr,"%s\n",cudaGetErrorString(status));std::exit(1);}
}
__device__ U64 Timer() {U64 value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;}
__global__ void Timers(U64* output,int samples) {
  if(threadIdx.x)return;
  U64 previous=Timer();
  for(int i=0;i<samples;++i) {auto next=Timer();output[i]=next-previous;previous=next;}
  U64 begin=Timer(),cycles=clock64();
  while(Timer()-begin<5000000ull)asm volatile("":::"memory");
  output[samples]=clock64()-cycles;output[samples+1]=Timer()-begin;
}
__global__ void AtomicTrial(U64* value,U64* output,int kind,int samples) {
  if(threadIdx.x)return;
  for(int i=0;i<samples;++i) {
    auto begin=clock64();
    if(kind==0) __threadfence();
    else if(kind==1) {
      cuda::atomic_ref<U64,cuda::thread_scope_device> counter(*value);
      counter.fetch_add(1,cuda::memory_order_acq_rel);
    }else asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(value):"memory");
    output[blockIdx.x*samples+i]=clock64()-begin;
  }
}
__global__ void V3Barrier(U64* counter,U64* output,int samples) {
  for(int i=0;i<samples;++i) {
    __syncthreads();
    if(threadIdx.x==0) {
      output[(i*gridDim.x+blockIdx.x)*2]=Timer();
      asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(counter):"memory");
      U64 value=0,begin=Timer(),target=U64(i+1)*gridDim.x;
      do {
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(value):"l"(counter):"memory");
        if(Timer()-begin>2000000000ull)asm volatile("trap;");
      }while(value<target);
      output[(i*gridDim.x+blockIdx.x)*2+1]=Timer();
    }
    __syncthreads();
  }
}
__global__ void PrefetchIssue(char const* input,U64* output,int bytes,int samples) {
  if(threadIdx.x)return;
  for(int i=0;i<samples;++i) {
    auto begin=clock64();
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"::"l"(input),"r"(bytes):"memory");
    output[i]=clock64()-begin;
  }
}
void Print(char const* name,std::vector<double> data,char const* unit,int grid=1,int bytes=0) {
  std::sort(data.begin(),data.end());
  std::printf("%s,%d,%d,%zu,%.9g,%.9g,%s\n",name,grid,bytes,data.size(),data[data.size()/2],
      data[std::min(data.size()-1,std::size_t(data.size()*.99))],unit);
}
int main() {
  constexpr int samples=4096;
  U64 *output,*value;char* input;
  Check(cudaMalloc(&output,264ull*samples*2*sizeof(U64)));Check(cudaMalloc(&value,8));
  Check(cudaMalloc(&input,256<<20));Check(cudaMemset(input,1,256<<20));
  std::vector<U64> host(264ull*samples*2);
  Timers<<<1,1>>>(output,65536);Check(cudaDeviceSynchronize());
  Check(cudaMemcpy(host.data(),output,65538*sizeof(U64),cudaMemcpyDeviceToHost));
  std::printf("name,grid_ctas,bytes,samples,median,p99,unit\n");
  std::vector<double> positive,all;
  for(int i=0;i<65536;++i) {all.push_back(host[i]);if(host[i])positive.push_back(host[i]);}
  Print("globaltimer_increment_all",all,"ns");
  Print("globaltimer_increment_nonzero",positive,"ns");
  Print("globaltimer_min_nonzero",{*std::min_element(positive.begin(),positive.end())},"ns");
  double cycles_per_ns=double(host[65536])/host[65537];
  Print("clock64_frequency",{cycles_per_ns*1000},"MHz");
  for(int kind=0;kind<3;++kind)for(int grid:{1,8,16}) {
    if(kind!=1 && grid!=1)continue;
    Check(cudaMemset(value,0,8));AtomicTrial<<<grid,128>>>(value,output,kind,samples);
    Check(cudaDeviceSynchronize());Check(cudaMemcpy(host.data(),output,grid*samples*8,cudaMemcpyDeviceToHost));
    std::vector<double> times;for(int i=0;i<grid*samples;++i)times.push_back(host[i]/cycles_per_ns);
    Print(kind==0?"threadfence":kind==1?"fetch_add_acq_rel":"red_release_gpu",times,"ns",grid);
  }
  for(int grid:{132,264}) {
    Check(cudaMemset(value,0,8));V3Barrier<<<grid,128>>>(value,output,64);
    Check(cudaDeviceSynchronize());Check(cudaMemcpy(host.data(),output,grid*64*16,cudaMemcpyDeviceToHost));
    std::vector<double> tails;
    for(int i=0;i<64;++i) {
      U64 last=0;for(int b=0;b<grid;++b)last=std::max(last,host[(i*grid+b)*2]);
      for(int b=0;b<grid;++b)tails.push_back(double(host[(i*grid+b)*2+1])-double(last));
    }
    Print("v3_last_arrival_to_departure",tails,"ns",grid);
  }
  for(int kib:{16,32,64,128,192}) {
    PrefetchIssue<<<1,1>>>(input,output,kib*1024,samples);Check(cudaDeviceSynchronize());
    Check(cudaMemcpy(host.data(),output,samples*8,cudaMemcpyDeviceToHost));
    std::vector<double> times;for(int i=0;i<samples;++i)times.push_back(host[i]/cycles_per_ns);
    Print("prefetch_issue_back_to_back",times,"ns",1,kib*1024);
  }
  Check(cudaFree(input));Check(cudaFree(value));Check(cudaFree(output));
}
