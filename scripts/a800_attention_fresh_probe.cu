// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseAttentionBody.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace tilemega;
using Tick=unsigned long long;

static void Check(cudaError_t code) {
  if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}

template<int D,int Q,int Norm,bool Fresh>
__global__ void Measure(Tick* samples,float* output) {
  using Body=backend::Sm90DenseAttentionBody<1,8,D,Q,Norm,1,64,2,0,1088,false,false,true>;
  extern __shared__ __align__(1024) char bytes[];
  auto& shared=*reinterpret_cast<typename Body::Storage*>(bytes);
  int tid=threadIdx.x,warp=tid/32;
  for(int i=tid;i<16*D;i+=128) {
    shared.query[i]=typename Body::Element(0.0625f);
    shared.NewKey()[i]=typename Body::Element(0.125f);
    shared.NewValue()[i]=typename Body::Element(0.25f);
  }
  for(int i=tid;i<64*D;i+=128) {
    shared.kv[0][0][i]=typename Body::Element(0.125f);
    shared.kv[0][1][i]=typename Body::Element(0.25f);
  }
  __syncthreads();
  unsigned query=Body::Async::Shared(shared.query);
  unsigned key=Body::Async::Shared(Fresh?shared.NewKey():shared.kv[0][0]);
  unsigned value=Body::Async::Shared(Fresh?shared.NewValue():shared.kv[0][1]);
  typename Body::State state{};
  state.maximum[0]=state.maximum[1]=-INFINITY;
  constexpr int warmup=16,rounds=256;
  for(int i=0;i<warmup+rounds;++i) {
    codegen::executor::ComputeSync();
    Tick start=clock64();
    if constexpr(Fresh) {
      if(warp==0)Body::template Block<16>(query,key,value,0,1,state);
    } else Body::template Block<64>(query,key,value,warp*16,64,state);
    codegen::executor::ComputeSync();
    if(tid==0 && i>=warmup)samples[i-warmup]=clock64()-start;
  }
  output[tid]=state.o[0][0];
}

__global__ void ClockRate(Tick* out) {
  auto stamp=[] {Tick value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;};
  Tick start=stamp(),cycles=clock64();
  Tick elapsed;
  do {elapsed=stamp()-start;} while(elapsed<10000000);
  out[0]=clock64()-cycles;
  out[1]=elapsed;
}

struct BarrierSpan {Tick begin,end;unsigned sm;};

__global__ __launch_bounds__(160,2) void CompactBarrier(BarrierSpan* spans) {
  extern __shared__ __align__(1024) char bytes[];
  if(threadIdx.x==0)*reinterpret_cast<volatile char*>(bytes)=0;
  if(threadIdx.x>=128)return;
  auto stamp=[] {
    Tick value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value)::"memory");return value;
  };
  codegen::executor::ComputeSync();
  Tick begin=stamp();
  for(int i=0;i<64;++i) {
    #pragma unroll
    for(int barrier=0;barrier<16;++barrier)codegen::executor::ComputeSync();
  }
  Tick end=stamp();
  if(threadIdx.x==0) {
    unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
    spans[blockIdx.x]={begin,end,sm};
  }
}

template<int Queries>
void CompactWorkspaceBarrier() {
  constexpr int grid=216,shared=backend::Sm80AttentionScratch::Layout(128,Queries,1,64,2,false,true).end;
  int residency=0;
  Check(cudaFuncSetAttribute(CompactBarrier,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residency,CompactBarrier,160,shared));
  if(residency!=2)throw std::runtime_error("unexpected compact barrier residency");
  BarrierSpan* device=nullptr;
  Check(cudaMalloc(&device,grid*sizeof(BarrierSpan)));
  std::vector<BarrierSpan> host(grid);
  std::vector<double> samples;
  for(int repeat=-1;repeat<9;++repeat) {
    CompactBarrier<<<grid,160,shared>>>(device);
    Check(cudaMemcpy(host.data(),device,grid*sizeof(BarrierSpan),cudaMemcpyDeviceToHost));
    if(repeat<0)continue;
    std::sort(host.begin(),host.end(),[](auto const& a,auto const& b) {
      return a.sm<b.sm;
    });
    for(int i=0;i<grid;i+=2)
      if(host[i].sm!=host[i+1].sm || (i && host[i-1].sm==host[i].sm) ||
          std::max(host[i].begin,host[i+1].begin)>=std::min(host[i].end,host[i+1].end))
        throw std::runtime_error("compact barrier probe did not overlap both CTAs");
    for(auto const& row:host)samples.push_back(double(row.end-row.begin)/1024);
  }
  Check(cudaFree(device));
  std::sort(samples.begin(),samples.end());
  std::printf("{\"status\":\"measured\",\"grid\":%d,\"dynamic_shared\":%d,\"residency\":%d,"
              "\"overlapping_sms\":108,\"consumer_threads\":128,\"iterations\":1024,"
              "\"samples\":%zu,\"barrier_ns\":%.3f,\"queries\":%d}\n",
              grid,shared,residency,samples.size(),samples[samples.size()/2],Queries);
}

template<int D,int Q,int Norm>
void Case() {
  using Body=backend::Sm90DenseAttentionBody<1,8,D,Q,Norm,1,64,2,0,1088,false,false,true>;
  constexpr int shared=sizeof(typename Body::Storage);
  Tick* samples=nullptr;
  float* output=nullptr;
  Check(cudaMalloc(&samples,256*sizeof(Tick)));
  Check(cudaMalloc(&output,128*sizeof(float)));
  ClockRate<<<1,1>>>(samples);
  Tick clock[2];
  Check(cudaMemcpy(clock,samples,sizeof(clock),cudaMemcpyDeviceToHost));
  double ns_per_tick=double(clock[1])/clock[0];
  auto run=[&](auto kernel) {
    Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    void* args[]={&samples,&output};
    Check(cudaLaunchKernel(reinterpret_cast<void const*>(kernel),1,128,args,shared));
    std::vector<Tick> host(256);
    float lane0=0;
    Check(cudaMemcpy(host.data(),samples,host.size()*sizeof(Tick),cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(&lane0,output,sizeof(lane0),cudaMemcpyDeviceToHost));
    if(!(lane0>0))throw std::runtime_error("attention probe did not produce a consumer output");
    std::sort(host.begin(),host.end());
    return host[host.size()/2]*ns_per_tick;
  };
  double full=run(Measure<D,Q,Norm,false>),fresh=run(Measure<D,Q,Norm,true>);
  std::printf("{\"status\":\"measured\",\"dim\":%d,\"queries\":%d,\"norm\":%d,"
      "\"full_ns\":%.3f,\"fresh_ns\":%.3f,\"samples\":256}\n",D,Q,Norm,full,fresh);
  Check(cudaFree(samples));Check(cudaFree(output));
}

int main(int argc,char** argv) {
  try {
    cudaDeviceProp device{};
    Check(cudaGetDeviceProperties(&device,0));
    if(device.major!=8 || device.minor!=0 || device.multiProcessorCount!=108 || argc!=2)
      throw std::invalid_argument("provide an attention shape on an A800");
#define CASE(D,Q,N) if(!std::strcmp(argv[1],"d" #D "-q" #Q "-norm" #N)) Case<D,Q,N>(); else
    CASE(128,1,2) CASE(128,2,1) CASE(128,3,0)
    CASE(128,4,0) CASE(128,8,1) CASE(64,4,0)
    if(!std::strcmp(argv[1],"compact-workspace-barrier"))CompactWorkspaceBarrier<4>();
    else if(!std::strcmp(argv[1],"compact-workspace-barrier-q3"))CompactWorkspaceBarrier<3>();
    else throw std::invalid_argument("unknown attention shape");
#undef CASE
    Check(cudaDeviceReset());std::this_thread::sleep_for(std::chrono::seconds(2));
    return 0;
  } catch(std::exception const& error) {
    std::fprintf(stderr,"%s\n",error.what());
    return 1;
  }
}
