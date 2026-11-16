// SPDX-License-Identifier: BSD-3-Clause
// DNN3 §9.2: resident operands, production K_PIPE_MMAS=1 main loop.
#include <tilemega/Backend/Sm90F3Gemm.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifndef F3_PROBE_TM
#define F3_PROBE_TM 128
#endif
#ifndef F3_PROBE_TN
#define F3_PROBE_TN 64
#endif
#ifndef F3_PROBE_GATHER
#define F3_PROBE_GATHER 0
#endif

using namespace tilemega::backend;
using U64=unsigned long long;
using Element=cutlass::bfloat16_t;
constexpr int kTM=F3_PROBE_TM,kTN=F3_PROBE_TN;
constexpr int kIterations=64,kSamples=72,kDiscard=8;
using Side=Sm90F3SideLayout<kTM,kTN,false,0,false>;
using Chain=Sm90F3Chain<Side>;
using Body=Sm90F3Gemm<kTM,kTN,2,1,
    F3_PROBE_GATHER?Sm90F3Mode::kGather:Sm90F3Mode::kNormal,
    kTM,kTN,64*kIterations,kTN,Chain>;

void Check(cudaError_t error) {
  if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
__device__ __forceinline__ U64 Clock() {
  U64 value;asm volatile("mov.u64 %0, %%clock64;":"=l"(value)::"memory");
  return value;
}
__device__ __forceinline__ U64 Timer() {
  U64 value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value)::"memory");
  return value;
}
__global__ void Calibrate(U64* output) {
  U64 start=Timer(),ticks=Clock();
  while(Timer()-start<20000000ull){}
  output[0]=Clock()-ticks;output[1]=Timer()-start;
}

__global__ __launch_bounds__(288,1) void ResidentMainLoop(U64* stamps,float* sink) {
  extern __shared__ __align__(1024) char shared[];
  auto* header=reinterpret_cast<Sm90F3Header*>(shared);
  auto& ring=header->rings[0];
  auto* arena=reinterpret_cast<Element*>(shared+Sm90F3Budget::kArena0);
  if(threadIdx.x==0){ring.Initialize();Sm90F3Async::Base::InitFence();}
  for(int i=int(threadIdx.x);i<2*Body::kSlotBytes/2;i+=288)arena[i]=Element(.125f);
  Sm90F3Async::Base::FenceProxyAsyncShared();
  __syncthreads();
  if(threadIdx.x<256 && (threadIdx.x&31)==0)ring.Prime();
  __syncthreads();
  Sm90F3Phase phase{};
  for(int sample=0;sample<kSamples;++sample) {
    __syncthreads();
    if(threadIdx.x>=256) {
      for(int k=0;k<kIterations;++k) {
        int slot=k%2;
        if(threadIdx.x==256) {
          Sm90F3Async::Base::Wait(ring.empty+slot,phase.Bit(phase.empty,slot));
          phase.empty^=1u<<slot;
        }
        __syncwarp();
        // The gather barrier needs all 32 lane arrivals and the leader's
        // transaction arrival even though this probe transfers no bytes.
        if constexpr(F3_PROBE_GATHER) {
          Sm90F3Async::Base::Arrive(ring.gfull+slot);
          if(threadIdx.x==256)Sm90F3Async::Base::Arrive(ring.gfull+slot);
        }else if(threadIdx.x==256)Sm90F3Async::Base::Arrive(ring.full+slot);
      }
    }else {
      Body::Accumulators accum;
      U64 start=Clock();
      Body::MainLoop(ring,arena,phase,accum,int(threadIdx.x)/128,0,kIterations);
      U64 end=Clock();
      if(threadIdx.x==0)stamps[sample]=end-start;
      sink[threadIdx.x]=accum.tile[0](0);
    }
  }
  __syncthreads();
  if(threadIdx.x==0)ring.Invalidate();
}

int main()try {
  if(!std::getenv("TILEMEGA_RUN_NONCE") ||
      !std::getenv("TILEMEGA_GPU_LOCK_HELD"))
    throw std::runtime_error("constant probes require the GPU scheduler and guard");
  U64* stamps=nullptr;float* sink=nullptr;
  Check(cudaMalloc(&stamps,kSamples*sizeof(U64)));
  Check(cudaMalloc(&sink,256*sizeof(float)));
  Calibrate<<<1,1>>>(stamps);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  U64 calibration[2];Check(cudaMemcpy(calibration,stamps,sizeof(calibration),cudaMemcpyDeviceToHost));
  double cycles_per_ns=double(calibration[0])/calibration[1];
  constexpr int bytes=Sm90F3Budget::kHeader+2*Body::kSlotBytes;
  Check(cudaFuncSetAttribute(ResidentMainLoop,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  int active=0;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,ResidentMainLoop,288,bytes));
  if(active<1)throw std::runtime_error("resident probe cannot launch one CTA");
  ResidentMainLoop<<<1,288,bytes>>>(stamps,sink);
  Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  std::vector<U64> raw(kSamples);Check(cudaMemcpy(raw.data(),stamps,kSamples*sizeof(U64),cudaMemcpyDeviceToHost));
  std::vector<float> values(256);Check(cudaMemcpy(values.data(),sink,256*sizeof(float),cudaMemcpyDeviceToHost));
  // Each output accumulates 4096 products of .125*.125, exactly 64.
  for(float value:values)if(value!=64.f)throw std::runtime_error("resident main-loop output mismatch");
  std::vector<double> ns;
  for(int i=kDiscard;i<kSamples;++i)ns.push_back(raw[i]/cycles_per_ns/kIterations);
  std::sort(ns.begin(),ns.end());
  std::cout<<std::setprecision(17)<<"{\"metric\":\"f3_k64_ns\",\"mode\":\""
      <<(F3_PROBE_GATHER?"gather":"normal")<<"\",\"tile_m\":"<<kTM
      <<",\"tile_n\":"<<kTN<<",\"ns\":"<<ns[ns.size()/2]
      <<",\"cycles_per_ns\":"<<cycles_per_ns<<",\"iterations_per_sample\":"<<kIterations
      <<",\"discard\":"<<kDiscard<<",\"method\":\"resident_smem_production_main_loop\",\"samples_ns\":[";
  for(std::size_t i=0;i<ns.size();++i)std::cout<<(i?",":"")<<ns[i];
  std::cout<<"]}\n";
  Check(cudaFree(stamps));Check(cudaFree(sink));return 0;
}catch(std::exception const& error){std::cerr<<error.what()<<'\n';return 1;}
