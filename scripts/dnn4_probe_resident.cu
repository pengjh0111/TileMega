// SPDX-License-Identifier: BSD-3-Clause
// DNN4 P-CAL-RES: resident operands and production K_PIPE_MMAS=1 sequencing.
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
#ifndef DNN4_SLOTS
#define DNN4_SLOTS 8
#endif

#ifndef F3_PROBE_GATHER
#define F3_PROBE_GATHER 0
#endif

using namespace tilemega::backend;
using U64=unsigned long long;
using Element=cutlass::bfloat16_t;
constexpr int kTM=F3_PROBE_TM,kTN=F3_PROBE_TN,kSlots=DNN4_SLOTS;
constexpr bool kAlias=kSlots*Sm90F3SlotBytes(kTM,kTN)>Sm90F3Budget::kSpanBudget;
constexpr int kStorageSlots=kAlias?1:kSlots;
constexpr int kIterations=64,kSamples=72,kDiscard=8;
using Side=Sm90F3SideLayout<kTM,kTN,false,0,false>;
using Chain=Sm90F3Chain<Side>;
using Body=Sm90F3Gemm<kTM,kTN,kAlias?2:kSlots,1,
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

// Immutable resident panels may alias when eight physical slots do not fit.
// Keep the production wait/commit/release order; no operand is rewritten here.
__device__ __forceinline__ void AliasedMainLoop(Sm90F3Ring& ring,Element* arena,
    Sm90F3Phase& phase,typename Body::Accumulators& accumulators,int wg) {
  using namespace cute;
  using Split=typename Body::Split;
  typename Body::Mma mma;
  auto thread=mma.get_slice(int(threadIdx.x)%128);
  auto& accum=accumulators.tile;
  #pragma unroll
  for(int s=0;s<Split::kSubTiles;++s)cute::clear(accum[s]);
  int previous=-1;
  for(int k=0;k<kIterations;++k) {
    int slot=k%kSlots;
    Sm90F3Async::Base::Wait(ring.full+slot,phase.Bit(phase.full,slot));
    phase.full^=1u<<slot;
    auto* a=Body::SlotA(arena,0);auto* b=Body::SlotB(arena,0);
    #pragma unroll
    for(int s=0;s<Split::kSubTiles;++s) {
      auto panel=make_tensor(make_smem_ptr(a+std::size_t(Split::ARows(wg)+s*64)*64),Sm90KLayout<64,64>{});
      auto page=make_tensor(make_smem_ptr(b+std::size_t(Split::BRows(wg))*64),Sm90KLayout<Split::kWgCols,64>{});
      auto ra=thread.partition_fragment_A(panel);auto rb=thread.partition_fragment_B(page);
      cute::warpgroup_fence_operand(accum[s]);cute::warpgroup_arrive();
      CUTE_UNROLL
      for(int step=0;step<size<2>(ra);++step)
        cute::gemm(mma,ra(cute::_,cute::_,step),rb(cute::_,cute::_,step),accum[s]);
      cute::warpgroup_commit_batch();
    }
    cute::warpgroup_wait<Split::kSubTiles>();
    #pragma unroll
    for(int s=0;s<Split::kSubTiles;++s)cute::warpgroup_fence_operand(accum[s]);
    if(previous>=0)Body::Release(ring,previous);
    previous=slot;
  }
  cute::warpgroup_wait<0>();
  #pragma unroll
  for(int s=0;s<Split::kSubTiles;++s)cute::warpgroup_fence_operand(accum[s]);
  if(previous>=0)Body::Release(ring,previous);
}

__global__ __launch_bounds__(384,1) void ResidentMainLoop(U64* stamps,float* sink) {
  extern __shared__ __align__(1024) char shared[];
  auto* header=reinterpret_cast<Sm90F3Header*>(shared);
  auto& ring=header->rings[0];
  auto* arena=reinterpret_cast<Element*>(shared+Sm90F3Budget::kArena0);
  if(threadIdx.x==0){ring.Initialize();Sm90F3Async::Base::InitFence();}
  for(int i=int(threadIdx.x);i<kStorageSlots*Body::kSlotBytes/2;i+=384)arena[i]=Element(.125f);
  Sm90F3Async::Base::FenceProxyAsyncShared();
  __syncthreads();
  if(threadIdx.x<256 && (threadIdx.x&31)==0)ring.Prime();
  __syncthreads();
  Sm90F3Phase phase{};
  if(threadIdx.x>=256) {
    Sm90F3Async::RegDealloc<56>();
    for(int sample=0;sample<kSamples;++sample)
      for(int k=0;k<kIterations;++k) {
        int slot=k%kSlots;
        if(threadIdx.x==256) {
          Sm90F3Async::Base::Wait(ring.empty+slot,phase.Bit(phase.empty,slot));
          phase.empty^=1u<<slot;
        }
        if constexpr(F3_PROBE_GATHER) {
          if(threadIdx.x<288 || threadIdx.x>=320) {
            Sm90F3Async::NamedBarrier<Sm90F3Async::kBarGather,96>();
            Sm90F3Async::Base::Arrive(ring.gfull+slot);
            if(threadIdx.x==256)Sm90F3Async::Base::Arrive(ring.gfull+slot);
          }
        }else if(threadIdx.x==256)Sm90F3Async::Base::Arrive(ring.full+slot);
        __syncwarp();
      }
  }else {
    Sm90F3Async::RegAlloc<224>();
    for(int sample=0;sample<kSamples;++sample) {
      Body::Accumulators accum;
      U64 start=Clock();
      if constexpr(kAlias) AliasedMainLoop(ring,arena,phase,accum,int(threadIdx.x)/128);
      else Body::MainLoop(ring,arena,phase,accum,int(threadIdx.x)/128,0,kIterations);
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
  constexpr int bytes=Sm90F3Budget::kHeader+kStorageSlots*Body::kSlotBytes;
  Check(cudaFuncSetAttribute(ResidentMainLoop,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  int active=0;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,ResidentMainLoop,384,bytes));
  if(active<1)throw std::runtime_error("resident probe cannot launch one CTA");
  ResidentMainLoop<<<1,384,bytes>>>(stamps,sink);
  Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  std::vector<U64> raw(kSamples);Check(cudaMemcpy(raw.data(),stamps,kSamples*sizeof(U64),cudaMemcpyDeviceToHost));
  std::vector<float> values(256);Check(cudaMemcpy(values.data(),sink,256*sizeof(float),cudaMemcpyDeviceToHost));
  // Each output accumulates 4096 products of .125*.125, exactly 64.
  for(float value:values)if(value!=64.f)throw std::runtime_error("resident main-loop output mismatch");
  std::vector<double> ns;
  for(int i=kDiscard;i<kSamples;++i)ns.push_back(raw[i]/cycles_per_ns/kIterations);
  std::sort(ns.begin(),ns.end());
  std::cout<<std::setprecision(17)<<"{\"metric\":\"resident_k64_ns\",\"mode\":\""
      <<(F3_PROBE_GATHER?"gather":"normal")<<"\",\"tile_m\":"<<kTM
      <<",\"tile_n\":"<<kTN<<",\"slots\":"<<kSlots<<",\"aliased_operands\":"<<(kAlias?"true":"false")<<",\"ns\":"<<(ns[(ns.size()-1)/2]+ns[ns.size()/2])/2
      <<",\"cycles_per_ns\":"<<cycles_per_ns<<",\"iterations_per_sample\":"<<kIterations
      <<",\"discard\":"<<kDiscard<<",\"method\":\"resident_smem_production_wait_commit_release_order\",\"samples_ns\":[";
  for(std::size_t i=0;i<ns.size();++i)std::cout<<(i?",":"")<<ns[i];
  std::cout<<"]}\n";
  Check(cudaFree(stamps));Check(cudaFree(sink));return 0;
}catch(std::exception const& error){std::cerr<<error.what()<<'\n';return 1;}
