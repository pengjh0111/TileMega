// SPDX-License-Identifier: BSD-3-Clause
// Last-arriver reduction at132 resident CTAs, with production fragment traffic.
#include <tilemega/Backend/Sm90F3Mark.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using U64=unsigned long long;
__device__ U64* split_stamps;
__device__ __forceinline__ U64 Clock() {
  U64 v;asm volatile("mov.u64 %0, %%clock64;":"=l"(v)::"memory");return v;
}
#undef TM_F3_TRACE_BEGIN
#undef TM_F3_TRACE_END
#define TM_F3_TRACE_BEGIN(kind) SPLIT_BEGIN_##kind
#define TM_F3_TRACE_END(kind) SPLIT_END_##kind
#define SPLIT_BEGIN_kSlotWait
#define SPLIT_END_kSlotWait
#define SPLIT_BEGIN_kMainLoop
#define SPLIT_END_kMainLoop
#define SPLIT_BEGIN_kSideWait
#define SPLIT_END_kSideWait
#define SPLIT_BEGIN_kEpilogue
#define SPLIT_END_kEpilogue
#define SPLIT_BEGIN_kStore
#define SPLIT_END_kStore
#define SPLIT_BEGIN_kSplitK U64 split_begin=Clock()
#define SPLIT_END_kSplitK if(threadIdx.x==0)split_stamps[blockIdx.x]=Clock()-split_begin
#include <tilemega/Backend/Sm90F3Gemm.cuh>
#include <tilemega/Backend/Sm90F3SwapAB.cuh>
using namespace tilemega::backend;
using E=cutlass::bfloat16_t;
#ifndef DNN4_SPLIT_TN
#define DNN4_SPLIT_TN 32
#endif
#ifndef DNN4_SPLIT_S
#define DNN4_SPLIT_S 6
#endif
#ifndef DNN4_SPLIT_SWAP
#define DNN4_SPLIT_SWAP 0
#endif
#ifndef DNN4_SPLIT_TM
#define DNN4_SPLIT_TM 64
#endif
constexpr int kM=DNN4_SPLIT_TM,kN=DNN4_SPLIT_TN,kS=DNN4_SPLIT_S;
constexpr int kGrid=132,kSamples=32,kDiscard=8;
constexpr int kTiles=kGrid/kS,kActive=kTiles*kS;
using Side=Sm90F3SideLayout<kM,kN,false,0,false>;
using Chain=Sm90F3Chain<Side>;
using B=std::conditional_t<DNN4_SPLIT_SWAP,
    Sm90F3GemmSwapAB<2,kS,kM,kN,64*kS,kN,Chain>,
    Sm90F3Gemm<kM,kN,2,kS,Sm90F3Mode::kNormal,kM,kN,64*kS,kN,Chain>>;
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ U64 Timer(){U64 v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v)::"memory");return v;}
__global__ void Calibrate(U64* p) {
  U64 start=Timer(),clock=Clock();while(Timer()-start<20000000ull){}
  p[0]=Clock()-clock;p[1]=Timer()-start;
}
template<class Body> __global__ __launch_bounds__(384,1)
void Probe(U64* stamps,E* output,float* partial,U64* counters,unsigned* last) {
  if(blockIdx.x>=kActive)return;
  extern __shared__ __align__(1024) char shared[];
  auto* h=reinterpret_cast<Sm90F3Header*>(shared);
  auto& ring=h->rings[0];
  if(threadIdx.x==0) {
    ring.Initialize();Sm90F3Async::Base::InitFence();
  }
  int output_tile=int(blockIdx.x)/kS,slice=int(blockIdx.x)%kS;
  auto* chunk=partial+std::size_t(output_tile)*kS*kM*kN;
  __syncthreads();
  if(threadIdx.x<256 && (threadIdx.x&31)==0)ring.Prime();
  __syncthreads();
  if(threadIdx.x>=256) {
    Sm90F3Async::RegDealloc<56>();
  }else {
    Sm90F3Async::RegAlloc<224>();
    Sm90F3GemmOperands p{};Sm90F3Phase phase{};
    p.epilogue.output=output+std::size_t(output_tile)*kM*kN;p.epilogue.output_stride=kN;
    p.epilogue.partial=chunk;p.epilogue.counters=counters+output_tile;
    typename Body::Accumulators accum;
    if constexpr(DNN4_SPLIT_SWAP)cute::fill(accum.tile[0],1.f);
    else {
      #pragma unroll
      for(int sub=0;sub<Body::Split::kSubTiles;++sub)cute::fill(accum.tile[sub],1.f);
    }
    auto* arena=reinterpret_cast<E*>(shared+Sm90F3Budget::kArena0);
    if constexpr(DNN4_SPLIT_SWAP) {
      typename Body::Mma mma;
      auto owned=mma.get_slice(int(threadIdx.x)%128).partition_C(
          cute::make_identity_tensor(cute::Shape<cute::_64,cute::Int<kM>>{}));
      Body::SplitKEpilogue(p,ring,reinterpret_cast<char*>(arena),phase,accum,owned,
          int(threadIdx.x)/128,slice,0,h);
    }else Body::SplitKEpilogue(p,ring,reinterpret_cast<char*>(arena),phase,accum,
        int(threadIdx.x)/128,slice,0,0,h,arena);
    if(threadIdx.x==0)last[blockIdx.x]=unsigned(h->last_arriver);
  }
  __syncthreads();
  if(threadIdx.x==0)ring.Invalidate();
}
int main()try {
  if(!std::getenv("TILEMEGA_RUN_NONCE") || !std::getenv("TILEMEGA_GPU_LOCK_HELD"))
    throw std::runtime_error("split probe requires scheduler/guard");
  U64* stamps;U64* counters;float* partial;E* output;unsigned* last;
  Check(cudaMalloc(&stamps,kSamples*kGrid*sizeof(U64)));
  Check(cudaMalloc(&counters,kGrid*sizeof(U64)));
  Check(cudaMalloc(&last,kSamples*kGrid*sizeof(unsigned)));
  Check(cudaMalloc(&partial,std::size_t(kTiles)*kS*kM*kN*sizeof(float)));
  Check(cudaMalloc(&output,std::size_t(kTiles)*kM*kN*sizeof(E)));
  Check(cudaMemset(counters,0,kGrid*sizeof(U64)));
  Check(cudaMemset(last,0,kSamples*kGrid*sizeof(unsigned)));
  Calibrate<<<1,1>>>(stamps);Check(cudaDeviceSynchronize());
  U64 cal[2];Check(cudaMemcpy(cal,stamps,sizeof(cal),cudaMemcpyDeviceToHost));
  double cycles_per_ns=double(cal[0])/cal[1];
  Check(cudaFuncSetAttribute(Probe<B>,cudaFuncAttributeMaxDynamicSharedMemorySize,Sm90F3Budget::kBudget));
  for(int sample=0;sample<kSamples;++sample) {
    U64* where=stamps+sample*kGrid;
    Check(cudaMemcpyToSymbol(split_stamps,&where,sizeof(where)));
    Probe<B><<<kGrid,384,Sm90F3Budget::kBudget>>>(where,output,partial,counters,last+sample*kGrid);
  }
  Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  std::vector<U64> raw(kSamples*kGrid);std::vector<E> values(kTiles*kM*kN);
  std::vector<unsigned> flags(kSamples*kGrid);
  Check(cudaMemcpy(raw.data(),stamps,raw.size()*sizeof(U64),cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(values.data(),output,values.size()*sizeof(E),cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(flags.data(),last,flags.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
  for(auto value:values)if(float(value)!=kS)throw std::runtime_error("split result mismatch");
  std::vector<double> times;
  for(int sample=kDiscard;sample<kSamples;++sample) {
    std::vector<U64> ctas;
    for(int cta=0;cta<kActive;++cta)if(flags[sample*kGrid+cta])ctas.push_back(raw[sample*kGrid+cta]);
    if(ctas.size()!=kTiles)throw std::runtime_error("last-arriver count mismatch");
    std::sort(ctas.begin(),ctas.end());times.push_back(ctas[ctas.size()/2]/cycles_per_ns);
  }
  std::sort(times.begin(),times.end());
  std::cout<<std::setprecision(17)<<"{\"tile_m\":"<<kM<<",\"tile_n\":"<<kN
      <<",\"slices\":"<<kS<<",\"swap\":"<<DNN4_SPLIT_SWAP
      <<",\"ns\":"<<times[times.size()/2]<<",\"grid\":132,\"active_ctas\":"<<kActive<<"}\n";
  Check(cudaFree(stamps));Check(cudaFree(counters));Check(cudaFree(partial));Check(cudaFree(output));Check(cudaFree(last));
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
