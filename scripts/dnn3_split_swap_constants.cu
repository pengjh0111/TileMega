// SPDX-License-Identifier: BSD-3-Clause
// Isolate production swap main loops/epilogues and last-arriver split-K.
#include <tilemega/Backend/Sm90F3Mark.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using U64=unsigned long long;
__device__ U64* probe_stamps;
__device__ int probe_sample;
__device__ __forceinline__ U64 ProbeClock() {
  U64 value;asm volatile("mov.u64 %0, %%clock64;":"=l"(value)::"memory");return value;
}
#ifndef F3_PROBE_FAMILY
#define F3_PROBE_FAMILY 0
#endif
#ifndef F3_PROBE_CONSUMER_ONLY
#define F3_PROBE_CONSUMER_ONLY 0
#endif
#ifndef F3_PROBE_TM
#define F3_PROBE_TM 128
#endif
#ifndef F3_PROBE_TN
#define F3_PROBE_TN 64
#endif
#undef TM_F3_TRACE_BEGIN
#undef TM_F3_TRACE_END
// Only the chosen production region retains clock reads; other markers vanish.
#define TM_F3_TRACE_BEGIN(kind) F3_PROBE_BEGIN_##kind
#define TM_F3_TRACE_END(kind) F3_PROBE_END_##kind
#define F3_PROBE_BEGIN_kSlotWait
#define F3_PROBE_END_kSlotWait
#define F3_PROBE_BEGIN_kSideWait
#define F3_PROBE_END_kSideWait
#define F3_PROBE_BEGIN_kStore
#define F3_PROBE_END_kStore
#if F3_PROBE_FAMILY==1
#define F3_PROBE_BEGIN_kMainLoop U64 probe_begin_kMainLoop=ProbeClock()
#define F3_PROBE_END_kMainLoop if(Sm90F3Async::ThreadIndex()==0)probe_stamps[2*probe_sample]=ProbeClock()-probe_begin_kMainLoop
#define F3_PROBE_BEGIN_kEpilogue U64 probe_begin_kEpilogue=ProbeClock()
#define F3_PROBE_END_kEpilogue if(Sm90F3Async::ThreadIndex()==0)probe_stamps[2*probe_sample+1]=ProbeClock()-probe_begin_kEpilogue
#define F3_PROBE_BEGIN_kSplitK
#define F3_PROBE_END_kSplitK
#else
#define F3_PROBE_BEGIN_kMainLoop
#define F3_PROBE_END_kMainLoop
#define F3_PROBE_BEGIN_kEpilogue
#define F3_PROBE_END_kEpilogue
#define F3_PROBE_BEGIN_kSplitK U64 probe_begin_kSplitK=ProbeClock()
#define F3_PROBE_END_kSplitK if(Sm90F3Async::ThreadIndex()==0)probe_stamps[2*probe_sample]=ProbeClock()-probe_begin_kSplitK
#endif
#include <tilemega/Backend/Sm90F3Gemm.cuh>
#include <tilemega/Backend/Sm90F3SwapAB.cuh>
using namespace tilemega::backend;
using Element=cutlass::bfloat16_t;
constexpr int kTM=F3_PROBE_TM,kTN=F3_PROBE_TN,kSamples=72,kDiscard=8;
constexpr bool kSwap=F3_PROBE_FAMILY!=0;
static_assert(!F3_PROBE_CONSUMER_ONLY || F3_PROBE_FAMILY==0);
constexpr int kProbeThreads=F3_PROBE_CONSUMER_ONLY?256:288;
template<int S> using Side=Sm90F3SideLayout<kTM,kTN,false,F3_PROBE_FAMILY==1?1:0,false>;
template<int S> using Chain=std::conditional_t<F3_PROBE_FAMILY==1,
    Sm90F3Chain<Side<S>,Sm90F3Step<Sm90F3Op::kBias>,Sm90F3Step<Sm90F3Op::kActivation>>,
    Sm90F3Chain<Side<S>>>;
template<int S> using Body=std::conditional_t<kSwap,
    Sm90F3GemmSwapAB<2,S,kTM,kTN,64*64*S,kTN,Chain<S>>,
    Sm90F3Gemm<kTM,kTN,2,S,Sm90F3Mode::kNormal,kTM,kTN,64*S,kTN,Chain<S>>>;
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ U64 Timer(){U64 v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v)::"memory");return v;}
__global__ void Calibrate(U64* output) {
  U64 start=Timer(),ticks=ProbeClock();while(Timer()-start<20000000ull){}
  output[0]=ProbeClock()-ticks;output[1]=Timer()-start;
}

// The plain split region has no producer work. Its isolated fixture can
// launch the same two consumer warpgroups without reserving an idle warp.
#if F3_PROBE_CONSUMER_ONLY
template<int S> __global__ __maxnreg__(224)
#else
template<int S> __global__ __launch_bounds__(288,1)
#endif
void Resident(U64* stamps,Element* output,float* partial,U64* counters,int sample) {
  using B=Body<S>;
  extern __shared__ __align__(1024) char shared[];
  auto* header=reinterpret_cast<Sm90F3Header*>(shared);
  auto& ring=header->rings[0];
  auto* arena=reinterpret_cast<Element*>(shared+Sm90F3Budget::kArena0);
  auto* side=reinterpret_cast<char*>(arena)+B::kSideOffset;
  if(Sm90F3Async::ThreadIndex()==0) {
    ring.Initialize();Sm90F3Async::Base::InitFence();probe_stamps=stamps;
  }
  if constexpr(F3_PROBE_FAMILY==1) {
    for(int i=int(threadIdx.x);i<2*B::kSlotBytes/2;i+=kProbeThreads)arena[i]=Element(.125f);
    for(int i=int(threadIdx.x);i<kTN;i+=kProbeThreads)reinterpret_cast<float*>(side)[i]=0.f;
  }
  Sm90F3Async::Base::FenceProxyAsyncShared();__syncthreads();
  if(threadIdx.x<256 && (Sm90F3Async::ThreadIndex()&31)==0)ring.Prime();
  __syncthreads();Sm90F3Phase phase{};Sm90F3GemmOperands p{};
  p.epilogue.output=output;p.epilogue.output_stride=kTN;
  p.epilogue.partial=partial;p.epilogue.counters=counters;
  // Untimed preparation stays outside the production split-K markers.
  if constexpr(F3_PROBE_FAMILY!=1)
    for(int i=int(threadIdx.x);i<S*kTM*kTN;i+=kProbeThreads)partial[i]=1.f;
  __syncthreads();
  {
    if(Sm90F3Async::ThreadIndex()==0){probe_sample=sample;counters[0]=S-1;}
    __syncthreads();
    if constexpr(F3_PROBE_FAMILY==1) {
      if(threadIdx.x>=256) {
        if(Sm90F3Async::ThreadIndex()==256) {
          for(int k=0;k<64;++k) {
            int slot=k%2;Sm90F3Async::Base::Wait(ring.empty+slot,phase.Bit(phase.empty,slot));
            phase.empty^=1u<<slot;Sm90F3Async::Base::Arrive(ring.full+slot);
          }
          Sm90F3Async::Base::Wait(&ring.side_empty,phase.side_empty&1u);
          phase.side_empty^=1u;Sm90F3Async::Base::Arrive(&ring.side_full);
        }
      }else B::Consume(p,ring,arena,side,phase,header,int(threadIdx.x)/128);
    }else if(threadIdx.x<256) {
      typename B::Accumulators accum;
      if constexpr(kSwap) {
        cute::fill(accum.tile[0],1.f);
        typename B::Mma mma;
        auto owned=mma.get_slice(int(threadIdx.x)%128).partition_C(
            cute::make_identity_tensor(cute::Shape<cute::_64,cute::Int<kTM>>{}));
        B::SplitKEpilogue(p,ring,side,phase,accum,owned,int(threadIdx.x)/128,S-1,0,header);
      }else {
        #pragma unroll
        for(int sub=0;sub<B::Split::kSubTiles;++sub)cute::fill(accum.tile[sub],1.f);
        B::SplitKEpilogue(p,ring,side,phase,accum,int(threadIdx.x)/128,S-1,0,0,header,arena);
      }
    }
    __syncthreads();
  }
  if(Sm90F3Async::ThreadIndex()==0)ring.Invalidate();
}

template<int S> void Run() {
  using B=Body<S>;
  U64* stamps;Element* output;float* partial;U64* counters;
  Check(cudaMalloc(&stamps,2*kSamples*sizeof(U64)));
  Check(cudaMalloc(&output,kTM*kTN*sizeof(Element)));
  Check(cudaMalloc(&partial,S*kTM*kTN*sizeof(float)));Check(cudaMalloc(&counters,sizeof(U64)));
  Calibrate<<<1,1>>>(stamps);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  U64 calibration[2];Check(cudaMemcpy(calibration,stamps,sizeof(calibration),cudaMemcpyDeviceToHost));
  double cycles_per_ns=double(calibration[0])/calibration[1];
  constexpr int bytes=Sm90F3Budget::kHeader+B::kLayoutBytes;
  Check(cudaFuncSetAttribute(Resident<S>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  // Each sample is a fresh resident region; launch overhead is not clocked.
  for(int sample=0;sample<kSamples;++sample)
    Resident<S><<<1,kProbeThreads,bytes>>>(stamps,output,partial,counters,sample);
  Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  std::vector<U64> raw(2*kSamples);Check(cudaMemcpy(raw.data(),stamps,raw.size()*sizeof(U64),cudaMemcpyDeviceToHost));
  std::vector<Element> values(kTM*kTN);Check(cudaMemcpy(values.data(),output,values.size()*sizeof(Element),cudaMemcpyDeviceToHost));
  for(auto value:values)if(float(value)!=(F3_PROBE_FAMILY==1?64.f:float(S)))
    throw std::runtime_error("production split/swap output mismatch");
  U64 counter=~U64(0);Check(cudaMemcpy(&counter,counters,sizeof(U64),cudaMemcpyDeviceToHost));
  if(F3_PROBE_FAMILY!=1 && counter!=0)throw std::runtime_error("last arriver did not reset counter");
  std::cout<<std::setprecision(17)<<"{\"tile_m\":"<<kTM<<",\"tile_n\":"<<kTN
      <<",\"probe_threads\":"<<kProbeThreads<<",\"slices\":"<<S<<",\"cycles_per_ns\":"<<cycles_per_ns<<",\"rows\":[";
  for(int metric=0;metric<(F3_PROBE_FAMILY==1?2:1);++metric) {
    std::vector<double> ns;
    for(int i=kDiscard;i<kSamples;++i)ns.push_back(raw[2*i+metric]/cycles_per_ns/
        (F3_PROBE_FAMILY==1 && metric==0?64:1));
    std::sort(ns.begin(),ns.end());
    std::cout<<(metric?",":"")<<"{\"metric\":\""<<(F3_PROBE_FAMILY==1?
        (metric==0?"f3_k64_ns":"f3_epilogue_ns"):"f3_splitk_ns")
        <<"\",\"mode\":\""<<(kSwap?"swapab":"normal")
        <<"\",\"class\":\""<<(F3_PROBE_FAMILY==1?"swapab_bias_act":"plain")
        <<"\",\"ns\":"<<ns[ns.size()/2]<<",\"samples_ns\":[";
    for(std::size_t i=0;i<ns.size();++i)std::cout<<(i?",":"")<<ns[i];
    std::cout<<"]}";
  }
  std::cout<<"]}\n";
  Check(cudaFree(stamps));Check(cudaFree(output));Check(cudaFree(partial));Check(cudaFree(counters));
}
int main(int argc,char** argv)try {
  if(!std::getenv("TILEMEGA_RUN_NONCE") || !std::getenv("TILEMEGA_GPU_LOCK_HELD"))
    throw std::runtime_error("constant probes require scheduler/guard");
  int slices=argc==2?std::stoi(argv[1]):1;
  if constexpr(F3_PROBE_FAMILY==1){if(slices!=1)throw std::runtime_error("resident swap has no split");Run<1>();}
  else switch(slices) {
#define F3_CASE(S) case S:Run<S>();break
    F3_CASE(2);F3_CASE(3);F3_CASE(4);F3_CASE(6);F3_CASE(7);F3_CASE(8);
    F3_CASE(9);F3_CASE(12);F3_CASE(14);F3_CASE(16);
#if F3_PROBE_FAMILY==2
    F3_CASE(28);F3_CASE(32);
#endif
#undef F3_CASE
    default:throw std::runtime_error("unsupported measured split");
  }
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
