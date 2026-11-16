// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90F3Mark.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using U64=unsigned long long;
__device__ __forceinline__ U64 Clock() {U64 v;asm volatile("mov.u64 %0, %%clock64;":"=l"(v)::"memory");return v;}
__device__ U64* epilogue_stamps;
__device__ unsigned epilogue_task[132];
#undef TM_F3_TRACE_BEGIN
#undef TM_F3_TRACE_END
#define TM_F3_TRACE_BEGIN(kind) GELU_BEGIN_##kind
#define TM_F3_TRACE_END(kind) GELU_END_##kind
#define GELU_BEGIN_kSlotWait
#define GELU_END_kSlotWait
#define GELU_BEGIN_kMainLoop
#define GELU_END_kMainLoop
#define GELU_BEGIN_kSideWait
#define GELU_END_kSideWait
#define GELU_BEGIN_kSplitK
#define GELU_END_kSplitK
#define GELU_BEGIN_kStore
#define GELU_END_kStore
#define GELU_BEGIN_kEpilogue U64 epilogue_begin=Clock()
#define GELU_END_kEpilogue if(threadIdx.x==0)epilogue_stamps[blockIdx.x*3+epilogue_task[blockIdx.x]++]=Clock()-epilogue_begin
#include <tilemega/Backend/Sm90F3Gemm.cuh>
using namespace tilemega::backend;
using E=cutlass::bfloat16_t;
__global__ void Accuracy(unsigned* worst,unsigned* bad) {
  constexpr unsigned count=1u<<24;
  for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<count+65536u;i+=gridDim.x*blockDim.x) {
    float x;
    if(i<count)x=-8.f+16.f*(float(i)/float(count-1));
    else {
      unsigned bits=i-count;
      if((bits&0x7f80u)==0x7f80u)continue;
      x=__uint_as_float(bits<<16);
    }
    float original=.5f*x*(1.f+erff(x*.70710678118654752f));
    float candidate=Sm90F3GeluErf(x),error=fabsf(candidate-original)/fmaxf(1.f,fabsf(x));
    if(!isfinite(candidate) || !(error<=1e-6f))atomicAdd(bad,1u);
    atomicMax(worst,__float_as_uint(error));
  }
}
using Side=Sm90F3SideLayout<128,128,false,2,true>;
template<bool Approx> struct Chain:Sm90F3Chain<Side> {
  static constexpr bool kNeedsStats=true;
  __device__ static __forceinline__ float Apply(float value,int row,int col,char const* side,
      float mean,float inverse,Sm90F3EpilogueOperands const&,float const* columns=nullptr) {
    float normalized=(value-mean*(columns?columns[0]:Side::Vector(side,0)[col]))*inverse+
        (columns?columns[1]:Side::Vector(side,1)[col]);
    if constexpr(Approx)return Sm90F3GeluErf(normalized);
    else return Sm90ForwardActivate(tilemega::codegen::DmActivation::kGeluErf,normalized);
  }
};
template<bool Approx> using Body=Sm90F3Gemm<128,128,2,1,Sm90F3Mode::kNormal,132*3*128,128,64,128,Chain<Approx>>;
template<bool Approx> __global__ __launch_bounds__(384,1)
void Epilogue(Sm90F3GemmOperands p,float value) {
  using B=Body<Approx>;
  extern __shared__ __align__(1024) char shared[];
  auto* h=reinterpret_cast<Sm90F3Header*>(shared);auto& ring=h->rings[0];
  if(threadIdx.x==0) {ring.Initialize();Sm90F3Async::Base::InitFence();epilogue_task[blockIdx.x]=0;}
  __syncthreads();
  if(threadIdx.x<256 && (threadIdx.x&31)==0)ring.Prime();
  __syncthreads();
  auto* arena=reinterpret_cast<E*>(shared+Sm90F3Budget::kArena0);
  auto* side=reinterpret_cast<char*>(arena)+B::kSideOffset;Sm90F3Phase phase{};
  if(threadIdx.x>=256) {
    Sm90F3Async::RegDealloc<56>();
    if(threadIdx.x==288)for(int task=0;task<3;++task)
      B::ProduceSide(p,ring,side,phase,int(blockIdx.x)+task*132,0,128);
  }else {
    Sm90F3Async::RegAlloc<224>();
    for(int task=0;task<3;++task) {
      typename B::Accumulators accum;cute::fill(accum.tile[0],value);
      B::Epilogue(p,ring,side,phase,accum,int(threadIdx.x)/128,
          int(blockIdx.x)+task*132,0,128,128,h,arena);
    }
  }
  __syncthreads();if(threadIdx.x==0)ring.Invalidate();
}
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ U64 Timer(){U64 v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v)::"memory");return v;}
__global__ void Calibrate(U64* output) {
  U64 start=Timer(),clock=Clock();while(Timer()-start<20000000ull){}
  output[0]=Clock()-clock;output[1]=Timer()-start;
}
template<bool Approx> double Measure(Sm90F3GemmOperands p,U64* stamps,double cycles_per_ns) {
  Check(cudaFuncSetAttribute(Epilogue<Approx>,cudaFuncAttributeMaxDynamicSharedMemorySize,Sm90F3Budget::kBudget));
  for(int i=0;i<50;++i)Epilogue<Approx><<<132,384,Sm90F3Budget::kBudget>>>(p,.125f);
  Check(cudaGetLastError());Check(cudaDeviceSynchronize());std::vector<double> times;
  for(int i=0;i<16;++i) {
    Epilogue<Approx><<<132,384,Sm90F3Budget::kBudget>>>(p,.125f);
    Check(cudaGetLastError());Check(cudaDeviceSynchronize());std::vector<U64> raw(132*3);
    Check(cudaMemcpy(raw.data(),stamps,raw.size()*sizeof(U64),cudaMemcpyDeviceToHost));
    std::sort(raw.begin(),raw.end());times.push_back(raw[raw.size()/2]/cycles_per_ns);
  }
  std::sort(times.begin(),times.end());return times[times.size()/2];
}
int main(int argc,char** argv)try {
  if(!std::getenv("TILEMEGA_RUN_NONCE") || !std::getenv("TILEMEGA_GPU_LOCK_HELD"))throw std::runtime_error("GELU probe needs guard");
  if(argc!=2)throw std::runtime_error("choose accuracy or epilogue");
  std::cout<<std::setprecision(17);
  if(std::string(argv[1])=="accuracy") {
    unsigned* counters;Check(cudaMalloc(&counters,2*sizeof(unsigned)));Check(cudaMemset(counters,0,2*sizeof(unsigned)));
    Accuracy<<<132,256>>>(counters,counters+1);Check(cudaGetLastError());Check(cudaDeviceSynchronize());unsigned result[2];
    Check(cudaMemcpy(result,counters,sizeof(result),cudaMemcpyDeviceToHost));float error;std::memcpy(&error,result,sizeof(float));
    std::cout<<"{\"max_relative\":"<<error<<",\"bad\":"<<result[1]<<",\"grid_points\":16777216,\"bf16_patterns\":65536}\n";
    Check(cudaFree(counters));return result[1]?1:0;
  }
  constexpr int rows=132*3*128;float *stats,*vectors;E* output;U64* stamps;
  Check(cudaMalloc(&stats,rows*2*sizeof(float)));Check(cudaMalloc(&vectors,256*sizeof(float)));
  Check(cudaMalloc(&output,std::size_t(rows)*128*sizeof(E)));Check(cudaMalloc(&stamps,132*3*sizeof(U64)));
  std::vector<float> host_stats(rows*2,0),host_vectors(256,0);
  for(int i=0;i<rows;++i)host_stats[i*2+1]=768.f*(1.f-1e-5f);
  std::fill(host_vectors.begin(),host_vectors.begin()+128,1.f);
  Check(cudaMemcpy(stats,host_stats.data(),host_stats.size()*sizeof(float),cudaMemcpyHostToDevice));
  Check(cudaMemcpy(vectors,host_vectors.data(),host_vectors.size()*sizeof(float),cudaMemcpyHostToDevice));
  Check(cudaMemcpyToSymbol(epilogue_stamps,&stamps,sizeof(stamps)));
  Calibrate<<<1,1>>>(stamps);Check(cudaDeviceSynchronize());U64 cal[2];Check(cudaMemcpy(cal,stamps,sizeof(cal),cudaMemcpyDeviceToHost));
  double cycles_per_ns=double(cal[0])/cal[1];Sm90F3GemmOperands p{};
  p.epilogue.output=output;p.epilogue.output_stride=128;p.epilogue.stats=stats;
  p.epilogue.vectors[0]=vectors;p.epilogue.vectors[1]=vectors+128;p.epilogue.norm_width=768;p.epilogue.epsilon=1e-5f;
  double baseline=Measure<false>(p,stamps,cycles_per_ns),candidate=Measure<true>(p,stamps,cycles_per_ns);
  std::cout<<"{\"baseline_ns\":"<<baseline<<",\"candidate_ns\":"<<candidate<<",\"ratio\":"<<candidate/baseline<<",\"active_ctas\":132,\"tasks_per_cta\":3}\n";
  Check(cudaFree(stats));Check(cudaFree(vectors));Check(cudaFree(output));Check(cudaFree(stamps));
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
