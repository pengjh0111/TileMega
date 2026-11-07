// SPDX-License-Identifier: BSD-3-Clause
// DNN3 §9.2: production epilogue with side operands resident in shared memory.
#include <tilemega/Backend/Sm90F3Gemm.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
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
#ifndef F3_PROBE_CLASS
#define F3_PROBE_CLASS 0
#endif
using namespace tilemega::backend;
using Element=cutlass::bfloat16_t;
using U64=unsigned long long;
constexpr int kTM=F3_PROBE_TM,kTN=F3_PROBE_TN,kClass=F3_PROBE_CLASS;
constexpr int kSamples=72,kDiscard=8;
static_assert(kClass>=0 && kClass<6);
using Side=Sm90F3SideLayout<kTM,kTN,kClass==2 || kClass==5,
    kClass==0?0:kClass<3?1:kClass<5?2:3,kClass>=3>;
using Bias=Sm90F3Step<Sm90F3Op::kBias>;
using Relu=Sm90F3Step<Sm90F3Op::kActivation>;
using Residual=Sm90F3Step<Sm90F3Op::kResidual,~0u>;
using Deferred=Sm90F3Step<Sm90F3Op::kDeferredLayerNorm,0,
    tilemega::codegen::DmActivation::kRelu,1>;
using Gelu=Sm90F3Step<Sm90F3Op::kActivation,0,
    tilemega::codegen::DmActivation::kGeluErf>;
using ResidualLN=Sm90F3Step<Sm90F3Op::kResidualLN,1,
    tilemega::codegen::DmActivation::kRelu,2>;
template<int C> struct Select;
template<> struct Select<0>{using Type=Sm90F3Chain<Side>;};
template<> struct Select<1>{using Type=Sm90F3Chain<Side,Bias,Relu>;};
template<> struct Select<2>{using Type=Sm90F3Chain<Side,Bias,Residual,Relu>;};
template<> struct Select<3>{using Type=Sm90F3Chain<Side,Deferred>;};
template<> struct Select<4>{using Type=Sm90F3Chain<Side,Deferred,Gelu>;};
template<> struct Select<5>{using Type=Sm90F3WriteRowStats<
    Sm90F3Chain<Side,Bias,ResidualLN>>;};
using Chain=typename Select<kClass>::Type;
using Body=Sm90F3Gemm<kTM,kTN,2,1,Sm90F3Mode::kNormal,kTM,kTN,64,kTN,Chain>;
constexpr bool kHasSide=Side::kBytes!=0;

void Check(cudaError_t error) {
  if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
__device__ __forceinline__ U64 Clock() {
  U64 value;asm volatile("mov.u64 %0, %%clock64;":"=l"(value)::"memory");return value;
}
__device__ __forceinline__ U64 Timer() {
  U64 value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value)::"memory");return value;
}
__global__ void Calibrate(U64* output) {
  U64 start=Timer(),ticks=Clock();
  while(Timer()-start<20000000ull){}
  output[0]=Clock()-ticks;output[1]=Timer()-start;
}

__global__ __launch_bounds__(288,1)
void ResidentEpilogue(U64* stamps,Element* output,float* stats) {
  extern __shared__ __align__(1024) char shared[];
  auto* header=reinterpret_cast<Sm90F3Header*>(shared);
  auto& ring=header->rings[0];
  char* arena=shared+Sm90F3Budget::kArena0;
  char* side=arena+Body::kSideOffset;
  if(threadIdx.x==0){ring.Initialize();Sm90F3Async::Base::InitFence();}
  for(int i=int(threadIdx.x);i<Side::kResidualBytes/2;i+=288)
    reinterpret_cast<Element*>(side)[i]=Element(.125f);
  for(int i=int(threadIdx.x);i<Body::kVectors*kTN;i+=288) {
    int vector=i/kTN;
    float value=(kClass==3 || kClass==4)?float(vector==0):
                kClass==5?float(vector==1):0.f;
    reinterpret_cast<float*>(side+Side::kVectorBase)[i]=value;
  }
  if constexpr(kClass>=3)
    for(int row=int(threadIdx.x);row<kTM;row+=288) {
      auto* raw=reinterpret_cast<float*>(side+Side::kRowStatBase);
      raw[2*row]=0.f;raw[2*row+1]=768.f;
    }
  __syncthreads();
  if(threadIdx.x<256 && (threadIdx.x&31)==0)ring.Prime();
  __syncthreads();
  Sm90F3Phase phase{};
  Sm90F3GemmOperands p{};
  p.epilogue.output=output;p.epilogue.output_stride=kTN;
  p.epilogue.norm_width=768;p.epilogue.epsilon=0.f;
  p.epilogue.row_stats=stats;
  for(int sample=0;sample<kSamples;++sample) {
    if constexpr(kHasSide)
      if(threadIdx.x==256) {
        Sm90F3Async::Base::Wait(&ring.side_empty,phase.side_empty&1u);
        phase.side_empty^=1u;
        Sm90F3Async::Base::Arrive(&ring.side_full);
      }
    // The timed region sees a completed side barrier, without a TMA transfer.
    __syncthreads();
    if(threadIdx.x<256) {
      Body::Accumulators accum;
      #pragma unroll
      for(int s=0;s<Body::Split::kSubTiles;++s)
        cute::fill(accum.tile[s],.5f);
      U64 start=Clock();
      Body::Epilogue(p,ring,side,phase,accum,int(threadIdx.x)/128,
          0,0,kTM,kTN,header,reinterpret_cast<Element*>(arena));
      U64 end=Clock();
      if(threadIdx.x==0)stamps[sample]=end-start;
    }
    __syncthreads();
  }
  if(threadIdx.x==0)ring.Invalidate();
}

int main()try {
  if(!std::getenv("TILEMEGA_RUN_NONCE") ||
      !std::getenv("TILEMEGA_GPU_LOCK_HELD"))
    throw std::runtime_error("constant probes require the GPU scheduler and guard");
  U64* stamps=nullptr;Element* output=nullptr;float* stats=nullptr;
  Check(cudaMalloc(&stamps,kSamples*sizeof(U64)));
  Check(cudaMalloc(&output,kTM*kTN*sizeof(Element)));
  Check(cudaMalloc(&stats,2*kTM*sizeof(float)));
  Calibrate<<<1,1>>>(stamps);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  U64 calibration[2];Check(cudaMemcpy(calibration,stamps,sizeof(calibration),cudaMemcpyDeviceToHost));
  double cycles_per_ns=double(calibration[0])/calibration[1];
  constexpr int bytes=Sm90F3Budget::kHeader+Body::kLayoutBytes;
  Check(cudaFuncSetAttribute(ResidentEpilogue,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  int active=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,ResidentEpilogue,288,bytes));
  if(active<1)throw std::runtime_error("resident epilogue cannot launch one CTA");
  ResidentEpilogue<<<1,288,bytes>>>(stamps,output,stats);
  Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  std::vector<U64> raw(kSamples);Check(cudaMemcpy(raw.data(),stamps,kSamples*sizeof(U64),cudaMemcpyDeviceToHost));
  std::vector<Element> values(kTM*kTN);
  Check(cudaMemcpy(values.data(),output,values.size()*sizeof(Element),cudaMemcpyDeviceToHost));
  float expected=kClass==2 || kClass==5?.625f:
      kClass==4?float(Element(float(.25*(1+std::erf(.5/std::sqrt(2.)))))):.5f;
  for(auto value:values)if(float(value)!=expected)
    throw std::runtime_error("resident epilogue output mismatch");
  if constexpr(kClass==5) {
    std::vector<float> rows(2*kTM);
    Check(cudaMemcpy(rows.data(),stats,rows.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(int row=0;row<kTM;++row)
      if(rows[2*row]!=.625f*kTN || rows[2*row+1]!=.390625f*kTN)
        throw std::runtime_error("resident epilogue row-statistics mismatch");
  }
  std::vector<double> ns;
  for(int i=kDiscard;i<kSamples;++i)ns.push_back(raw[i]/cycles_per_ns);
  std::sort(ns.begin(),ns.end());
  char const* classes[]={"plain","bias_act","bias_res_act","deferred_ln",
      "deferred_ln_gelu","residual_ln_rowstats"};
  std::cout<<std::setprecision(17)<<"{\"metric\":\"f3_epilogue_ns\",\"class\":\""
      <<classes[kClass]<<"\",\"tile_m\":"<<kTM<<",\"tile_n\":"<<kTN
      <<",\"ns\":"<<ns[ns.size()/2]<<",\"cycles_per_ns\":"<<cycles_per_ns
      <<",\"method\":\"resident_side_production_epilogue\",\"samples_ns\":[";
  for(std::size_t i=0;i<ns.size();++i)std::cout<<(i?",":"")<<ns[i];
  std::cout<<"]}\n";
  Check(cudaFree(stamps));Check(cudaFree(output));Check(cudaFree(stats));return 0;
}catch(std::exception const& error){std::cerr<<error.what()<<'\n';return 1;}
