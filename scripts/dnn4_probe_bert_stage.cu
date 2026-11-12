// SPDX-License-Identifier: BSD-3-Clause
// Isolate the admitted FFN bodies without Repeat bindings or neighboring stages.
#ifndef DNN4_BERT_BATCH
#define DNN4_BERT_BATCH 8
#endif
#ifndef DNN4_BERT_FFN
#define DNN4_BERT_FFN 1
#endif
#define TILEMEGA_F3_SIDE_NORM 1
#define TILEMEGA_F3_H800_GRID 1
#if DNN4_BERT_BATCH == 8
#define TILEMEGA_F3_REGISTER_BUDGET 2
#define TILEMEGA_F3_RECOMPUTE_TASK 1
#else
#define TILEMEGA_F3_REGISTER_BUDGET 3
#define TILEMEGA_F3_CYCLIC_SLOTS 1
#define TILEMEGA_F3_MATRIX_STORE 1
#endif
#include <tilemega/Backend/Sm90F3Gemm.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace tilemega::backend;
using E=cutlass::bfloat16_t;
using Tick=unsigned long long;
constexpr int kBatch=DNN4_BERT_BATCH,kFfn=DNN4_BERT_FFN,kGrid=132;
static_assert((kBatch==8 || kBatch==16) && (kFfn==1 || kFfn==2));
constexpr int kM=128*kBatch,kN=kFfn==1?3072:768,kK=kFfn==1?768:3072;
constexpr int kTM=128,kTN=kBatch==16?128:(kFfn==1?96:64);
constexpr int kSlots=kBatch==16?5:(kFfn==1?7:8);
constexpr bool kStore=kBatch==16 || kFfn==2;
using Side=Sm90F3SideLayout<kTM,kTN,kFfn==2,kFfn==1?2:3,true,12>;
using Norm=Sm90F3Step<Sm90ForwardOp::kDeferredLayerNorm,0,
    tilemega::codegen::DmActivation::kRelu,1>;
using Gelu=Sm90F3Step<Sm90ForwardOp::kActivation,0,
    tilemega::codegen::DmActivation::kGeluErf,0>;
using Residual=Sm90F3Step<Sm90ForwardOp::kResidualLN,1,
    tilemega::codegen::DmActivation::kRelu,2>;
using Chain=std::conditional_t<kFfn==1,Sm90F3Chain<Side,Norm,Gelu>,
    Sm90F3WriteRowStats<Sm90F3Chain<Side,
      Sm90F3Step<Sm90ForwardOp::kBias,0>,Residual>>>;
using G=Sm90F3Gemm<kTM,kTN,kSlots,1,Sm90F3Mode::kNormal,
    kM,kN,kK,kN,Chain,Sm90F3NoConv,Sm90F3NoPool,64,kStore>;
void Check(cudaError_t c) {
  if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));
}
template<class T> struct Device {
  T* data=nullptr;std::size_t count;
  explicit Device(std::size_t n):count(n){Check(cudaMalloc(&data,n*sizeof(T)));}
  ~Device(){cudaFree(data);}
  void Put(std::vector<T> const& v) {
    if(v.size()!=count)throw std::runtime_error("upload size");
    Check(cudaMemcpy(data,v.data(),count*sizeof(T),cudaMemcpyHostToDevice));
  }
  std::vector<T> Get()const {
    std::vector<T> v(count);
    Check(cudaMemcpy(v.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost));return v;
  }
};
__device__ __forceinline__ Tick Now(){return Sm90F3Async::GlobalTimer();}
__global__ __launch_bounds__(384,1) void Probe(Sm90F3GemmOperands p,
    const __grid_constant__ CUtensorMap activation,
    const __grid_constant__ CUtensorMap residual,
    const __grid_constant__ CUtensorMap output,Tick* stamps) {
  extern __shared__ __align__(1024) char shared[];
  auto* h=reinterpret_cast<Sm90F3Header*>(shared);auto& ring=h->rings[0];
  if(threadIdx.x==0){ring.Initialize();Sm90F3Async::Base::InitFence();}
  __syncthreads();
  if(threadIdx.x<256 && (threadIdx.x&31)==0)ring.Prime();
  __syncthreads();
  p.activation=&activation;p.residual=&residual;p.output_map=&output;
  // EpilogueOf supplies these shape constants in generated model kernels.
  p.epilogue.output_stride=kN;p.epilogue.residual_stride=kN;
  p.epilogue.norm_width=768;p.epilogue.epsilon=1e-5f;
  auto* arena=reinterpret_cast<E*>(shared+Sm90F3Budget::kArena0);
  auto* side=reinterpret_cast<char*>(arena)+G::kSideOffset;Sm90F3Phase phase{};
  if(threadIdx.x>=256) {
    Sm90F3Async::RegDealloc<72>();G::Produce(p,ring,arena,side,phase,0);
  }else {
    Sm90F3Async::RegAlloc<kBatch==8?208:216>();
    Tick begin=Now();
    G::Consume(p,ring,arena,side,phase,h,int(threadIdx.x)/128);
    if(threadIdx.x==0){stamps[2*blockIdx.x]=begin;stamps[2*blockIdx.x+1]=Now();}
  }
  __syncthreads();if(threadIdx.x==0)ring.Invalidate();
}
int main()try {
  if(!std::getenv("TILEMEGA_RUN_NONCE") || !std::getenv("TILEMEGA_GPU_LOCK_HELD"))
    throw std::runtime_error("stage probe requires scheduler and guard");
  Device<E> activation(std::size_t(kM)*kK),weight(std::size_t(kN)*kK),
      residual(std::size_t(kM)*kN),output(std::size_t(kM)*kN);
  Device<float> vectors(3*kN),stats(std::size_t(12)*kM*2),
      partials(std::size_t(kN/64)*kM*2);
  Device<Tick> stamps(2*kGrid);
  // Uniform values make both SW128 pages and every FP32 dot product exact.
  activation.Put(std::vector<E>(activation.count,E(.125f)));
  weight.Put(std::vector<E>(weight.count,E(.125f)));
  residual.Put(std::vector<E>(residual.count,E(.125f)));
  output.Put(std::vector<E>(output.count,E(-123.f)));
  std::vector<float> v(vectors.count,0.f),s(stats.count,0.f);
  std::fill(v.begin()+(kFfn==1?0:kN),v.begin()+(kFfn==1?kN:2*kN),1.f);
  for(std::size_t i=1;i<s.size();i+=2)s[i]=768.f*(1.f-1e-5f)/12;
  vectors.Put(v);stats.Put(s);partials.Put(std::vector<float>(partials.count,-123.f));
  CUtensorMap a{},r{},o{};
  tilemega::codegen::executor::EncodeTiledMap<2>(a,activation.data,{{kK,kM},{2*kK},{64,kTM}});
  tilemega::codegen::executor::EncodeTiledMap<2>(r,residual.data,{{kN,kM},{2*kN},{64,kTM}});
  tilemega::codegen::executor::EncodeTiledMap<2>(o,output.data,{{kN,kM},{2*kN},{64,64}});
  Sm90F3GemmOperands p{};p.weight=weight.data;
  p.epilogue.output=output.data;p.epilogue.output_stride=kN;
  p.epilogue.residual=residual.data;p.epilogue.residual_stride=kN;
  p.epilogue.stats=stats.data;p.epilogue.row_partials=partials.data;
  p.epilogue.norm_width=768;p.epilogue.epsilon=1e-5f;
  for(int i=0;i<3;++i)p.epilogue.vectors[i]=vectors.data+i*kN;
  Check(cudaFuncSetAttribute(Probe,cudaFuncAttributeMaxDynamicSharedMemorySize,Sm90F3Budget::kBudget));
  int resident=0;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Probe,384,Sm90F3Budget::kBudget));
  if(resident!=1)throw std::runtime_error("stage probe needs one CTA per SM");
  auto launch=[&](){Probe<<<kGrid,384,Sm90F3Budget::kBudget>>>(p,a,r,o,stamps.data);
    Check(cudaGetLastError());Check(cudaDeviceSynchronize());};
  for(int i=0;i<20;++i)launch();
  std::vector<Tick> windows,cta_max;
  for(int i=0;i<50;++i) {
    launch();auto t=stamps.Get();Tick first=~Tick(0),last=0,maximum=0;
    for(int c=0;c<kGrid;++c) {
      if(!t[2*c] || t[2*c+1]<t[2*c])throw std::runtime_error("unordered stamps");
      first=std::min(first,t[2*c]);last=std::max(last,t[2*c+1]);
      maximum=std::max(maximum,t[2*c+1]-t[2*c]);
    }
    windows.push_back(last-first);cta_max.push_back(maximum);
  }
  float expected=float(kK)/64+(kFfn==2?.125f:0.f);
  if constexpr(kFfn==1)expected=.5f*expected*(1.f+std::erf(expected*.70710678118654752f));
  auto values=output.Get();E rounded(expected);
  for(E x:values)if(x.raw()!=rounded.raw())throw std::runtime_error("stage BF16 output mismatch");
  if constexpr(kFfn==2) {
    auto partial=partials.Get();
    for(std::size_t i=0;i<partial.size();++i) {
      float reference=64*expected*(i%2?expected:1.f);
      if(!std::isfinite(partial[i]) ||
          std::abs(partial[i]-reference)>1e-4f*reference)
        throw std::runtime_error("row partial differs from FP32 reference");
    }
  }
  std::sort(windows.begin(),windows.end());std::sort(cta_max.begin(),cta_max.end());
  std::cout<<std::setprecision(17)<<"{\"batch\":"<<kBatch<<",\"ffn\":"<<kFfn
    <<",\"grid\":132,\"M\":"<<kM<<",\"N\":"<<kN<<",\"K\":"<<kK
    <<",\"tile_m\":"<<kTM<<",\"tile_n\":"<<kTN<<",\"slots\":"<<kSlots
    <<",\"tasks\":"<<G::kTasks<<",\"warmup\":20,\"samples\":50,\"window_ns\":"
    <<double(windows[24]+windows[25])/2<<",\"max_cta_ns\":"<<double(cta_max[24]+cta_max[25])/2
    <<",\"output_elements_checked\":"<<values.size()<<"}\n";
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
