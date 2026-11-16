// SPDX-License-Identifier: BSD-3-Clause
// Isolate the admitted FFN bodies without Repeat bindings or neighboring stages.
#ifndef DNN4_BERT_BATCH
#define DNN4_BERT_BATCH 8
#endif
#ifndef DNN4_BERT_FFN
#define DNN4_BERT_FFN 1
#endif
#ifndef DNN4_BERT_EPSILON
#define DNN4_BERT_EPSILON 1e-12f
#endif
#if defined(DNN4_BERT_CODE_EVICT) + defined(DNN4_BERT_DATA_EVICT) + defined(DNN4_BERT_DATA_PATTERN) > 1
#error "select one probe context"
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
#include <string>
#include <vector>

using namespace tilemega::backend;
using E=cutlass::bfloat16_t;
using Tick=unsigned long long;
constexpr int kBatch=DNN4_BERT_BATCH,kFfn=DNN4_BERT_FFN,kGrid=132;
constexpr float kEpsilon=DNN4_BERT_EPSILON;
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
#ifdef DNN4_BERT_CODE_EVICT
template<int KB> __device__ __noinline__ float Work(float);
// Reuse the Phase P straight-line code, frozen with this probe's build.
#include "dnn4_icache_ops.inc"
__global__ __launch_bounds__(256,1) void EvictBertCode(float* sink) {
  sink[blockIdx.x*256+threadIdx.x]=Work<1024>(float(threadIdx.x)*.001f);
}
#endif
#ifdef DNN4_BERT_DATA_EVICT
constexpr int kFlushVectors=128*1024*1024/sizeof(uint4);
__global__ void EvictBertData(uint4* buffer) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<kFlushVectors;
      i+=gridDim.x*blockDim.x) {
    unsigned x=unsigned(i)*747796405u+2891336453u;
    x=((x>>((x>>28)+4))^x)*277803737u;
    buffer[i]=make_uint4(x,x^0xa5a5a5a5u,x*1664525u+1013904223u,~x);
  }
}
#endif
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
  p.epilogue.norm_width=768;p.epilogue.epsilon=kEpsilon;
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
int main(int argc,char** argv)try {
  if(!std::getenv("TILEMEGA_RUN_NONCE") || !std::getenv("TILEMEGA_GPU_LOCK_HELD"))
    throw std::runtime_error("stage probe requires scheduler and guard");
#ifdef DNN4_BERT_CODE_EVICT
  if(argc!=2 || (std::string(argv[1])!="hot" && std::string(argv[1])!="l1-cold"))
    throw std::runtime_error("usage: probe {hot|l1-cold}");
  bool evict=std::string(argv[1])=="l1-cold";
  Device<float> sink(kGrid*256);
  Check(cudaFuncSetAttribute(EvictBertCode,cudaFuncAttributeMaxDynamicSharedMemorySize,Sm90F3Budget::kBudget));
  int evict_resident=0;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&evict_resident,EvictBertCode,256,Sm90F3Budget::kBudget));
  if(evict_resident!=1)throw std::runtime_error("code eviction needs one CTA per SM");
#elif defined(DNN4_BERT_DATA_EVICT)
  if(argc!=2 || (std::string(argv[1])!="hot" && std::string(argv[1])!="data-cold"))
    throw std::runtime_error("usage: probe {hot|data-cold}");
  bool evict=std::string(argv[1])=="data-cold";
  Device<uint4> flush(kFlushVectors);
#elif defined(DNN4_BERT_DATA_PATTERN)
  static_assert(kFfn==1,"the varying-data reference covers FFN1");
  if(argc!=2 || (std::string(argv[1])!="uniform" && std::string(argv[1])!="varying"))
    throw std::runtime_error("usage: probe {uniform|varying}");
  bool varying=std::string(argv[1])=="varying";
#else
  if(argc!=1)throw std::runtime_error("stage probe takes no arguments");
#endif
  Device<E> activation(std::size_t(kM)*kK),weight(std::size_t(kN)*kK),
      residual(std::size_t(kM)*kN),output(std::size_t(kM)*kN);
  Device<float> vectors(3*kN),stats(std::size_t(12)*kM*2),
      partials(std::size_t(kN/64)*kM*2);
  Device<Tick> stamps(2*kGrid);
  // Uniform values make both SW128 pages and every FP32 dot product exact.
  activation.Put(std::vector<E>(activation.count,E(.125f)));
  weight.Put(std::vector<E>(weight.count,E(.125f)));
#ifdef DNN4_BERT_DATA_PATTERN
  constexpr int kPatterns=32;
  std::vector<float> reference(kPatterns*kPatterns,0.f);
  if(varying) {
    auto value=[](unsigned index) {
      unsigned x=index*747796405u+2891336453u;
      x=((x>>((x>>28)+4))^x)*277803737u;
      return float(int((x^(x>>22))%9)-4)/16;
    };
    std::vector<E> a(activation.count),w(weight.count);
    for(int m=0;m<kM;++m)for(int k=0;k<kK;++k)
      a[std::size_t(m)*kK+k]=E(value((m%kPatterns)*kK+k));
    for(int n=0;n<kN;++n)for(int k=0;k<kK;++k) {
      int row=n%kTN,col=k%64;
      std::size_t packed=((n/kTN)*(kK/64)+k/64)*kTN*64+
          row*64+8*((col/8)^(row%8))+col%8;
      w[packed]=E(value((n%kPatterns)*kK+k+1234567u));
    }
    activation.Put(a);weight.Put(w);
    float squares=0;
    for(int part=0;part<12;++part)squares+=768.f*(1.f-kEpsilon)/12;
    float inverse=1.f/std::sqrt(squares/768.f+kEpsilon);
    for(int m=0;m<kPatterns;++m)for(int n=0;n<kPatterns;++n) {
      float dot=0;
      for(int k=0;k<kK;++k)
        dot+=float(a[m*kK+k])*float(E(value(n*kK+k+1234567u)));
      float normalized=dot*inverse;
      reference[m*kPatterns+n]=.5f*normalized*
          (1.f+std::erf(normalized*.70710678118654752f));
    }
  }
#endif
  residual.Put(std::vector<E>(residual.count,E(.125f)));
  output.Put(std::vector<E>(output.count,E(-123.f)));
  std::vector<float> v(vectors.count,0.f),s(stats.count,0.f);
  std::fill(v.begin()+(kFfn==1?0:kN),v.begin()+(kFfn==1?kN:2*kN),1.f);
  for(std::size_t i=1;i<s.size();i+=2)s[i]=768.f*(1.f-kEpsilon)/12;
  vectors.Put(v);stats.Put(s);partials.Put(std::vector<float>(partials.count,-123.f));
  CUtensorMap a{},r{},o{};
  tilemega::codegen::executor::EncodeTiledMap<2>(a,activation.data,{{kK,kM},{2*kK},{64,kTM}});
  tilemega::codegen::executor::EncodeTiledMap<2>(r,residual.data,{{kN,kM},{2*kN},{64,kTM}});
  tilemega::codegen::executor::EncodeTiledMap<2>(o,output.data,{{kN,kM},{2*kN},{64,64}});
  Sm90F3GemmOperands p{};p.weight=weight.data;
  p.epilogue.output=output.data;p.epilogue.output_stride=kN;
  p.epilogue.residual=residual.data;p.epilogue.residual_stride=kN;
  p.epilogue.stats=stats.data;p.epilogue.row_partials=partials.data;
  p.epilogue.norm_width=768;p.epilogue.epsilon=kEpsilon;
  for(int i=0;i<3;++i)p.epilogue.vectors[i]=vectors.data+i*kN;
  Check(cudaFuncSetAttribute(Probe,cudaFuncAttributeMaxDynamicSharedMemorySize,Sm90F3Budget::kBudget));
  int resident=0;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Probe,384,Sm90F3Budget::kBudget));
  if(resident!=1)throw std::runtime_error("stage probe needs one CTA per SM");
  auto launch=[&](){
#ifdef DNN4_BERT_CODE_EVICT
    if(evict) {
      EvictBertCode<<<kGrid,256,Sm90F3Budget::kBudget>>>(sink.data);
      Check(cudaGetLastError());
    }
#elif defined(DNN4_BERT_DATA_EVICT)
    if(evict) {
      EvictBertData<<<kGrid,256>>>(flush.data);
      Check(cudaGetLastError());
    }
#endif
    Probe<<<kGrid,384,Sm90F3Budget::kBudget>>>(p,a,r,o,stamps.data);
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
  double reference_cosine=1,reference_relative=0;
  Tick output_hash=14695981039346656037ull;
#ifdef DNN4_BERT_DATA_PATTERN
  if(varying) {
    double dot=0,aa=0,rr=0;
    for(int m=0;m<kM;++m)for(int n=0;n<kN;++n) {
      double a=float(values[std::size_t(m)*kN+n]),r=reference[(m%kPatterns)*kPatterns+n%kPatterns];
      if(!std::isfinite(a))throw std::runtime_error("nonfinite varying-data output");
      dot+=a*r;aa+=a*a;rr+=r*r;
      reference_relative=std::max(reference_relative,std::abs(a-r)/std::max(std::abs(r),1e-12));
    }
    reference_cosine=dot/std::sqrt(aa*rr);
    if(reference_cosine<.99999 || reference_relative>1e-2)
      throw std::runtime_error("varying-data FP32 reference mismatch");
  }else
#endif
  for(E x:values)if(x.raw()!=rounded.raw())throw std::runtime_error("stage BF16 output mismatch");
  for(E x:values) {
    output_hash^=x.raw()&255;output_hash*=1099511628211ull;
    output_hash^=x.raw()>>8;output_hash*=1099511628211ull;
  }
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
    <<",\"output_elements_checked\":"<<values.size()
    <<",\"epsilon\":"<<kEpsilon
    <<",\"output_hash\":"<<output_hash<<",\"reference_cosine\":"<<reference_cosine
    <<",\"reference_relative\":"<<reference_relative
#ifdef DNN4_BERT_CODE_EVICT
    <<",\"instruction_context\":\""<<(evict?"l1-cold":"hot")<<"\""
#elif defined(DNN4_BERT_DATA_EVICT)
    <<",\"data_context\":\""<<(evict?"data-cold":"hot")<<"\""
#elif defined(DNN4_BERT_DATA_PATTERN)
    <<",\"input_pattern\":\""<<(varying?"varying":"uniform")<<"\""
#endif
    <<"}\n";
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
