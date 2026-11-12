// SPDX-License-Identifier: BSD-3-Clause
// Recheck F4 constants with BERT's admitted P1 and matrix-store profile.
#define TILEMEGA_F3_SIDE_NORM 1
#define TILEMEGA_F3_REGISTER_BUDGET 3
#define TILEMEGA_F3_CYCLIC_SLOTS 1
#define TILEMEGA_F3_MATRIX_STORE 1
#define TILEMEGA_F3_H800_GRID 1
#define TILEMEGA_F3_ATTENTION_VECTOR_STORE 1
#include <tilemega/Backend/Sm90F3Mark.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using U64=unsigned long long;
__device__ U64* f4_epilogue_stamps;
__device__ unsigned f4_epilogue_task[132];
__device__ __forceinline__ U64 Clock(){U64 x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x));return x;}
#undef TM_F3_TRACE_BEGIN
#undef TM_F3_TRACE_END
#define TM_F3_TRACE_BEGIN(k) F4_BEGIN_##k
#define TM_F3_TRACE_END(k) F4_END_##k
#define F4_BEGIN_kSlotWait
#define F4_END_kSlotWait
#define F4_BEGIN_kMainLoop
#define F4_END_kMainLoop
#define F4_BEGIN_kSideWait
#define F4_END_kSideWait
#define F4_BEGIN_kSplitK
#define F4_END_kSplitK
#define F4_BEGIN_kStore
#define F4_END_kStore
#define F4_BEGIN_kEpilogue U64 f4_begin=Clock()
#define F4_END_kEpilogue if(threadIdx.x==0)f4_epilogue_stamps[3*blockIdx.x+f4_epilogue_task[blockIdx.x]++]=Clock()-f4_begin
#include <tilemega/Backend/Sm90F3Gemm.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
using namespace tilemega::backend;
using E=cutlass::bfloat16_t;
namespace ex=tilemega::codegen::executor;
void Check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
template<class T>struct Device {
 T* data=nullptr;std::size_t count;
 explicit Device(std::size_t n):count(n){Check(cudaMalloc(&data,n*sizeof(T)));}
 ~Device(){cudaFree(data);}
 void Put(std::vector<T> const& v){Check(cudaMemcpy(data,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T> Get(){std::vector<T> v(count);Check(cudaMemcpy(v.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
template<int TM,int TN,int Kind>struct ChainOf {
 static constexpr bool residual=Kind==2 || Kind==5,stats=Kind>=3;
 static constexpr int vectors=Kind==0?0:Kind==5?3:2;
 using Side=Sm90F3SideLayout<TM,TN,residual,vectors,stats,12>;
 using Base=std::conditional_t<Kind==0,Sm90F3Chain<Side>,
  std::conditional_t<Kind==1,Sm90F3Chain<Side,Sm90F3Step<Sm90F3Op::kScale,0>,Sm90F3Step<Sm90F3Op::kBias,1>,Sm90F3Step<Sm90F3Op::kActivation>>,
  std::conditional_t<Kind==2,Sm90F3Chain<Side,Sm90F3Step<Sm90F3Op::kScale,0>,Sm90F3Step<Sm90F3Op::kBias,1>,Sm90F3Step<Sm90F3Op::kResidual,4294967295u>,Sm90F3Step<Sm90F3Op::kActivation>>,
  std::conditional_t<Kind==3,Sm90F3Chain<Side,Sm90F3Step<Sm90F3Op::kDeferredLayerNorm,0,tilemega::codegen::DmActivation::kRelu,1>>,
  std::conditional_t<Kind==4,Sm90F3Chain<Side,Sm90F3Step<Sm90F3Op::kDeferredLayerNorm,0,tilemega::codegen::DmActivation::kRelu,1>,Sm90F3Step<Sm90F3Op::kActivation,0,tilemega::codegen::DmActivation::kGeluErf>>,
  Sm90F3Chain<Side,Sm90F3Step<Sm90F3Op::kBias,0>,Sm90F3Step<Sm90F3Op::kResidualLN,1,tilemega::codegen::DmActivation::kRelu,2>>>>>>>;
 using Type=std::conditional_t<Kind==5,Sm90F3WriteRowStats<Base>,Base>;
};
template<int TM,int TN,int Kind,bool Store>using Body=Sm90F3Gemm<TM,TN,2,1,Sm90F3Mode::kNormal,
    132*3*TM,768,64,768,typename ChainOf<TM,TN,Kind>::Type,Sm90F3NoConv,Sm90F3NoPool,64,Store>;
template<int TM,int TN,int Kind,bool Store>
__global__ __launch_bounds__(384,1) void Probe(Sm90F3GemmOperands p,float value,
    const __grid_constant__ CUtensorMap residual,const __grid_constant__ CUtensorMap output) {
 using B=Body<TM,TN,Kind,Store>;
 extern __shared__ __align__(1024) char shared[];
 auto* h=reinterpret_cast<Sm90F3Header*>(shared);auto& ring=h->rings[0];
 if(threadIdx.x==0){ring.Initialize();Sm90F3Async::Base::InitFence();f4_epilogue_task[blockIdx.x]=0;}
 __syncthreads();
 if(threadIdx.x<256 && (threadIdx.x&31)==0)ring.Prime();
 __syncthreads();
 p.residual=&residual;p.output_map=&output;
 auto* arena=reinterpret_cast<E*>(shared+Sm90F3Budget::kArena0);
 auto* side=reinterpret_cast<char*>(arena)+B::kSideOffset;Sm90F3Phase phase{};
 if(threadIdx.x>=256) {
  Sm90F3Async::RegDealloc<72>();
  if(threadIdx.x>=288 && threadIdx.x<320)for(int task=0;task<3;++task) {
   if(threadIdx.x==288)B::ProduceSide(p,ring,side,phase,int(blockIdx.x),0,TM);
   if constexpr(B::kSideNorm)B::PrepareSideNorm(p,ring,side,phase,TM);
  }
 }else {
  Sm90F3Async::RegAlloc<216>();int store_count=0;
  for(int task=0;task<3;++task) {
   typename B::Accumulators accum;
   #pragma unroll
   for(int sub=0;sub<B::Split::kSubTiles;++sub)cute::fill(accum.tile[sub],value);
   B::Epilogue(p,ring,side,phase,accum,int(threadIdx.x)/128,
       int(blockIdx.x)+132*task,0,TM,TN,h,arena,&store_count);
  }
  if constexpr(Store)if((threadIdx.x&127)==0) {
   Sm90F3Async::Base::BulkWait<0,false>();Sm90F3Async::Base::ProxyAsyncGlobalFence();
  }
 }
 __syncthreads();if(threadIdx.x==0)ring.Invalidate();
}
__global__ void Calibrate(U64* p) {
 U64 start,now;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(start));U64 clock=Clock();
 do{asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));}while(now-start<20000000ull);
 p[0]=Clock()-clock;p[1]=now-start;
}
template<int TM,int TN,int Kind,bool Store>int Run() {
 constexpr int M=132*3*TM,parts=12;
 Device<E> output(std::size_t(M)*768),residual(std::size_t(132*TM)*768);
 Device<float> vectors(768*3),stats(std::size_t(parts)*M*2),partials(std::size_t(parts)*M*2);
 Device<U64> stamps(132*3);
 std::vector<float> vs(vectors.count,0),ss(stats.count,0);
 if constexpr(Kind==5)std::fill(vs.begin()+768,vs.begin()+1536,1.f);
 else std::fill(vs.begin(),vs.begin()+768,1.f);
 for(std::size_t i=1;i<ss.size();i+=2)ss[i]=768.f*(1.f-1e-5f)/parts;
 vectors.Put(vs);stats.Put(ss);residual.Put(std::vector<E>(residual.count,E(.125f)));
 output.Put(std::vector<E>(output.count,E(-123)));
 CUtensorMap r{},o{};ex::EncodeTiledMap<2>(r,residual.data,{{768,132*TM},{1536},{64,TM}});
 ex::EncodeTiledMap<2>(o,output.data,{{768,M},{1536},{64,64}});
 Sm90F3GemmOperands p{};p.epilogue.output=output.data;p.epilogue.output_stride=768;
 p.epilogue.residual=residual.data;p.epilogue.residual_stride=768;p.epilogue.stats=stats.data;
 p.epilogue.row_partials=partials.data;p.epilogue.norm_width=768;p.epilogue.epsilon=1e-5f;
 for(int i=0;i<3;++i)p.epilogue.vectors[i]=vectors.data+768*i;
 Check(cudaMemcpyToSymbol(f4_epilogue_stamps,&stamps.data,sizeof(stamps.data)));
 Calibrate<<<1,1>>>(stamps.data);Check(cudaDeviceSynchronize());auto cal=stamps.Get();double cycles_per_ns=double(cal[0])/cal[1];
 auto kernel=Probe<TM,TN,Kind,Store>;
 Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Sm90F3Budget::kBudget));
 for(int i=0;i<20;++i)kernel<<<132,384,Sm90F3Budget::kBudget>>>(p,.125f,r,o);
 Check(cudaGetLastError());Check(cudaDeviceSynchronize());std::vector<double> times;
 for(int sample=0;sample<16;++sample) {
  kernel<<<132,384,Sm90F3Budget::kBudget>>>(p,.125f,r,o);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  auto raw=stamps.Get();std::sort(raw.begin(),raw.end());times.push_back(raw[raw.size()/2]/cycles_per_ns);
 }
 auto values=output.Get();float expected=Kind==2 || Kind==5?.25f:.125f;
 if constexpr(Kind==4)expected=.5f*expected*(1.f+std::erf(expected*.70710678118654752f));
 E rounded(expected);
 for(int m=0;m<M;++m)for(int n=0;n<TN;++n)
  if(values[std::size_t(m)*768+n].raw()!=rounded.raw())throw std::runtime_error("in-situ epilogue output differs from BF16 reference");
 std::sort(times.begin(),times.end());
 std::cout<<std::setprecision(17)<<"{\"tile_m\":"<<TM<<",\"tile_n\":"<<TN<<",\"class_index\":"<<Kind
     <<",\"store\":\""<<(Store?"tma":"reg")<<"\",\"ns\":"<<times[times.size()/2]<<",\"active_ctas\":132,\"tasks_per_cta\":3}\n";
 return 0;
}
int main(int argc,char** argv)try {
 if(!std::getenv("TILEMEGA_RUN_NONCE") || !std::getenv("TILEMEGA_GPU_LOCK_HELD"))throw std::runtime_error("epilogue calibration requires guard");
 if(argc!=2)throw std::runtime_error("choose an epilogue case");
 switch(std::stoi(argv[1])) {
#include "dnn4_bert_epilogue_cases.inc"
 default:throw std::runtime_error("invalid epilogue case");
 }
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
