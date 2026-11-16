// SPDX-License-Identifier: BSD-3-Clause
// Production attention/small-body work, without the inter-stage grid gate.
#include <tilemega/Backend/Sm90F3Attention.cuh>
#include <tilemega/Backend/Sm90F3Small.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifndef F3_PROBE_FAMILY
#define F3_PROBE_FAMILY 0
#endif
using namespace tilemega::backend;
using E=cutlass::bfloat16_t;
using U64=unsigned long long;
constexpr int kKind=F3_PROBE_FAMILY,kSamples=72,kDiscard=8;
static_assert(kKind>=0 && kKind<=6);
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device {
  T* data;std::size_t count;
  explicit Device(std::size_t n):count(n){Check(cudaMalloc(&data,n*sizeof(T)));}
  ~Device(){cudaFree(data);}
  void Put(std::vector<T> const& v){if(v.size()!=count)throw std::runtime_error("input size");
    Check(cudaMemcpy(data,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
  std::vector<T> Get(){std::vector<T> v(count);Check(cudaMemcpy(v.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
__device__ __forceinline__ U64 Clock(){U64 v;asm volatile("mov.u64 %0, %%clock64;":"=l"(v)::"memory");return v;}
__device__ U64 Timer(){U64 v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v)::"memory");return v;}
__global__ void Calibrate(U64* out){U64 start=Timer(),ticks=Clock();while(Timer()-start<20000000ull){}
  out[0]=Clock()-ticks;out[1]=Timer()-start;}

template<class Body> __global__ __launch_bounds__(288,1)
void Small(Sm90F3SmallOperands p,U64* stamps) {
  __shared__ Sm90F3Header header;
  Sm90F3Phase phase{};
  for(int sample=0;sample<kSamples;++sample) {
    __syncthreads();
    if(threadIdx.x<256) {
      U64 begin=Clock();
      Body::Consume(p,header.rings[0],nullptr,phase,&header,int(threadIdx.x)/128);
      U64 end=Clock();
      if(threadIdx.x==0)stamps[blockIdx.x*kSamples+sample]=end-begin;
    }
  }
}
__global__ __launch_bounds__(288,1) void Attention(
    const __grid_constant__ CUtensorMap q,const __grid_constant__ CUtensorMap k,
    const __grid_constant__ CUtensorMap v,E* output,U64* stamps) {
  extern __shared__ __align__(1024) char shared[];
  auto* header=reinterpret_cast<Sm90F3Header*>(shared);
  if(threadIdx.x==0){header->rings[0].Initialize();Sm90F3Async::Base::InitFence();}
  __syncthreads();
  if(threadIdx.x<256 && (threadIdx.x&31)==0)header->rings[0].Prime();
  __syncthreads();Sm90F3Phase phase{};
  Sm90F3AttentionOperands p{&q,&k,&v,output};
  for(int sample=0;sample<kSamples;++sample) {
    __syncthreads();
    if(threadIdx.x>=256)Sm90F3Attention<1>::Produce(p,header->rings[0],shared+2048,phase);
    else {
      U64 begin=Clock();Sm90F3Attention<1>::Consume(p,header->rings[0],shared+2048,phase);
      U64 end=Clock();if(threadIdx.x==0)stamps[blockIdx.x*kSamples+sample]=end-begin;
    }
  }
  __syncthreads();if(threadIdx.x==0)header->rings[0].Invalidate();
}
template<class Body> void Launch(Sm90F3SmallOperands p,U64* stamps,int grid) {
  Small<Body><<<grid,288>>>(p,stamps);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
}

int main(int,char**)try {
  if(!std::getenv("TILEMEGA_RUN_NONCE") || !std::getenv("TILEMEGA_GPU_LOCK_HELD"))
    throw std::runtime_error("calibration requires scheduler/guard");
  constexpr int grid=kKind==6?12:kKind==3?32:(kKind==1 || kKind==2)?8:1;
  constexpr int items=kKind==6?1:(kKind==4 || kKind==5)?8:256;
  Device<U64> stamps(grid*kSamples);
  Calibrate<<<1,1>>>(stamps.data);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  auto calibration=stamps.Get();double cycles_per_ns=double(calibration[0])/calibration[1];
  std::vector<E> expected;
  if constexpr(kKind==6) {
    Device<E> input(128*12*3*64),output(128*12*64);input.Put(std::vector<E>(input.count,E(.125f)));
    CUtensorMap maps[3]{};
    for(int part=0;part<3;++part)tilemega::codegen::executor::EncodeTiledMap<3>(maps[part],input.data+part*64,
        {{64,12,128},{3*64*2,3*12*64*2},{64,1,128}});
    constexpr int bytes=2048+Sm90F3Attention<1>::kSharedBytes;
    Check(cudaFuncSetAttribute(Attention,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    Attention<<<grid,288,bytes>>>(maps[0],maps[1],maps[2],output.data,stamps.data);
    Check(cudaGetLastError());Check(cudaDeviceSynchronize());
    for(auto v:output.Get())if(float(v)!=.125f)throw std::runtime_error("attention task output mismatch");
  }else {
    constexpr int in_count=kKind==0?2048*3:(kKind==1 || kKind==2)?32*32*64:8*768;
    constexpr int out_count=kKind==0?2048*8:(kKind==1 || kKind==2)?16*16*64:kKind==3?16*512:8*768;
    Device<E> input(in_count),output(out_count),weight(8*768),table(8*768);
    Device<float> stats(16),partial(16*2*512);Device<long long> ids(8);
    input.Put(std::vector<E>(in_count,E(kKind==5?.125f:1.f)));
    weight.Put(std::vector<E>(weight.count,E(kKind==5?.125f:1.f)));
    table.Put(std::vector<E>(table.count,E(kKind==5?.125f:0.f)));
    ids.Put({0,1,2,3,4,5,6,7});
    Sm90F3SmallOperands p{};p.input=input.data;p.output=output.data;p.weight=weight.data;p.table=table.data;
    p.stats=stats.data;p.accumulator=partial.data;p.index0=ids.data;p.rows=8;p.height=8;p.epsilon=1e-5f;
    if constexpr(kKind==0) {p.width=64;p.height=32;Launch<Sm90F3Layout<3,1>>(p,stamps.data,grid);}
    if constexpr(kKind==1 || kKind==2) {p.width=p.height=32;
      if constexpr(kKind==1)Launch<Sm90F3MaxPool<2,2,2,0,64,1>>(p,stamps.data,grid);
      else Launch<Sm90F3MaxPool<3,3,2,1,64,1>>(p,stamps.data,grid);}
    if constexpr(kKind==3) {
      std::vector<float> values(partial.count,0.f);
      for(int image=0;image<16;++image) {
        int begin=image*49,end=begin+49,first=begin/64,last=(end-1)/64;
        for(int part=first;part<=last;++part)for(int c=0;c<512;++c)
          values[(image*2+part-first)*512+c]=float(std::min(end,(part+1)*64)-std::max(begin,part*64));
      }
      partial.Put(values);p.channels=p.out_channels=512;p.rows=49;p.parts=2;
      Launch<Sm90F3GlobalPool<16>>(p,stamps.data,grid);
    }
    if constexpr(kKind==4)Launch<Sm90F3LayerNorm<768,768>>(p,stamps.data,grid);
    if constexpr(kKind==5)Launch<Sm90F3EmbeddingSum<768>>(p,stamps.data,grid);
    auto actual=output.Get();
    for(std::size_t i=0;i<actual.size();++i) {
      float ref=kKind==0?(i%8<3?1.f:0.f):kKind==4?0.f:kKind==5?.375f:1.f;
      if(float(actual[i])!=ref)throw std::runtime_error("small-body output mismatch");
    }
    if constexpr(kKind==5) {
      auto raw=stats.Get();for(int r=0;r<8;++r)
        if(raw[2*r]!=288.f || raw[2*r+1]!=108.f)throw std::runtime_error("embedding row statistics mismatch");
    }
  }
  auto raw=stamps.Get();std::vector<double> ns;
  for(int block=0;block<grid;++block)for(int sample=kDiscard;sample<kSamples;++sample)
    ns.push_back(raw[block*kSamples+sample]/cycles_per_ns/items);
  std::sort(ns.begin(),ns.end());
  char const* kinds[]={"layout","pool2","pool3","global_pool","layernorm","embedding","attention"};
  std::cout<<std::setprecision(17)<<"{\"work_items_per_cta\":"<<items<<",\"grid\":"<<grid
      <<",\"cycles_per_ns\":"<<cycles_per_ns<<",\"rows\":[{\"metric\":\""
      <<(kKind==6?"f3_attention_task_ns":"f3_small_ns")<<"\",\"kind\":\""<<kinds[kKind]
      <<"\",\"ns\":"<<ns[ns.size()/2]<<",\"samples_ns\":[";
  for(std::size_t i=0;i<ns.size();++i)std::cout<<(i?",":"")<<ns[i];
  std::cout<<"]}]}\n";return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
