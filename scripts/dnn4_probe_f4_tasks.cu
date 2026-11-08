// SPDX-License-Identifier: BSD-3-Clause
// In-situ F4 small-body and attention service, without the grid gate.
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
#ifndef DNN4_TASK_KIND
#define DNN4_TASK_KIND 0
#endif
#ifndef DNN4_ATTN_ROWS
#define DNN4_ATTN_ROWS 128
#endif
#ifndef DNN4_TASK_BATCH
#define DNN4_TASK_BATCH 16
#endif
using namespace tilemega::backend;
using E=cutlass::bfloat16_t;
using U64=unsigned long long;
constexpr int kKind=DNN4_TASK_KIND,kRows=DNN4_ATTN_ROWS;
constexpr int kGrid=132,kSamples=32,kDiscard=8,kBatch=DNN4_TASK_BATCH;
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device {
 T* data;std::size_t count;
 explicit Device(std::size_t n):count(n){Check(cudaMalloc(&data,n*sizeof(T)));}
 ~Device(){cudaFree(data);}
 void Put(std::vector<T> const& v){if(v.size()!=count)throw std::runtime_error("input size");Check(cudaMemcpy(data,v.data(),count*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T> Get(){std::vector<T> v(count);Check(cudaMemcpy(v.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
__device__ __forceinline__ U64 Clock(){U64 v;asm volatile("mov.u64 %0, %%clock64;":"=l"(v)::"memory");return v;}
__device__ U64 Timer(){U64 v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v)::"memory");return v;}
__global__ void Calibrate(U64* p){U64 start=Timer(),clock=Clock();while(Timer()-start<20000000ull){}p[0]=Clock()-clock;p[1]=Timer()-start;}
template<class Body> __global__ __launch_bounds__(384,1)
void Small(Sm90F3SmallOperands p,U64* stamps) {
 __shared__ Sm90F3Header h;Sm90F3Phase phase{};
 for(int sample=0;sample<kSamples;++sample) {
  __syncthreads();
  if(threadIdx.x<256) {
   U64 start=Clock();Body::Consume(p,h.rings[0],nullptr,phase,&h,int(threadIdx.x)/128);
   Sm90F3Async::NamedBarrier<2,256>();
   if(threadIdx.x==0)stamps[blockIdx.x*kSamples+sample]=Clock()-start;
  }
 }
}
using Attn=Sm90F3Attention<kBatch,kRows>;
__global__ __launch_bounds__(384,1) void Attention(
 const __grid_constant__ CUtensorMap q,const __grid_constant__ CUtensorMap k,
 const __grid_constant__ CUtensorMap v,E* output,U64* stamps) {
 extern __shared__ __align__(1024) char shared[];
 auto* h=reinterpret_cast<Sm90F3Header*>(shared);
 if(threadIdx.x==0){h->rings[0].Initialize();Sm90F3Async::Base::InitFence();}
 __syncthreads();
 if(threadIdx.x<256 && (threadIdx.x&31)==0)h->rings[0].Prime();
 __syncthreads();Sm90F3Phase phase{};
 Sm90F3AttentionOperands p{&q,&k,&v,output};
 if(threadIdx.x>=256) {
  Sm90F3Async::RegDealloc<56>();
  for(int sample=0;sample<kSamples;++sample) {
   __syncthreads();Attn::Produce(p,h->rings[0],shared+Sm90F3Budget::kArena0,phase);
  }
 }else {
  Sm90F3Async::RegAlloc<224>();
  for(int sample=0;sample<kSamples;++sample) {
   __syncthreads();U64 start=Clock();
   Attn::Consume(p,h->rings[0],shared+Sm90F3Budget::kArena0,phase);
   Sm90F3Async::NamedBarrier<2,256>();
   if(threadIdx.x==0)stamps[blockIdx.x*kSamples+sample]=Clock()-start;
  }
 }
 __syncthreads();if(threadIdx.x==0)h->rings[0].Invalidate();
}
template<class Body> void Launch(Sm90F3SmallOperands p,U64* stamps){Small<Body><<<kGrid,384>>>(p,stamps);Check(cudaGetLastError());Check(cudaDeviceSynchronize());}
int main()try {
 if(!std::getenv("TILEMEGA_RUN_NONCE") || !std::getenv("TILEMEGA_GPU_LOCK_HELD"))throw std::runtime_error("task calibration requires scheduler/guard");
 Device<U64> stamps(kGrid*kSamples);Calibrate<<<1,1>>>(stamps.data);Check(cudaDeviceSynchronize());auto cal=stamps.Get();double cycles=double(cal[0])/cal[1];
 long long work_items=0;
 if constexpr(kKind==6) {
  Device<E> input(kBatch*128*12*3*64),output(kBatch*128*12*64);input.Put(std::vector<E>(input.count,E(.125f)));
  CUtensorMap maps[3]{};
  for(int part=0;part<3;++part)tilemega::codegen::executor::EncodeTiledMap<3>(maps[part],input.data+part*64,{{64,12,kBatch*128},{3*64*2,3*12*64*2},{64,1,unsigned(part?128:kRows)}});
  constexpr int bytes=Sm90F3Budget::kArena0+Attn::kSharedBytes;
  Check(cudaFuncSetAttribute(Attention,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  Attention<<<kGrid,384,bytes>>>(maps[0],maps[1],maps[2],output.data,stamps.data);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  for(auto v:output.Get())if(float(v)!=.125f)throw std::runtime_error("attention output mismatch");work_items=Attn::kTasks;
 }else {
  constexpr int rows=kGrid*3*8;
  constexpr int in_count=kKind==0?kBatch*224*224*3:(kKind==1 || kKind==2)?kBatch*112*112*64:rows*768;
  constexpr int out_count=kKind==0?kBatch*224*224*8:(kKind==1 || kKind==2)?kBatch*56*56*64:kKind==3?kGrid*3*256:rows*768;
  Device<E> input(in_count),output(out_count),weight(rows*768),table(rows*768);
  Device<float> stats(rows*2),partial(kGrid*3*256*2);Device<long long> ids(rows);
  input.Put(std::vector<E>(in_count,E(kKind==5?.125f:1.f)));weight.Put(std::vector<E>(weight.count,E(kKind==5?.125f:1.f)));table.Put(std::vector<E>(table.count,E(kKind==5?.125f:0.f)));
  std::vector<long long> indices(rows);for(int i=0;i<rows;++i)indices[i]=i;ids.Put(indices);
  Sm90F3SmallOperands p{};p.input=input.data;p.output=output.data;p.weight=weight.data;p.table=table.data;p.stats=stats.data;p.accumulator=partial.data;p.index0=ids.data;p.rows=rows;p.height=rows;p.epsilon=1e-5f;
  if constexpr(kKind==0){p.width=p.height=224;Launch<Sm90F3Layout<3,kBatch>>(p,stamps.data);work_items=kBatch*224*224/8;}
  if constexpr(kKind==1 || kKind==2){p.width=p.height=112;
   if constexpr(kKind==1)Launch<Sm90F3MaxPool<2,2,2,0,64,kBatch,112,112>>(p,stamps.data);
   else Launch<Sm90F3MaxPool<3,3,2,1,64,kBatch,112,112>>(p,stamps.data);work_items=kBatch*56*56*8;}
  if constexpr(kKind==3){
   std::vector<float> parts(partial.count,0.f);
   for(int image=0;image<198;++image){int begin=image*49,end=begin+49,first=begin/64,last=(end-1)/64;
    for(int part=first;part<=last;++part)for(int c=0;c<512;++c)parts[(image*2+part-first)*512+c]=float(std::min(end,(part+1)*64)-std::max(begin,part*64));}
   partial.Put(parts);p.channels=p.out_channels=512;p.rows=49;p.parts=2;Launch<Sm90F3GlobalPool<198>>(p,stamps.data);work_items=kGrid*3*256;
  }
  if constexpr(kKind==4){Launch<Sm90F3LayerNorm<768,768>>(p,stamps.data);work_items=rows;}
  if constexpr(kKind==5){Launch<Sm90F3EmbeddingSum<768>>(p,stamps.data);work_items=rows;}
  auto values=output.Get();
  for(std::size_t i=0;i<values.size();++i){float expected=kKind==0?(i%8<3?1.f:0.f):kKind==4?0.f:kKind==5?.375f:1.f;if(float(values[i])!=expected)throw std::runtime_error("small body output mismatch at "+std::to_string(i));}
  if constexpr(kKind==5)for(auto v:stats.Get())if(v!=288.f && v!=108.f)throw std::runtime_error("embedding statistics mismatch");
 }
 auto raw=stamps.Get();std::vector<double> ns;
 for(int block=0;block<kGrid;++block) {
  long long group=kKind==6?1:(kKind==4 || kKind==5)?8:kKind==1 || kKind==2?1024:256;
  long long groups=(work_items+group-1)/group;
  long long count=0;for(long long g=block;g<groups;g+=kGrid)count+=std::min(group,work_items-g*group);
  if(count)for(int sample=kDiscard;sample<kSamples;++sample)ns.push_back(raw[block*kSamples+sample]/cycles/count);
 }
 std::sort(ns.begin(),ns.end());char const* names[]={"layout","pool2","pool3","global_pool","layernorm","embedding","attention"};
 std::vector<double> windows;
 for(int sample=kDiscard;sample<kSamples;++sample) {
   U64 longest=0;
   for(int block=0;block<kGrid;++block)longest=std::max(longest,raw[block*kSamples+sample]);
   windows.push_back(longest/cycles);
 }
 std::sort(windows.begin(),windows.end());
 std::cout<<std::setprecision(17)<<"{\"kind\":\""<<names[kKind]<<"\",\"rows\":"<<kRows<<",\"ns\":"<<ns[ns.size()/2]<<",\"window_ns\":"<<windows[windows.size()/2]<<",\"grid\":132,\"work_items\":"<<work_items<<",\"cycles_per_ns\":"<<cycles<<"}\n";
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
