// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Backend/Sm90DenseAttentionBody.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace tilemega;
using BF=cutlass::bfloat16_t;
namespace be=tilemega::backend;
namespace ex=tilemega::codegen::executor;
inline void Check(cudaError_t s) {if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
template<class T>T* Alloc(std::size_t n) {T* p;Check(cudaMalloc(&p,n*sizeof(T)));return p;}
__global__ void Fill(BF* p,std::size_t n,unsigned seed) {
  for(std::size_t i=blockIdx.x*256+threadIdx.x;i<n;i+=gridDim.x*256) {
    unsigned h=unsigned(i)^seed;h^=h>>16;h*=0x7feb352dU;h^=h>>15;h*=0x846ca68bU;h^=h>>16;
    p[i]=BF((float(h&65535)/65535.0f-.5f)*.04f);
  }
}
__global__ void Evict(unsigned* p) {
  for(unsigned i=blockIdx.x*256+threadIdx.x;i<(256u<<18);i+=gridDim.x*256)
    asm volatile("st.global.cg.u32 [%0], %1;"::"l"(p+i),"r"(i):"memory");
}
struct Bench {
  unsigned* flush=Alloc<unsigned>(256u<<18);
  cudaEvent_t start,end;cudaStream_t stream;
  Bench() {Check(cudaEventCreate(&start));Check(cudaEventCreate(&end));Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
  ~Bench() {cudaFree(flush);cudaEventDestroy(start);cudaEventDestroy(end);cudaStreamDestroy(stream);}
  template<class Launch>void Measure(char const* id,int past,int smem,Launch launch) {
    cudaGraph_t graph;cudaGraphExec_t exec;
    Check(cudaDeviceSynchronize());
    Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(stream);
    Check(cudaStreamEndCapture(stream,&graph));Check(cudaGraphInstantiate(&exec,graph,0));
    std::vector<double> samples;
    for(int i=-3;i<15;++i) {
      Evict<<<132,256,0,stream>>>(flush);
      Check(cudaEventRecord(start,stream));Check(cudaGraphLaunch(exec,stream));Check(cudaEventRecord(end,stream));
      Check(cudaEventSynchronize(end));float ms;Check(cudaEventElapsedTime(&ms,start,end));
      if(i>=0)samples.push_back(double(ms)*1000);
    }
    std::printf("{\"candidate\":\"%s\",\"past\":%d,\"shared_bytes\":%d,\"samples_us\":[",id,past,smem);
    for(unsigned i=0;i<samples.size();++i)std::printf("%s%.9g",i?",":"",samples[i]);
    std::printf("]}\n");std::fflush(stdout);
    Check(cudaGraphExecDestroy(exec));Check(cudaGraphDestroy(graph));
  }
};
template<class G,bool Head>
__global__ __launch_bounds__(160,1) void Linear(be::Sm90DenseOperands p,be::Sm90DenseHeadOperands h) {
  extern __shared__ __align__(1024) char s[];
  auto* ring=reinterpret_cast<be::Sm90DenseRingState*>(s);ring->Initialize();
  if constexpr(Head)G::template Run<0>(p,s,ring,be::Sm90DenseNoTail{},false,
      be::Sm90DenseHeadFinish<1,2048,1088,false>{&h});
  else G::template Run<0>(p,s,ring,be::Sm90DenseNoTail{},false);
  ring->Invalidate();
}
template<int N,int K,int TN,int TK,int WN,int Slots,int RS,bool Norm,bool Head>
void GemmCase(Bench& bench,char const* id) {
  using G=be::Sm90DenseGemm<1,N,K,TN,TK,Slots,WN,4/WN,
      Head?be::ServingEpilogueOp::kArgmaxPartial:be::ServingEpilogueOp::kStore,8,Norm,RS,false,Norm>;
  auto* weights=Alloc<BF>(std::size_t(N)*K);auto* input=Alloc<BF>(K);
  auto* output=Alloc<BF>(Head?264:N);auto* indices=Alloc<int>(132);
  auto* inverse=Alloc<float>(1);float one=1;Check(cudaMemcpy(inverse,&one,4,cudaMemcpyHostToDevice));
  auto* counter=Alloc<std::uint64_t>(1);Check(cudaMemset(counter,0,8));auto* tokens=Alloc<int>(1088);
  Fill<<<132,256>>>(weights,std::size_t(N)*K,0);Fill<<<132,256>>>(input,K,1);
  CUtensorMap map;ex::EncodeTiledMap<3>(map,input,{{64,1,K/64},{2*K,128},{64,8,TK/64}});
  auto* device_map=Alloc<CUtensorMap>(1);Check(cudaMemcpy(device_map,&map,sizeof(map),cudaMemcpyHostToDevice));
  be::Sm90DenseOperands p{device_map,weights,reinterpret_cast<BF*>(inverse),output,indices,1e-6f};
  be::Sm90DenseHeadOperands h{tokens,counter,nullptr,nullptr,575};
  auto kernel=Linear<G,Head>;Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,G::kSharedBytes));
  int resident;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,160,G::kSharedBytes));
  if(resident<1)throw std::runtime_error("candidate has no resident CTA");
  bench.Measure(id,-1,G::kSharedBytes,[&](cudaStream_t stream) {kernel<<<132,160,G::kSharedBytes,stream>>>(p,h);});
  for(void* ptr:{(void*)weights,(void*)input,(void*)output,(void*)indices,(void*)inverse,(void*)counter,(void*)tokens,(void*)device_map})Check(cudaFree(ptr));
}
template<class A>
__global__ __launch_bounds__(160,1) void Attention(be::Sm90DenseAttentionOperands p) {
  extern __shared__ __align__(1024) char s[];
  auto* ring=reinterpret_cast<be::Sm90DenseRingState*>(s);ring->Initialize();
  A::template Run<0>(p,s,ring,be::Sm90DenseNoTail{},false);ring->Invalidate();
}
template<class A>
__global__ void Merge(be::Sm90DenseAttentionOperands p) {
  extern __shared__ __align__(1024) char s[];
  if(blockIdx.x<8 && threadIdx.x<128)A::Merge(p,*reinterpret_cast<typename A::Storage*>(s+1024),blockIdx.x);
}
template<int S,int BK,int Slots,bool Stage>
void AttentionCase(Bench& bench,char const* id) {
  using A=be::Sm90DenseAttentionBody<1,8,128,2,true,S,BK,Slots,0,1088,false,Stage,true>;
  auto* qkv=Alloc<BF>(4096);auto* key=Alloc<BF>(8*1088*128);auto* value=Alloc<BF>(8*1088*128);
  auto* out=Alloc<BF>(2048);auto* partial=Alloc<float>(8*S*256);auto* lse=Alloc<float>(8*S*2);
  auto* arrivals=Alloc<std::uint64_t>(8);Check(cudaMemset(arrivals,0,64));
  Fill<<<132,256>>>(qkv,4096,2);Fill<<<132,256>>>(key,8*1088*128,3);Fill<<<132,256>>>(value,8*1088*128,4);
  CUtensorMap maps[2];
  ex::EncodeTiledMap<4>(maps[0],key,{{128,1088,8,1},{256,278528,2228224},{64,BK,1,1}});
  ex::EncodeTiledMap<4>(maps[1],value,{{128,1088,8,1},{256,278528,2228224},{64,BK,1,1}});
  auto* dm=Alloc<CUtensorMap>(2);Check(cudaMemcpy(dm,maps,sizeof(maps),cudaMemcpyHostToDevice));
  auto kernel=Attention<A>;Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,A::kSharedBytes));
  if constexpr(Stage && S>1)Check(cudaFuncSetAttribute(Merge<A>,cudaFuncAttributeMaxDynamicSharedMemorySize,A::kSharedBytes));
  for(int past:{165,370,574,779,984}) {
    be::Sm90DenseAttentionOperands p{dm,dm+1,qkv,key,value,out,nullptr,nullptr,nullptr,nullptr,partial,lse,arrivals,past,1e-6f,0};
    bench.Measure(id,past,A::kSharedBytes,[&](cudaStream_t stream) {
      kernel<<<132,160,A::kSharedBytes,stream>>>(p);
      if constexpr(Stage && S>1)Merge<A><<<132,160,A::kSharedBytes,stream>>>(p);
    });
  }
  for(void* ptr:{(void*)qkv,(void*)key,(void*)value,(void*)out,(void*)partial,(void*)lse,(void*)arrivals,(void*)dm})Check(cudaFree(ptr));
}
