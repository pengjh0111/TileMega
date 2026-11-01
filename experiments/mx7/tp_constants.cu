// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeHeadV2.h>
#include <mcr/mc_runtime_api.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
#ifndef MX7_TP
#define MX7_TP 2
#define MX7_BATCH 1
#endif
using namespace tilemega;
using namespace tilemega::backend::mx;
using Sync=tilemega::codegen::mx::StageSyncV2<512>;
constexpr int T=MX7_TP,B=MX7_BATCH,Iterations=1000,Hidden=2048,Vocab=151936/T,CoreBytes=32784;
using GemmOps=MxTpOperandsV2<MxGemmOperandsV2>;
using HeadOps=MxTpHeadOperandsV2<MxHeadOperandsV2>;
struct Args{GemmOps gemm;HeadOps head;MxTpHeadLinkV2 control;unsigned long long *counter,*stamps;};
__device__ __forceinline__ void Exchange(MxTpHeadLinkV2 const& x,unsigned e){
  auto* self=MxTpSelfV2(x.x,x.rank);
#pragma unroll
  for(int q=0;q<kMxTpMaxRanks;++q)if(q<x.world&&q!=x.rank){
    maca::PeerStore64(&x.x[q]->data[e&1][x.rank][0][0],std::uint64_t(e)<<32);
    maca::PeerStore64(&x.x[q]->data[e&1][x.rank][0][1],std::uint64_t(e)<<32);
  }
#pragma unroll
  for(int q=0;q<kMxTpMaxRanks;++q)if(q<x.world&&q!=x.rank){
    MxTpAwait(&self->data[e&1][q][0][0],e,{},21);
    MxTpAwait(&self->data[e&1][q][0][1],e,{},21);
  }
}
template<int TN,int K,bool AR>
__global__ void __launch_bounds__(512) Gemm(Args a){
  unsigned const grid=gridDim.x;__builtin_assume(grid==104);
  extern __shared__ __align__(16) char memory[];
  auto* gen=(unsigned*)(memory+CoreBytes);if(Sync::IsRole())*gen=0;Sync::CtaSync();
  using Shape=MxStaticShapeV2<Hidden,K,B,512>;
  using Body=MxGemm<TN,1,1,AR?MxOpV2::kResidualAllReduce:MxOpV2::kResidual,false,MxPart::kTileRange>;
  for(unsigned i=0;i<Iterations;++i){
    auto p=a.gemm;p.tp.epoch=i+1;
    auto pf=Body::Prefetch(p,Shape{});
    tilemega::codegen::mx::StageBarrierV2<Sync>(a.counter,(3ull*i+1)*104,gen,3*i+1,[]{},{});
    unsigned long long const start=maca::GlobalTimerNs();
    Body::Run(pf,p,Shape{},memory,[]{return MxNoPrefetchedV2{};});
    tilemega::codegen::mx::StageBarrierV2<Sync>(a.counter,(3ull*i+2)*104,gen,3*i+2,[]{},{});
    if(blockIdx.x==0&&Sync::IsRole()){a.stamps[i]=maca::GlobalTimerNs()-start;Exchange(a.control,i+1);}
    // The existing LL head handshake is outside the measured interval. It
    // makes buffer reuse causal on every rank in both comparison arms.
    tilemega::codegen::mx::StageBarrierV2<Sync>(a.counter,(3ull*i+3)*104,gen,3*i+3,[]{},{});
  }
}
template<bool TP>
__global__ void __launch_bounds__(512) Head(Args a){
  unsigned const grid=gridDim.x;__builtin_assume(grid==104);
  extern __shared__ __align__(16) char memory[];
  constexpr int core=CoreBytes>MxHead<16,8,1>::SmemBytes(B)?CoreBytes:MxHead<16,8,1>::SmemBytes(B);
  auto* gen=(unsigned*)(memory+core);if(Sync::IsRole())*gen=0;Sync::CtaSync();
  using Shape=MxStaticShapeV2<Vocab,Hidden,B,512>;using Body=MxHead<16,8,1,1,TP?T:1,Vocab>;
  for(unsigned i=0;i<Iterations;++i){
    auto p=a.head;p.past=i;p.tp.epoch=i+1;p.tp.parity=i&1;
    auto pf=Body::Prefetch(p,Shape{});
    tilemega::codegen::mx::StageBarrierV2<Sync>(a.counter,(3ull*i+1)*104,gen,3*i+1,[]{},{});
    unsigned long long const start=maca::GlobalTimerNs();
    Body::Run(pf,p,Shape{},memory,[]{return MxNoPrefetchedV2{};});
    tilemega::codegen::mx::StageBarrierV2<Sync>(a.counter,(3ull*i+2)*104,gen,3*i+2,[]{},{});
    if(blockIdx.x==0&&Sync::IsRole()){a.stamps[i]=maca::GlobalTimerNs()-start;Exchange(a.control,i+1);}
    tilemega::codegen::mx::StageBarrierV2<Sync>(a.counter,(3ull*i+3)*104,gen,3*i+3,[]{},{});
  }
}
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Rank{Args p{};cudaStream_t stream;std::vector<void*> owned;
 void* Make(std::size_t bytes,bool fine=false){void* v=nullptr;if(fine){if(mcExtMallocWithFlags(&v,bytes,mcDeviceMallocFinegrained)!=mcSuccess)throw std::runtime_error("fine allocation");}else Check(cudaMalloc(&v,bytes));owned.push_back(v);Check(cudaMemset(v,0,bytes));return v;}
};
template<class Kernel>void Fit(Kernel fn,int bytes){
 Check(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));cudaFuncAttributes at{};Check(cudaFuncGetAttributes(&at,fn));int resident=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fn,512,bytes));
 if(at.numRegs>255||at.localSizeBytes||at.sharedSizeBytes||resident!=1)throw std::runtime_error("constant kernel resource rejection");
}
template<int TN,int K,bool IsHead>void Pair(Rank (&s)[T],char const* stage){
 std::vector<unsigned long long> values[2][T];
 constexpr int bytes=(IsHead&&MxHead<16,8,1>::SmemBytes(B)>CoreBytes?MxHead<16,8,1>::SmemBytes(B):CoreBytes)+16;
 for(int arm=0;arm<2;++arm){
  for(int r=0;r<T;++r){Check(cudaSetDevice(r));s[r].p.gemm.a_stride=K;
   Check(cudaMemset(s[r].p.counter,0,64));
   if constexpr(IsHead){Fit(Head<true>,bytes);Fit(Head<false>,bytes);if(arm)Head<true><<<104,512,bytes,s[r].stream>>>(s[r].p);else Head<false><<<104,512,bytes,s[r].stream>>>(s[r].p);}
   else{Fit(Gemm<TN,K,true>,bytes);Fit(Gemm<TN,K,false>,bytes);if(arm)Gemm<TN,K,true><<<104,512,bytes,s[r].stream>>>(s[r].p);else Gemm<TN,K,false><<<104,512,bytes,s[r].stream>>>(s[r].p);}
   Check(cudaGetLastError());
  }
  for(int r=0;r<T;++r){Check(cudaSetDevice(r));Check(cudaStreamSynchronize(s[r].stream));values[arm][r].resize(Iterations);Check(cudaMemcpy(values[arm][r].data(),s[r].p.stamps,Iterations*8,cudaMemcpyDeviceToHost));}
 }
 std::vector<double> differences;
 for(int r=0;r<T;++r)for(int i=0;i<Iterations;++i)differences.push_back(double(values[1][r][i])-double(values[0][r][i]));
 std::sort(differences.begin(),differences.end());
 std::printf("{\"pass\":true,\"tp\":%d,\"b\":%d,\"stage\":\"%s\",\"tn\":%d,\"iterations\":%d,\"median_ns\":%.3f,\"p90_ns\":%.3f}\n",T,B,stage,TN,Iterations,(differences[differences.size()/2-1]+differences[differences.size()/2])/2,differences[differences.size()*9/10]);std::fflush(stdout);
}
int main(){try{
 Rank s[T];constexpr int maxK=6144/T>Hidden?6144/T:Hidden;std::vector<Bf16> ones(std::size_t(Vocab)*Hidden,Bf16(1.f));
 for(int r=0;r<T;++r){Check(cudaSetDevice(r));Check(cudaStreamCreate(&s[r].stream));auto& p=s[r].p;
  p.gemm.a=(Bf16*)s[r].Make(B*maxK*2);p.gemm.w=(Bf16*)s[r].Make(ones.size()*2);p.gemm.residual=(Bf16*)s[r].Make(B*Hidden*2);p.gemm.out=(Bf16*)s[r].Make(B*Hidden*2);
  Check(cudaMemcpy((void*)p.gemm.a,ones.data(),B*maxK*2,cudaMemcpyHostToDevice));Check(cudaMemcpy((void*)p.gemm.w,ones.data(),ones.size()*2,cudaMemcpyHostToDevice));
  p.gemm.residual_stride=Hidden;p.gemm.tp.rank=r;p.gemm.tp.world=T;p.gemm.tp.recv[r]=(std::uint64_t*)s[r].Make(T*B*Hidden*4,true);
  p.control.x[r]=(MxTpExchangeV2*)s[r].Make(sizeof(MxTpExchangeV2),true);p.control.rank=r;p.control.world=T;
  p.counter=(unsigned long long*)s[r].Make(64);p.stamps=(unsigned long long*)s[r].Make(Iterations*8);
  p.head.gemm=p.gemm;p.head.gemm.a_stride=Hidden;p.head.gemm.epsilon=1e-6f;
  p.head.partial=(MxHeadPartialV2*)s[r].Make(B*104*sizeof(MxHeadPartialV2));p.head.ticket=(unsigned*)s[r].Make(64);p.head.tokens=(int*)s[r].Make(B*(Iterations+1)*4);p.head.capacity=Iterations+1;
  p.head.tp.x[r]=(MxTpExchangeV2*)s[r].Make(sizeof(MxTpExchangeV2),true);p.head.tp.rank=r;p.head.tp.world=T;p.head.tp.vocab_local=Vocab;p.head.tp.prompt_len=0;
 }
 for(int r=0;r<T;++r){Check(cudaSetDevice(r));for(int q=0;q<T;++q){s[r].p.gemm.tp.recv[q]=s[q].p.gemm.tp.recv[q];s[r].p.head.tp.x[q]=s[q].p.head.tp.x[q];s[r].p.control.x[q]=s[q].p.control.x[q];if(q!=r){auto e=cudaDeviceEnablePeerAccess(q,0);if(e!=cudaErrorPeerAccessAlreadyEnabled)Check(e);cudaGetLastError();}}}
 Pair<16,2048/T,false>(s,"o");Pair<32,2048/T,false>(s,"o");Pair<64,2048/T,false>(s,"o");Pair<128,2048/T,false>(s,"o");
 Pair<16,6144/T,false>(s,"down");Pair<32,6144/T,false>(s,"down");Pair<64,6144/T,false>(s,"down");Pair<128,6144/T,false>(s,"down");Pair<16,Hidden,true>(s,"head");
 for(int r=0;r<T;++r){Check(cudaSetDevice(r));Check(cudaStreamDestroy(s[r].stream));for(void* p:s[r].owned)Check(cudaFree(p));}return 0;
}catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
