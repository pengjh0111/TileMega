// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeGemmV2.h>
#include <mcr/mc_runtime_api.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef MX7_TP
#define MX7_TP 2
#define MX7_BATCH 1
#endif
using namespace tilemega;
using namespace tilemega::backend::mx;
constexpr int T=MX7_TP, B=MX7_BATCH, N=2048, K=2048/T;
using Shape=MxStaticShapeV2<N,K,B,512>;
using Ops=MxTpOperandsV2<MxGemmOperandsV2>;
void Check(cudaError_t status) {
  if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
void* Alloc(std::size_t bytes,bool fine=false) {
  void* p=nullptr;
  if(fine) {
    if(mcExtMallocWithFlags(&p,bytes,mcDeviceMallocFinegrained)!=mcSuccess)
      throw std::runtime_error("fine allocation failed");
  } else Check(cudaMalloc(&p,bytes));
  Check(cudaMemset(p,0,bytes));return p;
}
template<int TN,int WN,MxPart Part,bool AR>
__global__ void __launch_bounds__(512) Stage(Ops p) {
  unsigned const grid=gridDim.x;__builtin_assume(grid==104);
  extern __shared__ __align__(16) char smem[];
  using Body=MxGemm<TN,WN,1,AR?MxOpV2::kResidualAllReduce:MxOpV2::kStore,false,Part>;
  auto pf=Body::Prefetch(p,Shape{});
  Body::Run(pf,p,Shape{},smem,[]{return MxNoPrefetchedV2{};});
}
struct Rank {
  Ops p{};cudaStream_t stream{};std::vector<void*> owned;
  void* Make(std::size_t n,bool fine=false){void* v=Alloc(n,fine);owned.push_back(v);return v;}
};
template<int TN,int WN,MxPart Part>
int One() {
  if constexpr(!MxGemmLegalV2(N,K,B,TN,WN,1,MxOpV2::kResidual,Part))return 0;
  else {
    Rank rank[T];std::vector<Bf16> a(B*K),w(N*K),old(B*N),partial[T],out(B*N);
    for(int i=0;i<B*N;++i)old[i]=Bf16(float((i*13)%97-48)/64);
    constexpr int bytes=std::max(32784,MxGemmSmemV2(B,TN,WN).bytes);
    std::size_t const partial_bytes=std::size_t(N/TN)*MxMaxContributors(N/TN,104)*B*(TN+1)*4;
    for(int r=0;r<T;++r) {
      Check(cudaSetDevice(r));auto& s=rank[r];Check(cudaStreamCreate(&s.stream));
      for(int i=0;i<B*K;++i)a[i]=Bf16(float((i*17+r*7)%31-15)/32);
      for(int i=0;i<N*K;++i)w[i]=Bf16(float((i*19+r*11)%37-18)/64);
      auto* da=(Bf16*)s.Make(a.size()*2);auto* dw=(Bf16*)s.Make(w.size()*2);
      auto* residual=(Bf16*)s.Make(old.size()*2);auto* output=(Bf16*)s.Make(old.size()*2);
      Check(cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice));
      Check(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));
      Check(cudaMemcpy(residual,old.data(),old.size()*2,cudaMemcpyHostToDevice));
      s.p.a=da;s.p.w=dw;s.p.residual=residual;s.p.out=output;
      s.p.partial=(float*)s.Make(std::max(std::size_t(16),partial_bytes));
      s.p.tickets=(unsigned*)s.Make(N/TN*4);s.p.a_stride=K;s.p.residual_stride=N;
      s.p.tp.rank=r;s.p.tp.world=T;s.p.tp.epoch=1;
      s.p.tp.recv[r]=(std::uint64_t*)s.Make(std::size_t(T)*B*N*4,true);
      Check(cudaFuncSetAttribute(Stage<TN,WN,Part,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
      Stage<TN,WN,Part,false><<<104,512,bytes,s.stream>>>(s.p);
      Check(cudaGetLastError());Check(cudaStreamSynchronize(s.stream));
      partial[r].resize(B*N);Check(cudaMemcpy(partial[r].data(),output,B*N*2,cudaMemcpyDeviceToHost));
    }
    for(int r=0;r<T;++r) {
      Check(cudaSetDevice(r));
      for(int q=0;q<T;++q) {
        rank[r].p.tp.recv[q]=rank[q].p.tp.recv[q];
        if(q!=r){auto e=cudaDeviceEnablePeerAccess(q,0);if(e!=cudaErrorPeerAccessAlreadyEnabled)Check(e);cudaGetLastError();}
      }
      Check(cudaFuncSetAttribute(Stage<TN,WN,Part,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
      cudaFuncAttributes attrs{};Check(cudaFuncGetAttributes(&attrs,Stage<TN,WN,Part,true>));int resident=0;
      Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Stage<TN,WN,Part,true>,512,bytes));
      std::printf("{\"resource\":true,\"tn\":%d,\"wn\":%d,\"part\":%d,\"rank\":%d,\"registers\":%d,\"stack\":%zu,\"static\":%zu,\"resident\":%d}\n",TN,WN,int(Part),r,attrs.numRegs,attrs.localSizeBytes,attrs.sharedSizeBytes,resident);
      if(attrs.numRegs>255||attrs.localSizeBytes||attrs.sharedSizeBytes||resident!=1)throw std::runtime_error("resource failure");
    }
    for(int r=0;r<T;++r) {Check(cudaSetDevice(r));Stage<TN,WN,Part,true><<<104,512,bytes,rank[r].stream>>>(rank[r].p);Check(cudaGetLastError());}
    unsigned errors=0;
    for(int r=0;r<T;++r) {
      Check(cudaSetDevice(r));Check(cudaStreamSynchronize(rank[r].stream));
      Check(cudaMemcpy(out.data(),rank[r].p.out,B*N*2,cudaMemcpyDeviceToHost));
      for(int i=0;i<B*N;++i) {
        float sum=0;for(int q=0;q<T;++q)sum+=float(partial[q][i]);
        Bf16 expected(float(Bf16(sum))+float(old[i]));
        if(out[i].raw()!=expected.raw()){if(errors<4)std::fprintf(stderr,"mismatch rank=%d i=%d got=%x ref=%x\n",r,i,out[i].raw(),expected.raw());++errors;}
      }
      Check(cudaStreamDestroy(rank[r].stream));for(void* p:rank[r].owned)Check(cudaFree(p));
    }
    std::printf("{\"case\":true,\"tp\":%d,\"batch\":%d,\"tn\":%d,\"wn\":%d,\"part\":%d,\"errors\":%u,\"pass\":%s}\n",T,B,TN,WN,int(Part),errors,errors?"false":"true");
    if(errors)throw std::runtime_error("AR differs from ordered BF16 reference");return 1;
  }
}
template<int TN,int WN>int Parts(){return One<TN,WN,MxPart::kTileRange>()+One<TN,WN,MxPart::kStreamK>();}
template<int TN>int Width(){return Parts<TN,1>()+Parts<TN,2>()+Parts<TN,4>()+Parts<TN,8>();}
int main(){try {int cases=Width<16>()+Width<32>()+Width<64>()+Width<128>();std::printf("{\"pass\":true,\"cases\":%d}\n",cases);return 0;}catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
