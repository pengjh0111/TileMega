// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeHeadV2.h>
#include <mcr/mc_runtime_api.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <vector>
#ifndef MX7_TP
#define MX7_TP 2
#define MX7_BATCH 1
#endif
using namespace tilemega;
using namespace tilemega::backend::mx;
constexpr int T=MX7_TP,B=MX7_BATCH,N=32768,K=256,Capacity=8;
using Shape=MxStaticShapeV2<N,K,B,512>;
using Body=MxHead<16,8,1,1,T,N>;
using Ops=MxTpHeadOperandsV2<MxHeadOperandsV2>;
void Check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
void* Make(std::vector<void*>& owned,std::size_t bytes,bool fine=false){
  void* p=nullptr;
  if(fine){if(mcExtMallocWithFlags(&p,bytes,mcDeviceMallocFinegrained)!=mcSuccess)throw std::runtime_error("fine allocation");}
  else Check(cudaMalloc(&p,bytes));
  owned.push_back(p);Check(cudaMemset(p,0,bytes));return p;
}
__global__ void __launch_bounds__(512) Head(Ops p){
  unsigned const grid=gridDim.x;__builtin_assume(grid==104);
  extern __shared__ __align__(16) char s[];
  auto pf=Body::Prefetch(p,Shape{});Body::Run(pf,p,Shape{},s,[]{return MxNoPrefetchedV2{};});
}
int main(){try {
  Ops p[T]{};std::vector<void*> owned[T];cudaStream_t stream[T];
  std::vector<Bf16> a(B*K,Bf16(1.f)),w(N*K,Bf16(1.f));
  constexpr int bytes=Body::SmemBytes(B)>32784?Body::SmemBytes(B):32784;
  for(int r=0;r<T;++r){
    Check(cudaSetDevice(r));Check(cudaStreamCreate(&stream[r]));auto& v=owned[r];
    p[r].gemm.a=(Bf16*)Make(v,B*K*2);p[r].gemm.w=(Bf16*)Make(v,N*K*2);
    p[r].gemm.a_stride=K;p[r].gemm.epsilon=1e-6f;
    p[r].partial=(MxHeadPartialV2*)Make(v,B*104*sizeof(MxHeadPartialV2));
    p[r].ticket=(unsigned*)Make(v,64);p[r].tokens=(int*)Make(v,B*Capacity*4);
    p[r].capacity=Capacity;p[r].tp.x[r]=(MxTpExchangeV2*)Make(v,sizeof(MxTpExchangeV2),true);
    p[r].tp.rank=r;p[r].tp.world=T;p[r].tp.vocab_local=N;p[r].tp.prompt_len=1;
    Check(cudaMemcpy((void*)p[r].gemm.a,a.data(),B*K*2,cudaMemcpyHostToDevice));
    Check(cudaFuncSetAttribute(Head,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    cudaFuncAttributes at{};Check(cudaFuncGetAttributes(&at,Head));int resident=0;
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Head,512,bytes));
    if(at.numRegs>255||at.localSizeBytes||at.sharedSizeBytes||resident!=1)throw std::runtime_error("resource failure");
  }
  for(int r=0;r<T;++r){Check(cudaSetDevice(r));for(int q=0;q<T;++q){p[r].tp.x[q]=p[q].tp.x[q];if(q!=r){auto e=cudaDeviceEnablePeerAccess(q,0);if(e!=cudaErrorPeerAccessAlreadyEnabled)Check(e);cudaGetLastError();}}}
  for(int test=0;test<4;++test){
    for(int r=0;r<T;++r){
      Check(cudaSetDevice(r));std::fill(w.begin(),w.end(),Bf16(1.f));
      auto set=[&](int row){for(int k=0;k<K;++k){int const packed=((row/16)*(K/32)+k/32)*512+(k%32/8)*128+(row%16)*8+k%8;w[packed]=Bf16(2.f);}};
      if(test==0&&r==0){set(5);set(9);}if(test==1&&r==0)set(5);if(test==1&&r==1)set(3);if(test==3&&r==1)set(7);
      Check(cudaMemcpy((void*)p[r].gemm.w,w.data(),N*K*2,cudaMemcpyHostToDevice));
      p[r].past=test;p[r].tp.epoch=test+1;p[r].tp.parity=test&1;
    }
    for(int r=0;r<T;++r){Check(cudaSetDevice(r));Head<<<104,512,bytes,stream[r]>>>(p[r]);Check(cudaGetLastError());}
    int const expected=test==2?0:test==3?N+7:5;
    for(int r=0;r<T;++r){Check(cudaSetDevice(r));Check(cudaStreamSynchronize(stream[r]));std::vector<int> tokens(B*Capacity);Check(cudaMemcpy(tokens.data(),p[r].tokens,B*Capacity*4,cudaMemcpyDeviceToHost));for(int b=0;b<B;++b)if(tokens[b*Capacity+test+1]!=expected){std::fprintf(stderr,"test=%d rank=%d got=%d expected=%d\n",test,r,tokens[b*Capacity+test+1],expected);throw std::runtime_error("head argmax mismatch");}}
  }
  for(int r=0;r<T;++r){Check(cudaSetDevice(r));Check(cudaStreamDestroy(stream[r]));for(void* v:owned[r])Check(cudaFree(v));}
  std::printf("{\"pass\":true,\"tp\":%d,\"batch\":%d,\"cases\":4}\n",T,B);return 0;
}catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
