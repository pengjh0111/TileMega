// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/AttentionPageLayout.h>
#include <tilemega/Codegen/tasks/IndependentAttentionTaskBody.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

using namespace tilemega::codegen;
using Element=cutlass::bfloat16_t;
using Arch=std::conditional_t<std::is_void_v<tilemega::arch::CurrentArch>,
    tilemega::arch::Sm80,tilemega::arch::CurrentArch>;
constexpr int Capacity=1024;
static void CheckCuda(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(2);}
}
template<class T> T* Allocate(int count) {
  T* p=nullptr;CheckCuda(cudaMallocManaged(&p,sizeof(T)*count));return p;
}
template<int D,int Q,int Bytes,AttentionPagePolicy Policy,bool Independent>
struct Case {
#ifdef TILEMEGA_LAYOUT_LEGACY
  using Body=PagedAttentionTaskBody<Arch,D,Q,false,Bytes,5>;
  using Local=IndependentAttentionTaskBody<Arch,D,Q,false>;
#else
  using Body=PagedAttentionTaskBody<Arch,D,Q,false,Bytes,5,false,Q,Policy>;
  using Local=IndependentAttentionTaskBody<Arch,D,Q,false,Bytes,Policy>;
#endif
  using Ring=typename Body::Ring;
  static constexpr int Header=(1024+sizeof(typename Body::SharedStorage)+1023)/1024*1024;
  static constexpr int Shared=Independent?sizeof(typename Local::SharedStorage):Header+5*Bytes;
};
template<int D,int Q,int Bytes,AttentionPagePolicy Policy,bool Independent>
__global__ void LayoutKernel(ServingAttentionOperands p,int chunk,int steps) {
  using C=Case<D,Q,Bytes,Policy,Independent>;
  extern __shared__ __align__(1024) unsigned char memory[];
  if constexpr(Independent) {
    auto& s=*reinterpret_cast<typename C::Local::SharedStorage*>(memory);
    for(int step=0;step<steps;++step,++p.past) {
      C::Local::Run(p,s,0,0,0,chunk);
      __syncthreads();
    }
  }else {
    typename C::Ring ring{reinterpret_cast<typename C::Ring::Slot*>(memory),
        reinterpret_cast<char*>(memory)+C::Header};
    auto& s=*reinterpret_cast<typename C::Body::SharedStorage*>(memory+1024);
    ring.Initialize();std::uint64_t sequence=0;
    for(int step=0;step<steps;++step,++p.past) {
      if(executor::IsCompute())C::Body::Run(p,0,0,chunk,ring,sequence,s);
      else C::Body::Load(p,0,0,chunk,ring,sequence);
      // Test only: order appended KV across steps while preserving the ring
      // sequence and storage, so stale reused slots remain observable.
      __syncthreads();
    }
  }
}
struct Totals {int cases=0,failures=0;float context_error=0,lse_error=0;};
template<int D,int Q,int Bytes,AttentionPagePolicy Policy,bool Independent>
Totals Test() {
  using C=Case<D,Q,Bytes,Policy,Independent>;
  CheckCuda(cudaFuncSetAttribute(LayoutKernel<D,Q,Bytes,Policy,Independent>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,C::Shared));
  auto* qkv=Allocate<Element>((Q+2)*D);
  auto* k=Allocate<Element>(Capacity*D);auto* v=Allocate<Element>(Capacity*D);
  auto* cosine=Allocate<Element>(Capacity*D);auto* sine=Allocate<Element>(Capacity*D);
  auto* context=Allocate<Element>(Q*D);
  auto* partial=Allocate<float>(Capacity*Q*D);auto* lse=Allocate<float>(Capacity*Q);
  for(int i=0;i<Capacity*D;++i){cosine[i]=Element(1);sine[i]=Element(0);}
  for(int i=0;i<(Q+2)*D;++i)qkv[i]=Element(.02f*(i%13-6));
  // The appended value is deliberately unlike the historical position code.
  for(int d=0;d<D;++d)qkv[(Q+1)*D+d]=Element(.75f+.01f*(d%7));
  struct Input {int past,extent,chunk,steps;};std::vector<Input> inputs;
  for(int count:{1,15,16,17,32,33,48,63,64,65,128,129,256})
    for(int steps:{1,3})inputs.push_back({count-1,512,0,steps});
  // Current token at the start/middle/end of a nominal Ec chunk. Causality
  // always makes it the last *valid* row, even in the first two cases.
  for(int ec:{32,64})for(int offset:{0,ec/2,ec-1})inputs.push_back({ec+offset,ec,1,1});
  Totals out;
  for(auto in:inputs) {
    for(int pos=0;pos<Capacity;++pos)for(int d=0;d<D;++d) {
      k[pos*D+d]=Element(.018f*(pos%17-8)+.01f*(d%11-5));
      v[pos*D+d]=Element(float(pos)/256+.03f*(d%13));
    }
    for(int i=0;i<Capacity*Q;++i)lse[i]=NAN;
    for(int i=0;i<Capacity*Q*D;++i)partial[i]=NAN;
    ServingAttentionOperands p{qkv,k,v,cosine,sine,nullptr,nullptr,
      context,partial,lse,1,1,Capacity,in.past,in.extent,1e-6f};
    LayoutKernel<D,Q,Bytes,Policy,Independent><<<1,Independent?128:160,C::Shared>>>(p,in.chunk,in.steps);
    CheckCuda(cudaDeviceSynchronize());
    int end=std::min((in.chunk+1)*in.extent,in.past+in.steps),begin=in.chunk*in.extent;
    // Read the appended cache back, but independently verify its expected value
    // before using it in the fp32 logical reference.
    bool pass=true;
    for(int pos=in.past;pos<in.past+in.steps && pos<end;++pos)for(int d=0;d<D;++d)
      pass=pass && k[pos*D+d]==qkv[Q*D+d] && v[pos*D+d]==qkv[(Q+1)*D+d];
    float error=0,log_error=0;
    for(int row=0;row<Q;++row) {
      std::vector<float> score(end-begin);float peak=-INFINITY;
      for(int pos=begin;pos<end;++pos) {
        float dot=0;for(int d=0;d<D;++d)dot+=float(qkv[row*D+d])*float(k[pos*D+d]);
        score[pos-begin]=dot/std::sqrt(float(D));peak=std::max(peak,score[pos-begin]);
      }
      float sum=0;for(auto& x:score){x=std::exp(x-peak);sum+=x;}
      float expected_lse=(peak+std::log(sum))/std::log(2.f);
      float got_lse=lse[in.chunk*Q+row];
      log_error=std::max(log_error,std::abs(got_lse-expected_lse));
      pass=pass && std::isfinite(got_lse) && std::abs(got_lse-expected_lse)<.006f;
      for(int d=0;d<D;++d) {
        float expected=0;for(int pos=begin;pos<end;++pos)expected+=score[pos-begin]*float(v[pos*D+d]);
        expected/=sum;float got=partial[(in.chunk*Q+row)*D+d];
        error=std::max(error,std::abs(got-expected));
        pass=pass && std::isfinite(got) && std::abs(got-expected)<.006f*(1+std::abs(expected));
      }
    }
    ++out.cases;out.failures+=!pass;
    out.context_error=std::max(out.context_error,error);out.lse_error=std::max(out.lse_error,log_error);
    if(!pass)std::printf("FAIL D=%d Q=%d page=%d policy=%d independent=%d count=%d chunk=%d extent=%d steps=%d context_error=%g lse_error=%g\n",D,Q,Bytes,int(Policy),Independent,end-begin,in.chunk,in.extent,in.steps,error,log_error);
  }
  std::printf("RESULT D=%d Q=%d page=%d policy=%d independent=%d cases=%d failures=%d context_error=%g lse_error=%g\n",D,Q,Bytes,int(Policy),Independent,out.cases,out.failures,out.context_error,out.lse_error);
  void* allocations[]={qkv,k,v,cosine,sine,context,partial,lse};
  for(void* p:allocations)CheckCuda(cudaFree(p));
  return out;
}
template<int D,int Q> int Shapes() {
  int failed=0;
#define RUN(B,P,I) failed+=Test<D,Q,B,AttentionPagePolicy::P,I>().failures
  RUN(8192,Packed,false);RUN(16384,Packed,false);
  RUN(8192,WarpPrivate,true);
#ifndef TILEMEGA_LAYOUT_LEGACY
  RUN(8192,WarpPrivate,false);RUN(16384,WarpPrivate,false);
  RUN(8192,Packed,true);
#endif
#undef RUN
  return failed;
}
int main() {
  int failed=Shapes<64,2>()+Shapes<64,4>()+Shapes<128,2>()+Shapes<128,4>();
  std::printf("attention_layout %s failures=%d\n",failed?"FAIL":"PASS",failed);
  return failed?1:0;
}
