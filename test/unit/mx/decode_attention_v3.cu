// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeAttentionV3.h>
#include <cuda_runtime.h>
#ifndef MX6_TEST_DIM
#define MX6_TEST_DIM 128
#endif
#ifndef MX6_TEST_Q
#define MX6_TEST_Q 8
#endif
#ifndef MX6_TEST_ROW_HEADS
#define MX6_TEST_ROW_HEADS 0
#endif
using namespace tilemega;
using namespace tilemega::backend::mx;
using tilemega::backend::LoadGlobal16Gvm;
struct AttentionArgsV3 { MxAttentionOperandsV2 ops; unsigned* calls; int b,hkv; };
template<int B,int Hkv>
struct ShapeV3 : MxStaticAttentionShapeV2<B,Hkv,512> {
  using Sync=tilemega::codegen::mx::StageSyncV3<512>;
};
template<class Body,class Shape>
__global__ void __launch_bounds__(512) AttentionKernelV3(AttentionArgsV3 a) {
  extern __shared__ __align__(16) char smem[];
  auto pf=Body::Prefetch(a.ops,Shape{});
  Body::Run(pf,a.ops,Shape{},smem,[&] {
    if(threadIdx.x%64==0)atomicAdd(a.calls+blockIdx.x*8+threadIdx.x/64,1u);
    return MxNoPrefetchedV2{};
  });
}
template<int B,int Hkv,bool Norm,int Kind,int Split>
int LaunchV3(AttentionArgsV3 const& a,cudaStream_t stream,int* resource,bool query) {
  if constexpr(Split>1&&B*Hkv*Split>104)return 4;
  else {
    using Body=MxAttentionV3<MX6_TEST_DIM,MX6_TEST_Q,Norm,Kind,1088,Norm?MX6_TEST_ROW_HEADS:0,Split>;
    using Shape=ShapeV3<B,Hkv>;
    constexpr int bytes=Body::kSharedBytes>32784?Body::kSharedBytes:32784;
    auto kernel=AttentionKernelV3<Body,Shape>;
    if(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)!=cudaSuccess)return 5;
    cudaFuncAttributes p{};if(cudaFuncGetAttributes(&p,kernel)!=cudaSuccess)return 6;
    int resident=0;
    if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,512,bytes)!=cudaSuccess)return 7;
    if(resource){resource[0]=p.numRegs;resource[1]=p.localSizeBytes;resource[2]=p.sharedSizeBytes;resource[3]=resident;}
    if(p.numRegs>255||p.localSizeBytes||p.sharedSizeBytes||resident!=1)return 8;
    if(query)return 0;
    kernel<<<104,512,bytes,stream>>>(a);
    return int(cudaGetLastError());
  }
}
template<int B,int Hkv,bool Norm,int Kind>
int SplitV3(AttentionArgsV3 const& a,int split,cudaStream_t s,int* r,bool q) {
  if(split==1)return LaunchV3<B,Hkv,Norm,Kind,1>(a,s,r,q);
  if(split==2)return LaunchV3<B,Hkv,Norm,Kind,2>(a,s,r,q);
  if(split==4)return LaunchV3<B,Hkv,Norm,Kind,4>(a,s,r,q);
  return 4;
}
template<int B,int Hkv,bool Norm>
int KindV3(AttentionArgsV3 const& a,int kind,int split,cudaStream_t s,int* r,bool q) {
  if(kind==0)return SplitV3<B,Hkv,Norm,0>(a,split,s,r,q);
  if(kind==1)return SplitV3<B,Hkv,Norm,1>(a,split,s,r,q);
  if(kind==2)return SplitV3<B,Hkv,Norm,2>(a,split,s,r,q);
  return 4;
}
template<int B,int Hkv>
int NormV3(AttentionArgsV3 const& a,int norm,int kind,int split,cudaStream_t s,int* r,bool q) {
  if constexpr(MX6_TEST_DIM==128)if(norm)return KindV3<B,Hkv,true>(a,kind,split,s,r,q);
  if(norm)return 4;
  return KindV3<B,Hkv,false>(a,kind,split,s,r,q);
}
extern "C" int mx6_attention(AttentionArgsV3 const* a,int norm,int kind,int split,void* stream,int* resource,int query) {
  if((norm!=0&&norm!=1)||(query!=0&&query!=1)||a->ops.capacity!=1088)return 4;
  auto s=static_cast<cudaStream_t>(stream);
#if MX6_TEST_ROW_HEADS
  if(a->hkv!=MX6_TEST_ROW_HEADS)return 4;
  if(a->b==1)return NormV3<1,MX6_TEST_ROW_HEADS>(*a,norm,kind,split,s,resource,bool(query));
  if(a->b==16)return NormV3<16,MX6_TEST_ROW_HEADS>(*a,norm,kind,split,s,resource,bool(query));
#else
  if(a->b==1&&a->hkv==4)return NormV3<1,4>(*a,norm,kind,split,s,resource,bool(query));
  if(a->b==1&&a->hkv==16)return NormV3<1,16>(*a,norm,kind,split,s,resource,bool(query));
  if(a->b==16&&a->hkv==4)return NormV3<16,4>(*a,norm,kind,split,s,resource,bool(query));
  if(a->b==16&&a->hkv==16)return NormV3<16,16>(*a,norm,kind,split,s,resource,bool(query));
#endif
  return 4;
}

template<bool Norm>
__global__ void __launch_bounds__(512) OriginalTransformV2(MxAttentionOperandsV2 p,Bf16* q,Bf16* k,int heads) {
  extern __shared__ __align__(16) char storage[];
  int const wave=threadIdx.x/64,lane=threadIdx.x%64;
  int const row=blockIdx.x*8+wave,head=row/(MX6_TEST_Q+1),component=row%(MX6_TEST_Q+1);
  if(head>=heads)return;
  float* exchange=reinterpret_cast<float*>(storage)+wave*64;
  float inverse=1.f;
#if MX6_TEST_ROW_HEADS
  if constexpr(Norm)inverse=MxProjectionInverseV2<MX6_TEST_DIM,MX6_TEST_ROW_HEADS,MX6_TEST_Q>(
      p.qkv,head/MX6_TEST_ROW_HEADS,component==MX6_TEST_Q?MX6_TEST_Q:0,p.epsilon,exchange,lane);
#endif
  constexpr int chunks=MX6_TEST_DIM/32;
  uint4 x[chunks],weight[chunks],cosine[chunks],sine[chunks];
#pragma unroll
  for(int dc=0;dc<chunks;++dc) {
    int const at=dc*32+(lane/16)*8;
    x[dc]=LoadGlobal16Gvm(p.qkv+(head*(MX6_TEST_Q+2)+component)*MX6_TEST_DIM+at);
    cosine[dc]=LoadGlobal16Gvm(p.cosine+p.past*MX6_TEST_DIM+at);
    sine[dc]=LoadGlobal16Gvm(p.sine+p.past*MX6_TEST_DIM+at);
    weight[dc]=uint4{};
    if constexpr(Norm) {
      int const offset=MX6_TEST_ROW_HEADS?head%MX6_TEST_ROW_HEADS*MX6_TEST_DIM:0;
      weight[dc]=LoadGlobal16Gvm((component==MX6_TEST_Q?p.k_norm:p.q_norm)+offset+at);
    }
  }
  __builtin_mxc_arrive_gvmcnt(0);
  MxRowTransformV2<MX6_TEST_DIM,Norm,(MX6_TEST_ROW_HEADS>0)>(x,weight,cosine,sine,p.epsilon,exchange,lane,inverse);
  if(lane%16==0) {
    Bf16* out=component==MX6_TEST_Q?k+head*MX6_TEST_DIM:q+(head*MX6_TEST_Q+component)*MX6_TEST_DIM;
#pragma unroll
    for(int dc=0;dc<chunks;++dc)MxStore16V2(out+dc*32+(lane/16)*8,x[dc]);
  }
}
extern "C" int mx6_attention_reference_transform(MxAttentionOperandsV2 const* p,void* q,void* k,int heads,int norm,void* stream) {
  auto s=static_cast<cudaStream_t>(stream);unsigned const blocks=(heads*(MX6_TEST_Q+1)+7)/8;
  if constexpr(MX6_TEST_DIM==128) {
    if(norm){OriginalTransformV2<true><<<blocks,512,32784,s>>>(*p,(Bf16*)q,(Bf16*)k,heads);return int(cudaGetLastError());}
  }
  if(norm)return 4;
  OriginalTransformV2<false><<<blocks,512,32784,s>>>(*p,(Bf16*)q,(Bf16*)k,heads);
  return int(cudaGetLastError());
}
