// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90Gmma.h>
#include <tilemega/Codegen/tasks/EncoderAttentionTaskBody.h>
#include <tilemega/Codegen/executor/TensorMap.h>
#include <tilemega/Codegen/tasks/Sm90PrefillAttentionTaskBody.h>

namespace tilemega::codegen {
struct Sm90EncoderAttentionOperands {
  EncoderAttentionOperands base;
  // Three tensor maps per (image, head), with the source row pitch including
  // the interleaved Q/K/V heads. A missing table selects the cp.async arm.
  executor::TensorMap const* tensor_maps=nullptr;
  unsigned map_heads=0;
};
template<class Arch>
struct Sm90EncoderAttentionTaskBody {
  using E=cutlass::bfloat16_t;
  using Gmma=backend::Sm90Gmma<Arch>;
  using Async=executor::Async<Arch>;
  using QKMma=decltype(cute::make_tiled_mma(typename Gmma::template SS<128>{}));
  using PVMma=decltype(cute::make_tiled_mma(typename Gmma::template RS<64,backend::gmma::Major::MN>{}));
  using QLayout=backend::Sm90KLayout<64,64>;
  using KLayout=backend::Sm90KLayout<128,64>;
  using VLayout=decltype(cute::tile_to_shape(backend::gmma::Layout_MN_SW128_Atom<E>{},cute::Shape<cute::_64,cute::_128>{}));
  struct SharedStorage {
    alignas(1024) E query[64*64];
    alignas(1024) E key[128*64];
    alignas(1024) E value[128*64];
    alignas(16) float scores[64*128];
    float sum[64];
    std::uint64_t barrier;
  };
  static constexpr int kSharedBytes=sizeof(SharedStorage),kThreads=128;
  static_assert(kSharedBytes==Sm90EncoderAttentionSharedBytes());
  __host__ __device__ static unsigned Count(EncoderAttentionOperands const& p) {return p.batch*p.heads*2;}
#if TILEMEGA_FORWARD_WGMMA_OUTLINE > 1
  __device__ static __noinline__
#else
  __device__ static __forceinline__
#endif
  void Run(Sm90EncoderAttentionOperands const& p,unsigned task,SharedStorage& s) {
    using namespace cute;
    using executor::ComputeThread;using executor::ComputeSync;
    auto const& in=p.base;
    if(!in.qkv || !in.context || !in.batch || !in.heads || in.key_padding) {asm volatile("trap;");return;}
    unsigned first=(task%2)*64,head=(task/2)%in.heads,image=task/(2*in.heads);
    if(image>=in.batch)return;
    ComputeSync();
    if(p.tensor_maps) {
      if(ComputeThread()==0)Async::Init(&s.barrier,1);
      Async::InitFence();ComputeSync();Async::ProxyAsyncGlobalFence();
      if(ComputeThread()==0) {
        auto* maps=p.tensor_maps+(std::size_t(image)*(p.map_heads?p.map_heads:in.heads)+head)*3;
        Async::ExpectTx(&s.barrier,(64+128+128)*64*2);
        Async::Tensor2D(s.query,maps,0,first,&s.barrier);
        Async::Tensor2D(s.key,maps+1,0,0,&s.barrier);
        Async::Tensor2D(s.value,maps+2,0,0,&s.barrier);
      }
      Async::Wait(&s.barrier,0);ComputeSync();
    }else {
      auto ql=as_position_independent_swizzle_layout(QLayout{});
      auto kl=as_position_independent_swizzle_layout(KLayout{});
      for(unsigned v=ComputeThread();v<128*8;v+=128) {
        unsigned row=v/8,d=v%8*8;
        auto* source=in.qkv+((std::size_t(image)*128+row)*in.heads+head)*192+d;
        if(row<64) {
          auto* qs=in.qkv+((std::size_t(image)*128+first+row)*in.heads+head)*192+d;
          Async::Copy16Bytes(s.query+ql(row,d),qs,16);
        }
        Async::Copy16Bytes(s.key+kl(row,d),source+64,16);
        Async::Copy16Bytes(s.value+kl(row,d),source+128,16);
      }
      cp_async_fence();cp_async_wait<0>();Async::FenceProxyAsyncShared();ComputeSync();
    }
    auto qkt=QKMma{}.get_slice(ComputeThread());
    auto pvt=PVMma{}.get_slice(ComputeThread());
    auto qc=qkt.partition_C(make_identity_tensor(Shape<_64,_128>{}));
    auto oc=pvt.partition_C(make_identity_tensor(Shape<_64,_64>{}));
    auto q=make_tensor(make_smem_ptr(s.query),QLayout{});
    auto k=make_tensor(make_smem_ptr(s.key),KLayout{});
    auto v=make_tensor(make_smem_ptr(s.value),VLayout{});
    auto rq=qkt.partition_fragment_A(q),rk=qkt.partition_fragment_B(k);
    auto rv=pvt.partition_fragment_B(v);
    auto score=partition_fragment_C(QKMma{},Shape<_64,_128>{});clear(score);Gmma::Fence(score);
    Gmma::Arrive();
    CUTE_UNROLL
    for(int d=0;d<size<2>(rq);++d)gemm(QKMma{},rq(_,_,d),rk(_,_,d),score);
    Gmma::Commit();Gmma::template Wait<0>();Gmma::Fence(score);
    CUTE_UNROLL
    for(int i=0;i<size(score);++i)s.scores[get<0>(qc(i))*128+get<1>(qc(i))]=score(i)*(1.4426950408889634f/8);
    ComputeSync();
    int lane=ComputeThread()%32,warp=ComputeThread()/32;
    for(int row=warp;row<64;row+=4) {
      float peak=-INFINITY;for(int j=0;j<4;++j)peak=fmaxf(peak,s.scores[row*128+lane+32*j]);
      for(int shift=16;shift;shift/=2)peak=fmaxf(peak,__shfl_xor_sync(0xffffffffu,peak,shift));
      float sum=0;for(int j=0;j<4;++j){float prob=exp2f(s.scores[row*128+lane+32*j]-peak);s.scores[row*128+lane+32*j]=prob;sum+=prob;}
      for(int shift=16;shift;shift/=2)sum+=__shfl_xor_sync(0xffffffffu,sum,shift);
      if(lane==0)s.sum[row]=sum;
    }
    ComputeSync();
    using LayoutHelper=Sm90PrefillAttentionTaskBody<Arch,64,1,64,64,64,false>;
    auto probabilities=make_fragment_like<E>(LayoutHelper::RegisterOperandLayout(score.layout(),shape<1>(typename PVMma::LayoutA_TV{})));
    auto pc=make_tensor(probabilities.data(),score.layout());
    CUTE_UNROLL
    for(int i=0;i<size(score);++i)pc(i)=E(s.scores[get<0>(qc(i))*128+get<1>(qc(i))]);
    auto output=partition_fragment_C(PVMma{},Shape<_64,_64>{});clear(output);Gmma::Fence(output);
    Gmma::Arrive();
    CUTE_UNROLL
    for(int r=0;r<size<2>(probabilities);++r)gemm(PVMma{},probabilities(_,_,r),rv(_,_,r),output);
    Gmma::Commit();Gmma::template Wait<0>();Gmma::Fence(output);ComputeSync();
    CUTE_UNROLL
    for(int i=0;i<size(output);++i)s.scores[get<0>(oc(i))*64+get<1>(oc(i))]=output(i)/s.sum[get<0>(oc(i))];
    ComputeSync();
    for(int v=ComputeThread();v<64*8;v+=128) {
      int row=v/8,d=v%8*8;alignas(16) E stored[8];
      CUTE_UNROLL
      for(int j=0;j<8;++j)stored[j]=E(s.scores[row*64+d+j]);
      auto* dst=in.context+((std::size_t(image)*128+first+row)*in.heads+head)*64+d;
      *reinterpret_cast<uint4*>(dst)=*reinterpret_cast<uint4 const*>(stored);
    }
    ComputeSync();
  }
};
} // namespace tilemega::codegen
