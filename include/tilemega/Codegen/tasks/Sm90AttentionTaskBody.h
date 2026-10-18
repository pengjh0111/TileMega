// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90Gmma.h>
#include <tilemega/Codegen/tasks/FusedAttentionTaskBody.h>
#include <tilemega/Codegen/executor/Async.cuh>

namespace tilemega::codegen {
template<class Arch,int D,int Q,bool Norm>
struct Sm90AttentionTaskBody {
  using Element=cutlass::bfloat16_t;
  using Gmma=backend::Sm90Gmma<Arch>;
  using Copy=executor::Async<Arch>;
  using Base=FusedAttentionTaskBody<Arch,D,Q,1,16,32,Norm>;
  using QueryLayout=backend::Sm90KLayout<8,D>;
  using ProbabilityLayout=backend::Sm90KLayout<8,64>;
  using MmaQK=decltype(cute::make_tiled_mma(typename Gmma::template SS<8>{}));
  using MmaPV=decltype(cute::make_tiled_mma(
      typename Gmma::template SS<8,backend::gmma::Major::MN>{}));
  struct SharedStorage {
    alignas(1024) Element query[8*D];
    alignas(1024) float scores[64*8];
    alignas(1024) Element probabilities[8*64];
    float maximum[8],sum[8],alpha[8];
  };
  template<class Owner,class PageSource,bool LocalPartial=false>
  __device__ static __forceinline__ void Run(ServingAttentionOperands const& p,
      int b,int g,int c,PageSource const& pages,std::uint64_t& sequence,SharedStorage& s,float* local_partial=nullptr) {
    using namespace cute;
    static_assert(arch::Caps<Arch>::kWgmma && Q<=8 && (D==64 || D==128));
    static_assert(sizeof(typename Gmma::Availability)>0);
    static_assert(Owner::kPageRows==64,"WGMMA attention requires one 64-key K|V page");
    int thread=executor::ComputeThread(),warp=thread/32,lane=thread%32;
    int begin=Owner::Begin(p,c),end=Owner::End(p,c);if(begin>=end)return;
    auto layout=Owner::Layout(end-begin,begin);
    auto query_offsets=as_position_independent_swizzle_layout(QueryLayout{});
    for(int q=warp;q<8;q+=4) {
      if(q<Q) {
#if TILEMEGA_QKV_EPILOGUE
        if(p.qkv_preprocessed) {
          if(lane*8<D)*reinterpret_cast<uint4*>(s.query+query_offsets(q,lane*8))=
              backend::LoadGlobal16(p.qkv+(b*p.heads_kv+g)*(Q+2)*D+q*D+lane*8);
        }else
#endif
        Base::RotateRow(p.qkv+(b*p.heads_kv+g)*(Q+2)*D+q*D,p.q_norm,
            p.cosine+p.past*D,p.sine+p.past*D,p.epsilon,[&](int d,uint4 value) {
              *reinterpret_cast<uint4*>(s.query+query_offsets(q,d))=value;
            });
      }
      else if(lane*8<D)*reinterpret_cast<uint4*>(s.query+query_offsets(q,lane*8))=make_uint4(0,0,0,0);
    }
    if(thread<8){s.maximum[thread]=-INFINITY;s.sum[thread]=0;}
    executor::ComputeSync();Copy::FenceProxyAsyncShared();executor::ComputeSync();
    MmaQK qk;MmaPV pv;
    auto qk_thread=qk.get_slice(thread);
    auto pv_thread=pv.get_slice(thread);
    auto output=partition_fragment_C(pv,Shape<Int<D>,_8>{});clear(output);
    auto output_coordinates=pv_thread.partition_C(make_identity_tensor(Shape<Int<D>,_8>{}));
    auto score_coordinates=qk_thread.partition_C(make_identity_tensor(Shape<_64,_8>{}));
    for(int wave=0;wave<layout.waves;++wave) {
      auto cursor=sequence+wave;pages.AwaitFull(cursor);
      auto* page=reinterpret_cast<Element*>(pages.Page(cursor));
      int start=begin+wave*64;
      if(warp==0 && p.past>=start && p.past<min(start+64,end)
#if TILEMEGA_QKV_EPILOGUE
          && !p.qkv_preprocessed
#endif
          )
        Owner::NewRow(p,page,p.past-start,b,g);
      executor::ComputeSync();Copy::FenceProxyAsyncShared();executor::ComputeSync();
      auto key=make_tensor(make_smem_ptr(page),backend::Sm90KLayout<64,D>{});
      auto query=make_tensor(make_smem_ptr(s.query),QueryLayout{});
      auto keys=qk_thread.partition_fragment_A(key);
      auto queries=qk_thread.partition_fragment_B(query);
      auto scores=partition_fragment_C(qk,Shape<_64,_8>{});clear(scores);
      Gmma::Fence(scores);Gmma::Arrive();
      CUTE_UNROLL
      for(int k=0;k<size<2>(keys);++k)gemm(qk,keys(_,_,k),queries(_,_,k),scores);
      Gmma::Commit();Gmma::template Wait<0>();Gmma::Fence(scores);
      CUTE_UNROLL
      for(int i=0;i<size(scores);++i)
        s.scores[get<1>(score_coordinates(i))*64+get<0>(score_coordinates(i))]=scores(i);
      executor::ComputeSync();
      // Eight query owners use a fixed key order. This first candidate favors
      // deterministic arithmetic; warp-parallel softmax remains an E2E choice.
      if(thread<8) {
        int q=thread;float maximum=s.maximum[q];
        for(int key_index=0;key_index<64;++key_index) {
          float score=q<Q && start+key_index<end
              ?s.scores[q*64+key_index]*(1.4426950408889634f/sqrtf(float(D))):-INFINITY;
          s.scores[q*64+key_index]=score;maximum=fmaxf(maximum,score);
        }
        float alpha=isfinite(s.maximum[q])?exp2f(s.maximum[q]-maximum):0;
        float added=0;
        auto probability_offsets=as_position_independent_swizzle_layout(ProbabilityLayout{});
        for(int key_index=0;key_index<64;++key_index) {
          float score=s.scores[q*64+key_index];
          float probability=isfinite(score)?exp2f(score-maximum):0;added+=probability;
          s.probabilities[probability_offsets(q,key_index)]=Element(probability);
        }
        s.maximum[q]=maximum;s.sum[q]=s.sum[q]*alpha+added;s.alpha[q]=alpha;
      }
      executor::ComputeSync();Copy::FenceProxyAsyncShared();executor::ComputeSync();
      CUTE_UNROLL
      for(int i=0;i<size(output);++i)output(i)*=s.alpha[get<1>(output_coordinates(i))];
      using ValueLayout=decltype(composition(backend::Sm90KLayout<64,D>{},
          Layout<Shape<Int<D>,_64>,Stride<_64,_1>>{}));
      auto value=make_tensor(make_smem_ptr(page+64*D),ValueLayout{});
      auto probability=make_tensor(make_smem_ptr(s.probabilities),ProbabilityLayout{});
      auto values=pv_thread.partition_fragment_A(value);
      auto probabilities=pv_thread.partition_fragment_B(probability);
      Gmma::Fence(output);Gmma::Arrive();
      CUTE_UNROLL
      for(int k=0;k<size<2>(values);++k)gemm(pv,values(_,_,k),probabilities(_,_,k),output);
      Gmma::Commit();Gmma::template Wait<0>();Gmma::Fence(output);
      // The last async reader completes before the producer can reuse this slot.
      executor::ComputeSync();pages.Release(cursor);
    }
    sequence+=layout.pages_per_task;
    int splits=AttentionPartialBlocks(p);
    CUTE_UNROLL
    for(int i=0;i<size(output);++i) {
      int d=get<0>(output_coordinates(i)),q=get<1>(output_coordinates(i));
      if(q<Q) {
        float normalized=s.sum[q]>0?output(i)/s.sum[q]:0;
        if constexpr(LocalPartial) {
          local_partial[q*D+d]=normalized;
          if(d==0)local_partial[Q*D+q]=s.sum[q]>0?s.maximum[q]+log2f(s.sum[q]):-INFINITY;
        }else {
        if(splits==1)p.context[((std::size_t(b)*p.heads_kv+g)*Q+q)*D+d]=Element(normalized);
        else {
          std::size_t row=((std::size_t(b)*p.heads_kv+g)*splits+c)*Q+q;
          p.partial[row*D+d]=normalized;
          if(d==0)p.lse[row]=s.sum[q]>0?s.maximum[q]+log2f(s.sum[q]):-INFINITY;
        }
        }
      }
    }
    executor::ComputeSync();
  }
};
} // namespace tilemega::codegen
