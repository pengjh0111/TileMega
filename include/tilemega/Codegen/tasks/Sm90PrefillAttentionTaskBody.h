// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90Gmma.h>
#include <tilemega/Codegen/tasks/FusedAttentionTaskBody.h>
#include <tilemega/Codegen/executor/Async.cuh>
#if TILEMEGA_PREFILL_ATTENTION_TMA
#include <tilemega/Codegen/executor/TensorMap.h>
#endif

namespace tilemega::codegen {

// Explicit prefill candidate with QK SS / PV RS computation. The default
// transport retains cp.async and the existing RoPE rounding sequence. Opt-in
// TMA uses preprocessed Q/K/V, strided per-head Q maps, and two KV slots.
template<class Arch,int D,int Q,int Tokens,int QRows,int KvTile,bool Norm>
struct Sm90PrefillAttentionTaskBody {
  static_assert(Tokens>1 && QRows%64==0 && (D==64 || D==128) && Q>=1 && Q<=8);
  using Element=cutlass::bfloat16_t;
  using Legacy=FusedAttentionTaskBody<Arch,D,Q,Tokens,QRows,KvTile,Norm>;
  using Gmma=backend::Sm90Gmma<Arch>;
  using QKMma=decltype(cute::make_tiled_mma(typename Gmma::template SS<64>{}));
  using PVMma=decltype(cute::make_tiled_mma(typename Gmma::template RS<D,backend::gmma::Major::MN>{}));
  using QLayout=backend::Sm90KLayout<64,D>;
  static constexpr bool kTma=TILEMEGA_PREFILL_ATTENTION_TMA!=0;
  static constexpr int kQueryMapRows=ServingPrefillQueryMapRows(Q);
  using QueryMapLayout=backend::Sm90KLayout<kQueryMapRows,D>;
  using VLayout=std::conditional_t<kTma,
      decltype(cute::composition(QLayout{},cute::Layout<cute::Shape<cute::Int<D>,cute::_64>,cute::Stride<cute::_64,cute::_1>>{})),
      decltype(cute::tile_to_shape(backend::gmma::Layout_MN_SW128_Atom<Element>{},
      cute::Shape<cute::Int<D>,cute::_64>{}))>;
  struct SharedStorage {
    alignas(1024) Element query[64*D];
    alignas(1024) Element key[(kTma?2:1)*64*D];
    alignas(1024) Element value[(kTma?2:1)*64*D];
#if TILEMEGA_PREFILL_ATTENTION_TMA
    alignas(1024) Element query_staging[Q*kQueryMapRows*D];
    std::uint64_t query_barrier,kv_barrier[2];
#endif
    alignas(16) float score[64*64];
    float row_max[64],row_sum[64],alpha[64];
  };
  static_assert(sizeof(SharedStorage)==ServingPrefillWgmmaSharedBytes(D,Q,kTma));
  template<class Emit>
  __device__ static void PrefetchRanges(ServingAttentionOperands const& p,int b,int g,
      int c,unsigned mask,Emit emit) {Legacy::PrefetchRanges(p,b,g,c,mask,emit);}

  __device__ static void LoadQuery(ServingAttentionOperands const& p,SharedStorage& s,
      int batch,int group,int first) {
    auto layout=cute::as_position_independent_swizzle_layout(QLayout{});
    constexpr int width=(Q+2)*D;
    int warp=ComputeThread()/32,lane=ComputeThread()%32;
    for(int row=warp;row<64;row+=4) {
      int r=first+row,token=r/Q,head=r%Q;
      if(r<Tokens*Q) {
        auto* x=p.qkv+(batch*Tokens+token)*p.heads_kv*width+group*width+head*D;
#if TILEMEGA_QKV_EPILOGUE
        if(p.qkv_preprocessed) {
          if(lane*8<D)*reinterpret_cast<uint4*>(s.query+layout(row,lane*8))=backend::LoadGlobal16(x+lane*8);
          continue;
        }
#endif
        Legacy::RotateRow(x,p.q_norm,p.cosine+(p.past+token)*D,p.sine+(p.past+token)*D,
            p.epsilon,[&](int d,uint4 value){*reinterpret_cast<uint4*>(s.query+layout(row,d))=value;});
      }else if(lane*8<D)*reinterpret_cast<uint4*>(s.query+layout(row,lane*8))=make_uint4(0,0,0,0);
    }
  }
  __device__ static void LoadKV(ServingAttentionOperands const& p,SharedStorage& s,
      int batch,int group,int begin,int limit,int first,int task_first) {
    using Async=executor::Async<Arch>;
    auto kl=cute::as_position_independent_swizzle_layout(QLayout{});
    auto vl=cute::as_position_independent_swizzle_layout(VLayout{});
    for(int i=ComputeThread()*8;i<64*D;i+=128*8) {
      int row=i/D,d=i%D,pos=begin+row;
      if(pos<p.past && pos<limit) {
        std::size_t offset=((std::size_t(batch)*p.heads_kv+group)*p.capacity+pos)*D+d;
        Async::Copy16Bytes(s.key+kl(row,d),p.key_cache+offset,16);
        Async::Copy16Bytes(s.value+vl(d,row),p.value_cache+offset,16);
      }else if(pos>=limit) {
        *reinterpret_cast<uint4*>(s.key+kl(row,d))=make_uint4(0,0,0,0);
        *reinterpret_cast<uint4*>(s.value+vl(d,row))=make_uint4(0,0,0,0);
      }
    }
    int warp=ComputeThread()/32,lane=ComputeThread()%32;
    constexpr int width=(Q+2)*D;
    for(int row=warp;row<64;row+=4) {
      int pos=begin+row;
      if(pos<p.past || pos>=limit)continue;
#if TILEMEGA_QKV_EPILOGUE
      if(p.qkv_preprocessed) {
        if(lane*8<D) {
          int d=lane*8;auto offset=((std::size_t(batch)*p.heads_kv+group)*p.capacity+pos)*D+d;
          *reinterpret_cast<uint4*>(s.key+kl(row,d))=backend::LoadGlobal16(p.key_cache+offset);
          *reinterpret_cast<uint4*>(s.value+vl(d,row))=backend::LoadGlobal16(p.value_cache+offset);
        }
        continue;
      }
#endif
      int token=pos-p.past;
      auto* x=p.qkv+(batch*Tokens+token)*p.heads_kv*width+group*width+Q*D;
      bool writer=first==task_first && token*Q>=task_first && token*Q<task_first+QRows;
      Legacy::RotateRow(x,p.k_norm,p.cosine+pos*D,p.sine+pos*D,p.epsilon,[&](int d,uint4 value) {
        *reinterpret_cast<uint4*>(s.key+kl(row,d))=value;
        if(writer)*reinterpret_cast<uint4*>(p.key_cache+
            ((std::size_t(batch)*p.heads_kv+group)*p.capacity+pos)*D+d)=value;
      });
      if(lane*8<D) {
        int d=lane*8;auto value=backend::LoadGlobal16(x+D+d);
        *reinterpret_cast<uint4*>(s.value+vl(d,row))=value;
        if(writer)*reinterpret_cast<uint4*>(p.value_cache+
            ((std::size_t(batch)*p.heads_kv+group)*p.capacity+pos)*D+d)=value;
      }
    }
    cute::cp_async_fence();
  }
  template<class CLayout,class AShape>
  __device__ static auto RegisterOperandLayout(CLayout c,AShape a) {
    using namespace cute;
    return make_layout(make_shape(a,shape<1>(c),make_shape(shape<2>(c),size<0>(c)/size(a))),
        make_stride(stride<0>(c),stride<1>(c),make_stride(stride<2>(c),size<2>(a)*stride<0,2>(c))));
  }
#if TILEMEGA_PREFILL_ATTENTION_TMA
  static_assert(TILEMEGA_QKV_EPILOGUE,"prefill TMA requires QKV preprocessing");
  __device__ static __forceinline__ void InitTma(ServingAttentionOperands const& p,SharedStorage& s) {
    using A=executor::Async<Arch>;
    if(!p.qkv_preprocessed || !p.query_tensor_maps || !p.key_tensor_map || !p.value_tensor_map) {
      asm volatile("trap;");return;
    }
    if(ComputeThread()==0) {
      A::Init(&s.query_barrier,1);A::Init(&s.kv_barrier[0],1);A::Init(&s.kv_barrier[1],1);A::InitFence();
      auto* maps=static_cast<executor::TensorMap const*>(p.query_tensor_maps);
      for(int head=0;head<Q;++head)A::PrefetchTensorMap(maps+head);
      A::PrefetchTensorMap(p.key_tensor_map);A::PrefetchTensorMap(p.value_tensor_map);
    }
    ComputeSync();A::ProxyAsyncGlobalFence();
  }
  __device__ static __forceinline__ void LoadQueryTma(ServingAttentionOperands const& p,
      SharedStorage& s,int first,unsigned phase) {
    using A=executor::Async<Arch>;
    auto raw=cute::as_position_independent_swizzle_layout(QueryMapLayout{});
    auto query=cute::as_position_independent_swizzle_layout(QLayout{});
    if(ComputeThread()==0) {
      A::ExpectTx(&s.query_barrier,Q*kQueryMapRows*D*sizeof(Element));
      auto* maps=static_cast<executor::TensorMap const*>(p.query_tensor_maps);
      for(int head=0;head<Q;++head)for(int d=0;d<D;d+=64)
        A::Tensor2D(s.query_staging+head*kQueryMapRows*D+raw(0,d),maps+head,d,first/Q,&s.query_barrier);
    }
    A::Wait(&s.query_barrier,phase);ComputeSync();
    // QPerKV=3 crosses token boundaries within each 64-row query tile.
    // Per-head strided maps preserve the holes occupied by K/V in QKV.
    for(int i=ComputeThread()*8;i<64*D;i+=128*8) {
      int row=i/D,d=i%D,logical=first+row;
      auto value=logical<Tokens*Q?*reinterpret_cast<uint4*>(s.query_staging+
          (logical%Q)*kQueryMapRows*D+raw(logical/Q-first/Q,d)):make_uint4(0,0,0,0);
      *reinterpret_cast<uint4*>(s.query+query(row,d))=value;
    }
    A::FenceProxyAsyncShared();ComputeSync();
  }
  __device__ static __forceinline__ void IssueKvTma(ServingAttentionOperands const& p,
      SharedStorage& s,int slot,int begin) {
    using A=executor::Async<Arch>;
    auto layout=cute::as_position_independent_swizzle_layout(QLayout{});
    if(ComputeThread()==0) {
      A::ExpectTx(&s.kv_barrier[slot],4*64*D);
      for(int d=0;d<D;d+=64) {
        A::Tensor2D(s.key+slot*64*D+layout(0,d),p.key_tensor_map,d,begin,&s.kv_barrier[slot]);
        A::Tensor2D(s.value+slot*64*D+layout(0,d),p.value_tensor_map,d,begin,&s.kv_barrier[slot]);
      }
    }
  }
  __device__ static __forceinline__ void WaitKvTma(SharedStorage& s,int slot,
      unsigned phase,int begin,int limit) {
    using A=executor::Async<Arch>;
    A::Wait(&s.kv_barrier[slot],phase);ComputeSync();
    if(begin+64>limit) {
      auto layout=cute::as_position_independent_swizzle_layout(QLayout{});
      for(int i=ComputeThread()*8;i<64*D;i+=128*8)if(begin+i/D>=limit) {
        *reinterpret_cast<uint4*>(s.key+slot*64*D+layout(i/D,i%D))=make_uint4(0,0,0,0);
        *reinterpret_cast<uint4*>(s.value+slot*64*D+layout(i/D,i%D))=make_uint4(0,0,0,0);
      }
      A::FenceProxyAsyncShared();ComputeSync();
    }
  }
#endif
  __device__ static __forceinline__ void Run(ServingAttentionOperands const& p,SharedStorage& s,
      int batch,int group,int qb,int chunk) {
    using namespace cute;
    using Async=executor::Async<Arch>;
    if(chunk!=0 || p.block_extent<p.past+Tokens) {asm volatile("trap;");return;}
    int task_first=qb*QRows;
    int limit=min((chunk+1)*p.block_extent,p.past+Tokens);
    limit=min(limit,p.past+(min(task_first+QRows,Tokens*Q)-1)/Q+1);
    int block_begin=chunk*p.block_extent;
    if(block_begin>=limit)return;
    auto qkt=QKMma{}.get_slice(ComputeThread());
    auto pvt=PVMma{}.get_slice(ComputeThread());
    auto qc=qkt.partition_C(make_identity_tensor(Shape<_64,_64>{}));
    auto oc=pvt.partition_C(make_identity_tensor(Shape<_64,Int<D>>{}));
    auto q=make_tensor(make_smem_ptr(s.query),QLayout{});
    auto rq=qkt.partition_fragment_A(q);
#if TILEMEGA_PREFILL_ATTENTION_TMA
    InitTma(p,s);unsigned query_generation=0,kv_generation0=0,kv_generation1=0;
#endif
    for(int first=task_first;first<min(task_first+QRows,Tokens*Q);first+=64) {
      ComputeSync();
#if TILEMEGA_PREFILL_ATTENTION_TMA
      LoadQueryTma(p,s,first,query_generation++&1);
      int blocks=(limit-block_begin+63)/64;
      auto issue=[&](int it) {
        if(it<blocks) {
          IssueKvTma(p,s,it&1,block_begin+64*it);
          if(it&1)++kv_generation1;else ++kv_generation0;
        }
      };
      issue(0);issue(1);
#else
      LoadQuery(p,s,batch,group,first);
#endif
      if(ComputeThread()<64){s.row_max[ComputeThread()]=-INFINITY;s.row_sum[ComputeThread()]=0;}
      auto output=partition_fragment_C(PVMma{},Shape<_64,Int<D>>{});clear(output);Gmma::Fence(output);
      for(int begin=block_begin;begin<limit;begin+=64) {
#if TILEMEGA_PREFILL_ATTENTION_TMA
        int iteration=(begin-block_begin)/64,slot=iteration&1;
        WaitKvTma(s,slot,((slot?kv_generation1:kv_generation0)-1)&1,begin,limit);
#else
        constexpr int slot=0;
        LoadKV(p,s,batch,group,begin,limit,first,task_first);
        cp_async_wait<0>();Async::FenceProxyAsyncShared();ComputeSync();
#endif
        auto k=make_tensor(make_smem_ptr(s.key+slot*64*D),QLayout{});
        auto v=make_tensor(make_smem_ptr(s.value+slot*64*D),VLayout{});
        auto rk=qkt.partition_fragment_B(k);
        auto rv=pvt.partition_fragment_B(v);
        auto score=partition_fragment_C(QKMma{},Shape<_64,_64>{});clear(score);Gmma::Fence(score);
        Gmma::Arrive();
        CUTE_UNROLL
        for(int d=0;d<size<2>(rq);++d)gemm(QKMma{},rq(_,_,d),rk(_,_,d),score);
        Gmma::Commit();Gmma::template Wait<0>();Gmma::Fence(score);
        CUTE_UNROLL
        for(int i=0;i<size(score);++i) {
          int row=get<0>(qc(i)),col=get<1>(qc(i)),token=(first+row)/Q,pos=begin+col;
          s.score[row*64+col]=first+row<Tokens*Q && pos<limit && pos<=p.past+token
              ?score(i)*(1.4426950408889634f/sqrtf(float(D))):-INFINITY;
        }
        ComputeSync();
        int lane=ComputeThread()%32,warp=ComputeThread()/32;
        for(int row=warp;row<64;row+=4) {
          float peak=fmaxf(s.score[row*64+lane],s.score[row*64+lane+32]);
          for(int shift=16;shift;shift/=2)peak=fmaxf(peak,__shfl_xor_sync(0xffffffffu,peak,shift));
          float old=s.row_max[row],maximum=fmaxf(old,peak);
          float alpha=isfinite(old)?exp2f(old-maximum):0;
          float p0=isfinite(s.score[row*64+lane])?exp2f(s.score[row*64+lane]-maximum):0;
          float p1=isfinite(s.score[row*64+lane+32])?exp2f(s.score[row*64+lane+32]-maximum):0;
          s.score[row*64+lane]=p0;s.score[row*64+lane+32]=p1;
          float sum=p0+p1;for(int shift=16;shift;shift/=2)sum+=__shfl_xor_sync(0xffffffffu,sum,shift);
          if(lane==0){s.row_max[row]=maximum;s.row_sum[row]=alpha*s.row_sum[row]+sum;s.alpha[row]=alpha;}
        }
        ComputeSync();
        CUTE_UNROLL
        for(int i=0;i<size(output);++i)output(i)*=s.alpha[get<0>(oc(i))];
        auto probability=make_fragment_like<Element>(RegisterOperandLayout(score.layout(),shape<1>(typename PVMma::LayoutA_TV{})));
        auto pc=make_tensor(probability.data(),score.layout());
        CUTE_UNROLL
        for(int i=0;i<size(score);++i)pc(i)=Element(s.score[get<0>(qc(i))*64+get<1>(qc(i))]);
        Gmma::Fence(output);Gmma::Arrive();
        CUTE_UNROLL
        for(int r=0;r<size<2>(probability);++r)gemm(PVMma{},probability(_,_,r),rv(_,_,r),output);
        Gmma::Commit();Gmma::template Wait<0>();Gmma::Fence(output);ComputeSync();
#if TILEMEGA_PREFILL_ATTENTION_TMA
        // PV is the last reader of this slot; the next TMA cannot precede it.
        issue(iteration+2);
#endif
      }
      CUTE_UNROLL
      for(int i=0;i<size(output);++i) {
        int row=get<0>(oc(i)),d=get<1>(oc(i)),r=first+row;
        if(r<Tokens*Q) {
          auto index=((std::size_t(batch)*Tokens+r/Q)*p.heads_kv*Q+group*Q+r%Q)*D+d;
          p.context[index]=Element(s.row_sum[row]>0?output(i)/s.row_sum[row]:0);
        }
      }
      ComputeSync();
    }
#if TILEMEGA_PREFILL_ATTENTION_TMA
    if(ComputeThread()==0) {
      Async::Invalidate(&s.query_barrier);
      Async::Invalidate(&s.kv_barrier[0]);Async::Invalidate(&s.kv_barrier[1]);
    }
    ComputeSync();
#endif
  }
};
} // namespace tilemega::codegen
