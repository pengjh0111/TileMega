// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90Gmma.h>
#include <tilemega/Codegen/tasks/FusedAttentionTaskBody.h>
#include <tilemega/Codegen/executor/Async.cuh>

namespace tilemega::codegen {

// Explicit prefill candidate. RoPE and current-token KV materialization retain
// the existing BF16 rounding sequence; transport remains cp.async until a QKV
// epilogue owns those writes. Only QK/PV computation uses Hopper instructions.
template<class Arch,int D,int Q,int Tokens,int QRows,int KvTile,bool Norm>
struct Sm90PrefillAttentionTaskBody {
  static_assert(Tokens>1 && QRows%64==0 && (D==64 || D==128) && Q>=1 && Q<=8);
  using Element=cutlass::bfloat16_t;
  using Legacy=FusedAttentionTaskBody<Arch,D,Q,Tokens,QRows,KvTile,Norm>;
  using Gmma=backend::Sm90Gmma<Arch>;
  using QKMma=decltype(cute::make_tiled_mma(typename Gmma::template SS<64>{}));
  using PVMma=decltype(cute::make_tiled_mma(typename Gmma::template RS<D,backend::gmma::Major::MN>{}));
  using QLayout=backend::Sm90KLayout<64,D>;
  using VLayout=decltype(cute::tile_to_shape(backend::gmma::Layout_MN_SW128_Atom<Element>{},
      cute::Shape<cute::Int<D>,cute::_64>{}));
  struct SharedStorage {
    alignas(1024) Element query[64*D];
    alignas(1024) Element key[64*D];
    alignas(1024) Element value[64*D];
    alignas(16) float score[64*64];
    float row_max[64],row_sum[64],alpha[64];
  };
  static_assert(sizeof(SharedStorage)==ServingPrefillWgmmaSharedBytes(D));
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
    auto k=make_tensor(make_smem_ptr(s.key),QLayout{});
    auto v=make_tensor(make_smem_ptr(s.value),VLayout{});
    auto rq=qkt.partition_fragment_A(q),rk=qkt.partition_fragment_B(k);
    auto rv=pvt.partition_fragment_B(v);
    for(int first=task_first;first<min(task_first+QRows,Tokens*Q);first+=64) {
      ComputeSync();LoadQuery(p,s,batch,group,first);
      if(ComputeThread()<64){s.row_max[ComputeThread()]=-INFINITY;s.row_sum[ComputeThread()]=0;}
      auto output=partition_fragment_C(PVMma{},Shape<_64,Int<D>>{});clear(output);Gmma::Fence(output);
      for(int begin=block_begin;begin<limit;begin+=64) {
        LoadKV(p,s,batch,group,begin,limit,first,task_first);
        cp_async_wait<0>();Async::FenceProxyAsyncShared();ComputeSync();
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
  }
};
} // namespace tilemega::codegen
