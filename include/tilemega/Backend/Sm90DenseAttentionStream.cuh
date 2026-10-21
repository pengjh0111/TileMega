// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseAttention.h>
#include <tilemega/Backend/Sm90DenseAttentionMma.cuh>

namespace tilemega::backend {

template<int Batch,int D,int Q,bool Norm,int Chunks,int BlockKeys,int Slots,
         int NewRowWork=0,int Capacity=1088>
struct Sm90DenseAttentionStream {
  using Element=cutlass::bfloat16_t;
  using Math=Sm90DenseAttentionMath;
  using Partition=Sm90DenseAttentionPartition<Chunks>;
  using Reduction=Sm90DenseAttention<1,D,Q,Norm,Chunks,NewRowWork,Capacity>;
  using Async=codegen::executor::Async<arch::Sm90>;
  using QK=Sm90DenseAttentionMma<16,D>;
  using PV=Sm90DenseAttentionMma<D,16,true>;
  static_assert((Batch==8 || Batch==16) && (BlockKeys==32 || BlockKeys==64 || BlockKeys==128));
  static_assert(Slots>=2 && Slots<=8 && Capacity>=1087);
  static constexpr int kTasks=Batch*8*Chunks,kThreads=160;
  struct alignas(16) Storage {
    Element query[16*D],new_key[D],new_value[D];
    Element key[Slots*BlockKeys*D],value[Slots*BlockKeys*D];
    float output[4*Q*D],maximum[4*Q],sum[4*Q];
    float weight[(Chunks>4?Chunks:4)*Q],lse[Q];
    unsigned last;
  };
  static constexpr int kSharedBytes=1024+sizeof(Storage);

  __device__ static void Query(Sm90DenseAttentionOperands p,Storage& s,int bg,
                               Sm90DenseKeyRange range) {
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    auto* group=p.qkv+std::size_t(bg)*(Q+2)*D;
    for(int q=warp;q<16;q+=4) {
      if(q<Q)Math::template Rotate<D,Norm>(group+q*D,p.q_norm,p.cosine+p.past*D,
          p.sine+p.past*D,p.epsilon,s.query+q*D);
      else if(lane*8<D)*reinterpret_cast<uint4*>(s.query+q*D+lane*8)=make_uint4(0,0,0,0);
    }
    if(range.Contains(p.past) && warp==Q%4) {
      Math::template Rotate<D,Norm>(group+Q*D,p.k_norm,p.cosine+p.past*D,
          p.sine+p.past*D,p.epsilon,s.new_key,p.key+(std::size_t(bg)*Capacity+p.past)*D);
      if(lane*8<D) {
        uint4 v=*reinterpret_cast<uint4 const*>(group+(Q+1)*D+lane*8);
        *reinterpret_cast<uint4*>(s.new_value+lane*8)=v;
        *reinterpret_cast<uint4*>(p.value+(std::size_t(bg)*Capacity+p.past)*D+lane*8)=v;
      }
    }
  }

  __device__ static __noinline__ void Produce(Sm90DenseAttentionOperands p,
      Sm90DenseRingState& state,Storage& s,int tasks,unsigned phase) {
    int lane=threadIdx.x-128,sequence=0;
    for(int local=0;local<tasks;++local) {
      int task=int(blockIdx.x)+local*int(gridDim.x),bg=task/Chunks;
      auto range=Partition::Range(p.past,task%Chunks,NewRowWork);
      for(int begin=range.begin;begin<range.end;begin+=BlockKeys,++sequence) {
        int slot=sequence%Slots,history=max(0,min(begin+BlockKeys,min(range.end,p.past))-begin);
        Async::Wait(state.empty+slot,(phase>>slot)&1);
        if(lane==0) {
          unsigned prefix=state.prefix_bytes[slot];state.prefix_bytes[slot]=0;
          if(history) {
            Async::ProxyAsyncGlobalFence();
            if(!prefix)Async::ExpectTx(state.full+slot,4*history*D);
            std::size_t offset=(std::size_t(bg)*Capacity+begin)*D;
            int done=prefix/(4*D);
            if(done<history) {
              Async::Bulk(s.key+slot*BlockKeys*D+done*D,p.key+offset+done*D,
                  2*(history-done)*D,state.full+slot);
              Async::Bulk(s.value+slot*BlockKeys*D+done*D,p.value+offset+done*D,
                  2*(history-done)*D,state.full+slot);
            }
          }else Async::Arrive(state.full+slot);
        }
        Async::CompleteCopies(state.full+slot);phase^=1u<<slot;
      }
    }
    __syncwarp();
    if(lane==0 && tasks)for(int i=0;i<Slots;++i)state.producer_generations[i]=(phase>>i)&1;
  }

  __device__ static __noinline__ void Consume(Sm90DenseAttentionOperands p,
      Sm90DenseRingState& state,Storage& s,int tasks,unsigned phase) {
    using namespace cute;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31,sequence=0;
    auto scores_coords=QK::Coordinates();auto output_coords=PV::Coordinates();
    for(int local=0;local<tasks;++local) {
      int task=int(blockIdx.x)+local*int(gridDim.x),bg=task/Chunks;
      auto range=Partition::Range(p.past,task%Chunks,NewRowWork);
      Query(p,s,bg,range);
      // A ready bulk barrier covers historical transfers only. Other warps
      // must also acquire the current-row stores before patching that slot.
      codegen::executor::ComputeSync();
      auto output=PV::Accumulator();
      auto maximum=make_tensor<float>(Shape<_2>{});maximum(0)=-INFINITY;maximum(1)=-INFINITY;
      auto sum=make_tensor<float>(Shape<_2>{});clear(sum);
      for(int begin=range.begin;begin<range.end;begin+=BlockKeys,++sequence) {
        int slot=sequence%Slots,rows=min(BlockKeys,range.end-begin);
        Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
        auto* key=s.key+slot*BlockKeys*D;auto* value=s.value+slot*BlockKeys*D;
        for(int i=threadIdx.x*8;i<BlockKeys*D;i+=128*8) {
          int row=i/D,d=i%D;
          if(row>=rows) {
            *reinterpret_cast<uint4*>(key+i)=make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(value+i)=make_uint4(0,0,0,0);
          }else if(begin+row==p.past) {
            *reinterpret_cast<uint4*>(key+i)=*reinterpret_cast<uint4*>(s.new_key+d);
            *reinterpret_cast<uint4*>(value+i)=*reinterpret_cast<uint4*>(s.new_value+d);
          }
        }
        codegen::executor::ComputeSync();
        for(int interval=warp*16;interval<BlockKeys;interval+=64) {
          auto score=QK::Accumulator();QK::QK(s.query,key+interval*D,score);
          auto block_max=make_tensor<float>(Shape<_2>{});block_max(0)=-INFINITY;block_max(1)=-INFINITY;
          for_each(make_int_sequence<size(score)>{},[&](auto i) {
            constexpr int group=QK::template QueryGroup<int(i)>();
            int q=get<0>(scores_coords(i)),position=interval+get<1>(scores_coords(i));
            score(i)=q<Q && position<rows?score(i)*(Reduction::kLog2E/sqrtf(float(D))):-INFINITY;
            block_max(Int<group>{})=fmaxf(block_max(Int<group>{}),score(i));
          });
          auto scale=make_tensor<float>(Shape<_2>{});
          for_each(make_int_sequence<2>{},[&](auto r) {
            float m=block_max(r);
            m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,1));m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,2));
            float next=fmaxf(maximum(r),m);scale(r)=isfinite(maximum(r))?exp2f(maximum(r)-next):0;
            maximum(r)=next;
          });
          auto increment=make_tensor<float>(Shape<_2>{});clear(increment);
          for_each(make_int_sequence<size(score)>{},[&](auto i) {
            constexpr int group=QK::template QueryGroup<int(i)>();
            score(i)=isfinite(score(i))?exp2f(score(i)-maximum(Int<group>{})):0;
            increment(Int<group>{})+=score(i);
          });
          for_each(make_int_sequence<2>{},[&](auto r) {
            float value=increment(r);value+=__shfl_xor_sync(0xffffffff,value,1);
            value+=__shfl_xor_sync(0xffffffff,value,2);sum(r)=sum(r)*scale(r)+value;
          });
          for_each(make_int_sequence<size(output)>{},[&](auto i) {
            constexpr int group=PV::template QueryGroup<int(i)>();output(i)*=scale(Int<group>{});
          });
          PV::PV(score,scores_coords,value+interval*D,output);
        }
        Async::Arrive(state.empty+slot);
      }
      for_each(make_int_sequence<size(output)>{},[&](auto i) {
        int q=get<0>(output_coords(i)),d=get<1>(output_coords(i));
        if(q<Q)s.output[(warp*Q+q)*D+d]=output(i);
      });
      if((lane&3)==0 && lane/4<Q) {
        s.maximum[warp*Q+lane/4]=maximum(0);s.sum[warp*Q+lane/4]=sum(0);
      }
      codegen::executor::ComputeSync();
      Reduction::Partial(p,s,task);
      if constexpr(Chunks>1)
        if(Sm90DenseLastArriver::Publish<Chunks>(p.arrivals+bg,&s.last))Reduction::Merge(p,s,bg);
      codegen::executor::ComputeSync();
    }
    if(threadIdx.x==0 && tasks)for(int i=0;i<Slots;++i)state.consumer_generations[i]=(phase>>i)&1;
  }

  template<class Tail=Sm90DenseNoTail>
  __device__ static void Run(Sm90DenseAttentionOperands p,char* shared,
      Sm90DenseRingState* persistent=nullptr,Tail tail={}) {
    auto& state=persistent?*persistent:*reinterpret_cast<Sm90DenseRingState*>(shared);
    auto& s=*reinterpret_cast<Storage*>(shared+1024);
    int tasks=kTasks/int(gridDim.x)+(int(blockIdx.x)<kTasks%int(gridDim.x));
    unsigned phase=0;
    #pragma unroll
    for(int i=0;i<Slots;++i)phase|=(threadIdx.x<128?
        state.consumer_generations[i]:state.producer_generations[i])<<i;
    if(threadIdx.x<128)Consume(p,state,s,tasks,phase);
    else {
      if constexpr(Tail::kBefore) {
        if(threadIdx.x==128)tail.Before();
        __syncwarp();
      }
      Produce(p,state,s,tasks,phase);
      if(threadIdx.x==128)tail();
    }
    __syncthreads();
  }
};

}  // namespace tilemega::backend
