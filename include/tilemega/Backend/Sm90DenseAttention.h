// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90DenseAttentionMath.cuh>
#include <tilemega/Backend/Sm90DenseAttentionPartition.h>
#include <tilemega/Backend/Sm90DenseLastArriver.cuh>
#include <tilemega/Backend/Sm90DenseRing.cuh>

namespace tilemega::backend {

struct Sm90DenseAttentionOperands {
  cutlass::bfloat16_t const* qkv;
  cutlass::bfloat16_t *key,*value,*output;
  cutlass::bfloat16_t const *cosine,*sine,*q_norm,*k_norm;
  float *partial,*lse;
  std::uint64_t* arrivals;
  int past;
  float epsilon;
};

// Low-batch chunks issue their complete historical interval at once. The
// producer never reads the current cache row; its owner rotates and publishes
// that row locally, independently of the historical asynchronous transfers.
template<int Batch,int D,int Q,bool Norm,int Chunks,int NewRowWork=0,int Capacity=1088>
struct Sm90DenseAttention {
  using Element=cutlass::bfloat16_t;
  using Math=Sm90DenseAttentionMath;
  using Partition=Sm90DenseAttentionPartition<Chunks>;
  using Async=codegen::executor::Async<arch::Sm90>;
  static_assert(Batch>=1 && Batch<=4 && (D==64 || D==128) && Q>=2 && Q<=4);
  static_assert(NewRowWork>=0 && Capacity>=1087);
  static constexpr int kMaxKeys=Partition::MaxKeys(Capacity,NewRowWork);
  static constexpr int kThreads=160,kTasks=Batch*8*Chunks;
  static constexpr float kLog2E=1.4426950408889634f;
  struct alignas(16) Storage {
    Element query[Q*D],key[kMaxKeys*D],value[kMaxKeys*D];
    float output[4*Q*D],maximum[4*Q],sum[4*Q];
    float weight[(Chunks>4?Chunks:4)*Q],lse[Q];
    unsigned last;
  };
  static constexpr int kSharedBytes=1024+sizeof(Storage);

  __device__ static void Produce(Sm90DenseAttentionOperands p,Sm90DenseRingState& state,
                                 Storage& s,int tasks,unsigned phase) {
    int lane=threadIdx.x-128;
    for(int local=0;local<tasks;++local) {
      int task=int(blockIdx.x)+local*int(gridDim.x),bg=task/Chunks,c=task%Chunks;
      auto range=Partition::Range(p.past,c,NewRowWork);
      int rows=min(range.end,p.past)-range.begin;
      Async::Wait(state.empty,phase);
      if(lane==0) {
        unsigned prefix=state.prefix_bytes[0];state.prefix_bytes[0]=0;
        if(rows) {
          Async::ProxyAsyncGlobalFence();
          if(!prefix)Async::ExpectTx(state.full,4*rows*D);
          std::size_t offset=(std::size_t(bg)*Capacity+range.begin)*D;
          int done=prefix/(4*D);
          if(done<rows) {
            Async::Bulk(s.key+done*D,p.key+offset+done*D,2*(rows-done)*D,state.full);
            Async::Bulk(s.value+done*D,p.value+offset+done*D,2*(rows-done)*D,state.full);
          }
        } else Async::Arrive(state.full);
      }
      Async::CompleteCopies(state.full);
      phase^=1;
    }
    __syncwarp();
    if(lane==0 && tasks)state.producer_generations[0]=phase;
  }

  __device__ static void QueryAndNewRow(Sm90DenseAttentionOperands p,Storage& s,
                                       int bg,Sm90DenseKeyRange range) {
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    auto* group=p.qkv+std::size_t(bg)*(Q+2)*D;
    auto* cosine=p.cosine+p.past*D;auto* sine=p.sine+p.past*D;
    if(warp<Q)Math::template Rotate<D,Norm>(group+warp*D,p.q_norm,cosine,sine,
                                           p.epsilon,s.query+warp*D);
    if(range.Contains(p.past) && warp==Q%4) {
      int row=p.past-range.begin;
      auto* key=p.key+(std::size_t(bg)*Capacity+p.past)*D;
      auto* value=p.value+(std::size_t(bg)*Capacity+p.past)*D;
      Math::template Rotate<D,Norm>(group+Q*D,p.k_norm,cosine,sine,p.epsilon,
                                    s.key+row*D,key);
      if(lane*8<D) {
        uint4 v=*reinterpret_cast<uint4 const*>(group+(Q+1)*D+lane*8);
        *reinterpret_cast<uint4*>(s.value+row*D+lane*8)=v;
        *reinterpret_cast<uint4*>(value+lane*8)=v;
      }
    }
  }

  __device__ static void Compute(Storage& s,int keys) {
    using namespace cute;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    auto query=make_tensor<float>(Shape<Int<Q>,Int<D/64>,_2>{});
    auto output=make_tensor<float>(Shape<Int<Q>,Int<D/64>,_2>{});clear(output);
    auto maximum=make_tensor<float>(Shape<Int<Q>>{});
    auto sum=make_tensor<float>(Shape<Int<Q>>{});clear(sum);
    for_each(make_int_sequence<Q>{},[&](auto q) {
      maximum(q)=-INFINITY;
      for_each(make_int_sequence<D/64>{},[&](auto segment) {
        float2 x=Math::Pair(s.query+int(q)*D+int(segment)*64+lane*2);
        query(q,segment,0)=x.x;query(q,segment,1)=x.y;
      });
    });
    for(int row=warp;row<keys;row+=4) {
      auto key=make_tensor<float>(Shape<Int<D/64>,_2>{});
      auto value=make_tensor<float>(Shape<Int<D/64>,_2>{});
      for_each(make_int_sequence<D/64>{},[&](auto segment) {
        float2 k=Math::Pair(s.key+row*D+int(segment)*64+lane*2);
        float2 v=Math::Pair(s.value+row*D+int(segment)*64+lane*2);
        key(segment,0)=k.x;key(segment,1)=k.y;
        value(segment,0)=v.x;value(segment,1)=v.y;
      });
      for_each(make_int_sequence<Q>{},[&](auto q) {
        float dot=0;
        for_each(make_int_sequence<D/64>{},[&](auto segment) {
          dot+=query(q,segment,0)*key(segment,0);
          dot+=query(q,segment,1)*key(segment,1);
        });
        float score=Math::WarpSum(dot)*(kLog2E/sqrtf(float(D)));
        float next=fmaxf(maximum(q),score),scale=exp2f(maximum(q)-next);
        float probability=exp2f(score-next);
        sum(q)=sum(q)*scale+probability;maximum(q)=next;
        for_each(make_int_sequence<D/64>{},[&](auto segment) {
          output(q,segment,0)=output(q,segment,0)*scale+probability*value(segment,0);
          output(q,segment,1)=output(q,segment,1)*scale+probability*value(segment,1);
        });
      });
    }
    for_each(make_int_sequence<Q>{},[&](auto q) {
      if(lane==0) {s.maximum[warp*Q+int(q)]=maximum(q);s.sum[warp*Q+int(q)]=sum(q);}
      for_each(make_int_sequence<D/64>{},[&](auto segment) {
        *reinterpret_cast<float2*>(s.output+(warp*Q+int(q))*D+int(segment)*64+lane*2)=
            make_float2(output(q,segment,0),output(q,segment,1));
      });
    });
  }

  template<class Scratch>
  __device__ static void Partial(Sm90DenseAttentionOperands p,Scratch& s,int task) {
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    if(warp<Q) {
      float m=lane<4?s.maximum[lane*Q+warp]:-INFINITY;
      m=Math::WarpMax(m);
      float weight=lane<4 && isfinite(m)?exp2f(s.maximum[lane*Q+warp]-m):0;
      float total=Math::WarpSum(lane<4?weight*s.sum[lane*Q+warp]:0);
      if(lane<4)s.weight[lane*Q+warp]=total?weight/total:0;
      if(lane==0)s.lse[warp]=total?m+log2f(total):-INFINITY;
    }
    codegen::executor::ComputeSync();
    for(int i=threadIdx.x*4;i<Q*D;i+=128*4) {
      int q=i/D;float4 result=make_float4(0,0,0,0);
      #pragma unroll
      for(int rank=0;rank<4;++rank) {
        float w=s.weight[rank*Q+q];
        float4 v=*reinterpret_cast<float4*>(s.output+rank*Q*D+i);
        result.x+=w*v.x;result.y+=w*v.y;result.z+=w*v.z;result.w+=w*v.w;
      }
      if constexpr(Chunks==1) {
        Element* out=p.output+std::size_t(task)*Q*D+i;
        out[0]=Element(result.x);out[1]=Element(result.y);
        out[2]=Element(result.z);out[3]=Element(result.w);
      } else *reinterpret_cast<float4*>(p.partial+std::size_t(task)*Q*D+i)=result;
    }
    if constexpr(Chunks>1)if(threadIdx.x<Q)p.lse[task*Q+threadIdx.x]=s.lse[threadIdx.x];
  }

  template<class Scratch>
  __device__ static void Merge(Sm90DenseAttentionOperands p,Scratch& s,int bg) {
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    if(warp<Q) {
      float m=-INFINITY;
      #pragma unroll
      for(int c=lane;c<Chunks;c+=32)m=fmaxf(m,p.lse[(bg*Chunks+c)*Q+warp]);
      m=Math::WarpMax(m);
      float total=0;
      #pragma unroll
      for(int c=lane;c<Chunks;c+=32) {
        float w=exp2f(p.lse[(bg*Chunks+c)*Q+warp]-m);
        s.weight[c*Q+warp]=w;total+=w;
      }
      total=Math::WarpSum(total);
      #pragma unroll
      for(int c=lane;c<Chunks;c+=32)s.weight[c*Q+warp]/=total;
    }
    codegen::executor::ComputeSync();
    for(int i=threadIdx.x*4;i<Q*D;i+=128*4) {
      int q=i/D;float4 result=make_float4(0,0,0,0);
      #pragma unroll
      for(int c=0;c<Chunks;++c) {
        float w=s.weight[c*Q+q];
        float4 v=*reinterpret_cast<float4 const*>(p.partial+std::size_t(bg*Chunks+c)*Q*D+i);
        result.x+=w*v.x;result.y+=w*v.y;result.z+=w*v.z;result.w+=w*v.w;
      }
      Element* out=p.output+std::size_t(bg)*Q*D+i;
      out[0]=Element(result.x);out[1]=Element(result.y);
      out[2]=Element(result.z);out[3]=Element(result.w);
    }
  }

  __device__ static __noinline__ void Consume(Sm90DenseAttentionOperands p,
      Sm90DenseRingState& state,Storage& s,int tasks,unsigned phase) {
    for(int local=0;local<tasks;++local) {
      int task=int(blockIdx.x)+local*int(gridDim.x),bg=task/Chunks,c=task%Chunks;
      auto range=Partition::Range(p.past,c,NewRowWork);
      QueryAndNewRow(p,s,bg,range);
      Async::Wait(state.full,phase);phase^=1;
      codegen::executor::ComputeSync();
      Compute(s,range.end-range.begin);
      // Historical operands are dead before publishing empty; the producer
      // may refill them while consumers finish the deterministic reductions.
      Async::Arrive(state.empty);
      codegen::executor::ComputeSync();
      Partial(p,s,task);
      if constexpr(Chunks>1)
        if(Sm90DenseLastArriver::Publish<Chunks>(p.arrivals+bg,&s.last))Merge(p,s,bg);
      codegen::executor::ComputeSync();
    }
    if(threadIdx.x==0 && tasks)state.consumer_generations[0]=phase;
  }

  template<class Tail=Sm90DenseNoTail>
  __device__ static void Run(Sm90DenseAttentionOperands p,char* shared,
      Sm90DenseRingState* persistent=nullptr,Tail tail={}) {
    auto& state=persistent?*persistent:*reinterpret_cast<Sm90DenseRingState*>(shared);
    auto& s=*reinterpret_cast<Storage*>(shared+1024);
    int tasks=kTasks/int(gridDim.x)+(int(blockIdx.x)<kTasks%int(gridDim.x));
    if(threadIdx.x<128)Consume(p,state,s,tasks,state.consumer_generations[0]);
    else {
      if constexpr(Tail::kBefore) {
        if(threadIdx.x==128)tail.Before();
      }
      Produce(p,state,s,tasks,state.producer_generations[0]);
      if(threadIdx.x==128)tail();
    }
    __syncthreads();
  }
};

}  // namespace tilemega::backend
