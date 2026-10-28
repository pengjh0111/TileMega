// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90ForwardEpilogue.cuh>
#include <tilemega/Backend/Sm90DenseAttentionMath.cuh>
#include <tilemega/Backend/Sm90DenseLayout.h>
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <tilemega/Backend/Sm90DenseRoles.cuh>
#include <tilemega/Codegen/executor/Async.cuh>

// Encoder attention for the native DNN forward (prompt §10 N-1 item 7).
// A task is (head, 16 queries); BERT seq 128 x 12 heads x D64 gives 96 tasks.
// The head's K and V (16 KB each at seq 128) are read once through a
// SWIZZLE_128B map, S = Q K^T is one 16 x Seq tile, the softmax is exact over
// the whole row (no online rescaling), and O = P V finishes the task.
namespace tilemega::backend {

struct Sm90ForwardEncoderAttentionOperands {
  // The three projections live in the plan's packed [Seq][3 Heads D] buffer,
  // so each is a 3D {D, Heads, Seq} view of it with its own base offset.
  CUtensorMap const* key;
  CUtensorMap const* value;
  CUtensorMap const* query;
  cutlass::bfloat16_t* output;        // [Seq][Heads * D]
  float scale;
  int heads,sequence;
};

template<int Seq,int D,int Heads,int Rows,int Slots>
struct Sm90ForwardEncoderAttention {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  using Math=Sm90DenseAttentionMath;
  static_assert(Rows==16 && D==64 && Seq%64==0);
  static_assert(Slots>=1 && Slots<=2);
  static constexpr int kThreads=160,kPanels=D/64;
  static constexpr int kTasks=Heads*(Seq/Rows);
  static constexpr int kKvBytes=2*Seq*D;
  static constexpr int kQueryBytes=2*Rows*D;
  static constexpr int kScoreBytes=4*Rows*Seq;
  static constexpr Sm90DenseLayout kLayout{1024,1024,
      1024+Slots*(2*kKvBytes+kQueryBytes),
      (1024+Slots*(2*kKvBytes+kQueryBytes)+kScoreBytes+1023)/1024*1024};
  static constexpr int kSharedBytes=kLayout.end;

  __device__ static __forceinline__ unsigned Swizzled(int row,int col,int rows) {
    return (col/64)*rows*128+row*128+((((col%64)/8)^(row%8))*16)+(col%8)*2;
  }
  __device__ static __forceinline__ void LoadX4(unsigned (&r)[4],unsigned address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(address));
  }
  __device__ static __forceinline__ void LoadX4T(unsigned (&r)[4],unsigned address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(address));
  }
  __device__ static __forceinline__ void Mma(float& c0,float& c1,float& c2,float& c3,
      unsigned const (&a)[4],unsigned b0,unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
  }

  __device__ static __forceinline__ void Run(Sm90ForwardEncoderAttentionOperands const& p,
      char* shared,Sm90DenseRingState& state) {
    auto* kv=reinterpret_cast<Element*>(shared+1024);
    auto* queries=kv+Slots*2*Seq*D;
    auto* scores=reinterpret_cast<float*>(shared+kLayout.scratch);
    int tasks=kTasks/int(gridDim.x)+(int(blockIdx.x)<kTasks%int(gridDim.x));
    unsigned phase=0;
    #pragma unroll
    for(int i=0;i<Slots;++i)phase|=(threadIdx.x<128?
        state.consumer_generations[i]:state.producer_generations[i])<<i;
    if(threadIdx.x>=128) {
      Sm90DenseRoles::Producer();
      if(threadIdx.x==128) {
        Async::ProxyAsyncGlobalFence();
        for(int local=0;local<tasks;++local) {
          int task=int(blockIdx.x)+local*int(gridDim.x);
          int head=task/(Seq/Rows),block=task%(Seq/Rows),slot=local%Slots;
          Async::Wait(state.empty+slot,(phase>>slot)&1);
          Async::ExpectTx(state.full+slot,2*kKvBytes+kQueryBytes);
          auto* destination=kv+slot*2*Seq*D;
          #pragma unroll
          for(int panel=0;panel<Seq/64;++panel) {
            Async::Tensor3D(destination+panel*64*D,p.key,0,head,panel*64,state.full+slot);
            Async::Tensor3D(destination+Seq*D+panel*64*D,p.value,0,head,panel*64,state.full+slot);
          }
          Async::Tensor3D(queries+slot*Rows*D,p.query,0,head,block*Rows,state.full+slot);
          phase^=1u<<slot;
        }
        if(tasks)for(int slot=0;slot<Slots;++slot)state.producer_generations[slot]=(phase>>slot)&1;
      }
      __syncwarp();
      __syncthreads();
      return;
    }
    Sm90DenseRoles::Consumer();
    int warp=int(threadIdx.x)>>5,lane=int(threadIdx.x)&31;
    for(int local=0;local<tasks;++local) {
      int task=int(blockIdx.x)+local*int(gridDim.x);
      int head=task/(Seq/Rows),block=task%(Seq/Rows),slot=local%Slots;
      Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
      auto const* keys=kv+slot*2*Seq*D;
      auto const* values=keys+Seq*D;
      unsigned q=Async::Shared(queries+slot*Rows*D);
      // S = Q K^T with mma.sync m16n8k16 over the whole head dimension: warp
      // w owns the 16-key groups of [w Seq/4, (w+1) Seq/4). The operand
      // addressing is the native dense attention's (Sm90DenseAttentionBody).
      for(int first=warp*(Seq/4);first<(warp+1)*(Seq/4);first+=16) {
        float sc[8]={0,0,0,0,0,0,0,0};
        #pragma unroll
        for(int k=0;k<D/16;++k) {
          unsigned a[4],b[4];
          LoadX4(a,q+Swizzled(lane%16,k*16+(lane/16)*8,Rows));
          int j=lane/8,row=first+(j/2)*8+lane%8,col=k*16+(j%2)*8;
          LoadX4(b,Async::Shared(keys)+Swizzled(row,col,Seq));
          Mma(sc[0],sc[1],sc[2],sc[3],a,b[0],b[1]);
          Mma(sc[4],sc[5],sc[6],sc[7],a,b[2],b[3]);
        }
        // C fragment of m16n8k16: thread t holds rows t/4 and t/4 + 8 at the
        // columns 2 (t % 4) and 2 (t % 4) + 1 of its own 8-key tile.
        #pragma unroll
        for(int e=0;e<8;++e) {
          int row=(lane/4)+8*((e%4)/2),col=first+(e/4)*8+2*(lane%4)+(e&1);
          if(row<Rows && col<Seq)scores[row*Seq+col]=sc[e]*p.scale;
        }
      }
      codegen::executor::ComputeSync();
      // Exact row softmax over the whole sequence, then O = P V.
      for(int row=int(threadIdx.x);row<Rows;row+=128) {
        float maximum=-INFINITY;
        for(int col=0;col<Seq;++col)maximum=fmaxf(maximum,scores[row*Seq+col]);
        float total=0;
        for(int col=0;col<Seq;++col) {
          float value=__expf(scores[row*Seq+col]-maximum);scores[row*Seq+col]=value;total+=value;
        }
        float inverse=total>0?1.0f/total:0.0f;
        for(int col=0;col<Seq;++col)scores[row*Seq+col]*=inverse;
      }
      codegen::executor::ComputeSync();
      // O = P V reads the swizzled V tile directly: the product is 16 x D per
      // task, so its FMAs are a small share of the body's work.
      auto const* value_base=reinterpret_cast<char const*>(values);
      for(int i=int(threadIdx.x);i<Rows*D;i+=128) {
        int row=i/D,channel=i%D;
        float sum=0;
        for(int col=0;col<Seq;++col)
          sum+=scores[row*Seq+col]*
              float(*reinterpret_cast<Element const*>(value_base+Swizzled(col,channel,Seq)));
        p.output[(std::size_t(block)*Rows+row)*p.heads*D+head*D+channel]=Element(sum);
      }
      codegen::executor::ComputeSync();
      Async::FenceProxyAsyncShared();
      Async::Arrive(state.empty+slot);
    }
    if(threadIdx.x==0 && tasks)
      for(int slot=0;slot<Slots;++slot)state.consumer_generations[slot]=(phase>>slot)&1;
    __syncthreads();
  }
};

}  // namespace tilemega::backend
