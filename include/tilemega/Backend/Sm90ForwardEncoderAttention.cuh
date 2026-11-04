// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90ForwardEpilogue.cuh>
#include <tilemega/Backend/Sm90ForwardMark.cuh>
#include <tilemega/Backend/Sm90DenseAttentionMath.cuh>
#include <tilemega/Backend/Sm90DenseLayout.h>
#include <tilemega/Backend/Sm90ForwardResources.h>
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <tilemega/Backend/Sm90DenseRoles.cuh>
#include <tilemega/Codegen/executor/Async.cuh>

namespace tilemega::backend {

struct Sm90ForwardEncoderAttentionOperands {
  // Q/K/V are {D, Heads, Batch*Seq} views of the packed projection rows.
  CUtensorMap const* key;
  CUtensorMap const* value;
  CUtensorMap const* query;
  cutlass::bfloat16_t* output;        // [Batch*Seq][Heads*D]
  float scale;
  int heads,sequence;
};

// ScalarReference is a compile-time test control, never selected by production
// codegen. Rows=16 preserves the original scalar softmax/PV arithmetic order.
template<int Batch,int Seq,int D,int Heads,int Rows,int Slots,bool ScalarReference=false>
struct Sm90ForwardEncoderAttention {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  using Math=Sm90DenseAttentionMath;
  static_assert(Batch>=1 && D==64 && Seq%64==0 && Seq%Rows==0);
  static_assert(Rows==16 || Rows==32 || Rows==64 || Rows==128);
  static_assert(Slots>=1 && Slots<=2);
  static_assert(!ScalarReference || Rows==16);
  static constexpr int kThreads=160,kPanels=D/64,kLanesPerRow=128/Rows;
  static constexpr int kTasks=Batch*Heads*(Seq/Rows);
  static constexpr int kKvBytes=2*Seq*D,kQueryBytes=2*Rows*D;
  static constexpr int kScoreBytes=4*Rows*Seq;
  static constexpr auto kLayout=Sm90ForwardAttentionLayout(Seq,Rows,Slots,ScalarReference);
  static constexpr int kScoreOffset=kLayout.scratch;
  // P has independent storage: no BF16 write may alias unread FP32 scores.
  static constexpr int kProbabilityOffset=Sm90ForwardAlign(kScoreOffset+kScoreBytes);
  static constexpr int kProbabilityBytes=ScalarReference?0:2*Rows*Seq;
  static constexpr int kSharedBytes=kLayout.end;
  static_assert(kScoreBytes>=4*Rows*D,"the PV output reuses scores after P is complete");

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
    auto* scores=reinterpret_cast<float*>(shared+kScoreOffset);
    auto* probability=reinterpret_cast<Element*>(shared+kProbabilityOffset);
    int tasks=kTasks/int(gridDim.x)+(int(blockIdx.x)<kTasks%int(gridDim.x));
    unsigned phase=0;
    // Only the issuing producer reads its phase. Other producer lanes do not
    // use it and could otherwise read while lane 0 publishes its next phase.
    if(threadIdx.x<128) {
      #pragma unroll
      for(int i=0;i<Slots;++i)phase|=state.consumer_generations[i]<<i;
    }else if(threadIdx.x==128) {
      #pragma unroll
      for(int i=0;i<Slots;++i)phase|=state.producer_generations[i]<<i;
    }
    if(threadIdx.x>=128) {
      Sm90DenseRoles::Producer();
      if(threadIdx.x==128) {
        Async::ProxyAsyncGlobalFence();
        for(int local=0;local<tasks;++local) {
          int task=int(blockIdx.x)+local*int(gridDim.x);
          int image=task/(Heads*(Seq/Rows));
          int head=(task/(Seq/Rows))%Heads,block=task%(Seq/Rows),slot=local%Slots;
          Async::Wait(state.empty+slot,(phase>>slot)&1);
          Async::ExpectTx(state.full+slot,2*kKvBytes+kQueryBytes);
          auto* destination=kv+slot*2*Seq*D;
          #pragma unroll
          for(int panel=0;panel<Seq/64;++panel) {
            Async::Tensor3D(destination+panel*64*D,p.key,0,head,image*Seq+panel*64,state.full+slot);
            Async::Tensor3D(destination+Seq*D+panel*64*D,p.value,0,head,image*Seq+panel*64,state.full+slot);
          }
          Async::Tensor3D(queries+slot*Rows*D,p.query,0,head,image*Seq+block*Rows,state.full+slot);
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
    TM_FWD_TRACE_ACC(wait_trace);TM_FWD_TRACE_ACC(qk_trace);
    TM_FWD_TRACE_ACC(softmax_trace);TM_FWD_TRACE_ACC(pv_trace);
    for(int local=0;local<tasks;++local) {
      int task=int(blockIdx.x)+local*int(gridDim.x);
      int image=task/(Heads*(Seq/Rows));
      int head=(task/(Seq/Rows))%Heads,block=task%(Seq/Rows),slot=local%Slots;
      TM_FWD_TRACE_BEGIN(wait_trace);
      Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
      TM_FWD_TRACE_END(wait_trace);
      if(local==0) {TM_FWD_TRACE_FIRST_READY();}
      TM_FWD_TRACE_BEGIN(qk_trace);
      auto const* keys=kv+slot*2*Seq*D;
      auto const* values=keys+Seq*D;
      unsigned q=Async::Shared(queries+slot*Rows*D);
      // Every warp retains its original key quarter; query tiles extend M.
      for(int query_base=0;query_base<Rows;query_base+=16)
        for(int first=warp*(Seq/4);first<(warp+1)*(Seq/4);first+=16) {
          float sc[8]={0,0,0,0,0,0,0,0};
          #pragma unroll
          for(int k=0;k<D/16;++k) {
            unsigned a[4],b[4];
            LoadX4(a,q+Swizzled(query_base+lane%16,k*16+(lane/16)*8,Rows));
            int j=lane/8,row=first+(j/2)*8+lane%8,col=k*16+(j%2)*8;
            LoadX4(b,Async::Shared(keys)+Swizzled(row,col,Seq));
            Mma(sc[0],sc[1],sc[2],sc[3],a,b[0],b[1]);
            Mma(sc[4],sc[5],sc[6],sc[7],a,b[2],b[3]);
          }
          #pragma unroll
          for(int e=0;e<8;++e) {
            int row=query_base+(lane/4)+8*((e%4)/2);
            int col=first+(e/4)*8+2*(lane%4)+(e&1);
            scores[row*Seq+col]=sc[e]*p.scale;
          }
        }
      codegen::executor::ComputeSync();
      TM_FWD_TRACE_END(qk_trace);
      TM_FWD_TRACE_BEGIN(softmax_trace);
      if constexpr(ScalarReference) {
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
      }else {
        int row=int(threadIdx.x)/kLanesPerRow,lane_in_row=int(threadIdx.x)%kLanesPerRow;
        float maximum=-INFINITY;
        for(int col=lane_in_row;col<Seq;col+=kLanesPerRow)
          maximum=fmaxf(maximum,scores[row*Seq+col]);
        #pragma unroll
        for(int delta=kLanesPerRow/2;delta;delta/=2)
          maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,delta,kLanesPerRow));
        float total=0;
        for(int col=lane_in_row;col<Seq;col+=kLanesPerRow) {
          float value=__expf(scores[row*Seq+col]-maximum);
          scores[row*Seq+col]=value;total+=value;
        }
        #pragma unroll
        for(int delta=kLanesPerRow/2;delta;delta/=2)
          total+=__shfl_xor_sync(0xffffffffu,total,delta,kLanesPerRow);
        float inverse=total>0?1.0f/total:0.0f;
        for(int col=lane_in_row;col<Seq;col+=kLanesPerRow)
          *reinterpret_cast<Element*>(reinterpret_cast<char*>(probability)+Swizzled(row,col,Rows))=
              Element(scores[row*Seq+col]*inverse);
      }
      codegen::executor::ComputeSync();
      TM_FWD_TRACE_END(softmax_trace);
      TM_FWD_TRACE_BEGIN(pv_trace);
      if constexpr(ScalarReference) {
        auto const* value_base=reinterpret_cast<char const*>(values);
        for(int i=int(threadIdx.x);i<Rows*D;i+=128) {
          int row=i/D,channel=i%D;
          float sum=0;
          for(int col=0;col<Seq;++col)
            sum+=scores[row*Seq+col]*
                float(*reinterpret_cast<Element const*>(value_base+Swizzled(col,channel,Seq)));
          p.output[(std::size_t(image)*Seq+block*Rows+row)*p.heads*D+head*D+channel]=Element(sum);
        }
      }else {
        // A warp owns m16 x n16 tiles. V is physical [Seq,D], so B loads
        // transpose each 8x8 subtile; the QK path reads physical [N,K].
        for(int tile=warp;tile<(Rows/16)*(D/16);tile+=4) {
          int m16=(tile/(D/16))*16,n16=(tile%(D/16))*16;
          float out[8]={0,0,0,0,0,0,0,0};
          #pragma unroll
          for(int k16=0;k16<Seq;k16+=16) {
            unsigned a[4],b[4];
            LoadX4(a,Async::Shared(probability)+Swizzled(m16+lane%16,k16+(lane/16)*8,Rows));
            int row=k16+((lane/8)%2)*8+lane%8,col=n16+((lane/8)/2)*8;
            LoadX4T(b,Async::Shared(values)+Swizzled(row,col,Seq));
            Mma(out[0],out[1],out[2],out[3],a,b[0],b[1]);
            Mma(out[4],out[5],out[6],out[7],a,b[2],b[3]);
          }
          #pragma unroll
          for(int e=0;e<8;++e) {
            int row=m16+(lane/4)+8*((e%4)/2);
            int col=n16+(e/4)*8+2*(lane%4)+(e&1);
            scores[row*D+col]=out[e];
          }
        }
        codegen::executor::ComputeSync();
        for(int i=int(threadIdx.x);i<Rows*D;i+=128) {
          int row=i/D,channel=i%D;
          p.output[(std::size_t(image)*Seq+block*Rows+row)*p.heads*D+head*D+channel]=Element(scores[i]);
        }
      }
      codegen::executor::ComputeSync();
      TM_FWD_TRACE_END(pv_trace);
      Async::FenceProxyAsyncShared();
      Async::Arrive(state.empty+slot);
    }
    TM_FWD_TRACE_FLUSH(wait_trace,kKvWait);TM_FWD_TRACE_FLUSH(qk_trace,kAttentionQk);
    TM_FWD_TRACE_FLUSH(softmax_trace,kAttentionSoftmax);TM_FWD_TRACE_FLUSH(pv_trace,kAttentionPv);
    if(threadIdx.x==0 && tasks)
      for(int slot=0;slot<Slots;++slot)state.consumer_generations[slot]=(phase>>slot)&1;
    __syncthreads();
  }
};

}  // namespace tilemega::backend
