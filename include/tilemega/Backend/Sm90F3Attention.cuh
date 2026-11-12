// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90F3Async.cuh>
#include <tilemega/Backend/Sm90F3Ring.cuh>
#include <tilemega/Backend/Sm90F3Mark.cuh>
#include <cutlass/bfloat16.h>
#include <cmath>
#include <cuda.h>

#ifndef TILEMEGA_F3_ATTENTION_VECTOR_STORE
#define TILEMEGA_F3_ATTENTION_VECTOR_STORE 0
#endif
#if TILEMEGA_F3_ATTENTION_VECTOR_STORE != 0 && TILEMEGA_F3_ATTENTION_VECTOR_STORE != 1
#error "TILEMEGA_F3_ATTENTION_VECTOR_STORE must be0 or1"
#endif

namespace tilemega::backend {

struct Sm90F3AttentionOperands {
  CUtensorMap const* query;
  CUtensorMap const* key;
  CUtensorMap const* value;
  cutlass::bfloat16_t* output;
};

template<int Batch,int Rows=(16*Batch<128?16*Batch:128)>
struct Sm90F3Attention {
  using Element=cutlass::bfloat16_t;
  using Async=Sm90F3Async;
  using Base=Sm90F3Async::Base;
  static constexpr int Seq=128,D=64,Heads=12,kBlocks=Seq/Rows;
  static_assert(Batch>=1 && (Rows==16 || Rows==32 || Rows==64 || Rows==128));
  static constexpr int kSlots=2,kTasks=Batch*Heads*kBlocks;
  static constexpr int kSlotElements=(Rows+2*Seq)*D,kSlotBytes=kSlotElements*2;
  static constexpr int kSharedBytes=kSlots*kSlotBytes,kLayoutBytes=kSharedBytes;
  static constexpr int kSideOffset=0;
  static constexpr bool kCanPreissue=false;

  __device__ static __forceinline__ int LocalTasks() {
    return kTasks/int(tilemega::backend::Sm90F3Async::GridSize())+(int(tilemega::backend::Sm90F3Async::BlockIndex())<kTasks%int(tilemega::backend::Sm90F3Async::GridSize()));
  }
  __device__ static __forceinline__ unsigned Swizzled(int row,int col) {
    return row*128+((((col/8)^(row%8))*16))+(col%8)*2;
  }
  __device__ static __forceinline__ unsigned Pack(float a,float b) {
    return unsigned(Element(a).raw())|(unsigned(Element(b).raw())<<16);
  }

  __device__ static __forceinline__ void Produce(Sm90F3AttentionOperands const& p,
      Sm90F3Ring& ring,char* arena,Sm90F3Phase& phase) {
    if(tilemega::backend::Sm90F3Async::ThreadIndex()==256) {
      Base::ProxyAsyncGlobalFence();
      for(int local=0;local<LocalTasks();++local) {
        int task=int(tilemega::backend::Sm90F3Async::BlockIndex())+local*int(tilemega::backend::Sm90F3Async::GridSize()),slot=local%kSlots;
        int image=task/(Heads*kBlocks),head=(task/kBlocks)%Heads;
        int query_row=(task%kBlocks)*Rows;
        Base::Wait(ring.empty+slot,phase.Bit(phase.empty,slot));
        phase.empty^=1u<<slot;
        Base::ExpectTx(ring.full+slot,kSlotBytes);
        char* panel=arena+slot*kSlotBytes;
        Base::Tensor3D(panel,p.query,0,head,image*Seq+query_row,ring.full+slot);
        Base::Tensor3D(panel+2*Rows*D,p.key,0,head,image*Seq,ring.full+slot);
        Base::Tensor3D(panel+2*(Rows+Seq)*D,p.value,0,head,image*Seq,ring.full+slot);
      }
    }
    __syncwarp();
  }

  // Immediate softmax packing keeps 121 registers, no spills in P-UNIT.
  // Outlining still spills at the device-call register ceiling.
  __device__ static __forceinline__ void Consume(Sm90F3AttentionOperands const& p,
      Sm90F3Ring& ring,char* arena,Sm90F3Phase& phase) {
    int warp=int(threadIdx.x)/32,lane=int(threadIdx.x)%32;
    for(int local=0,task=int(tilemega::backend::Sm90F3Async::BlockIndex());task<kTasks;++local,task+=int(tilemega::backend::Sm90F3Async::GridSize())) {
      int slot=local%kSlots;
      int image=task/(Heads*kBlocks),head=(task/kBlocks)%Heads;
      int query_row=(task%kBlocks)*Rows;
      TM_F3_TRACE_BEGIN(kSlotWait);
      Base::Wait(ring.full+slot,phase.Bit(phase.full,slot));
      TM_F3_TRACE_END(kSlotWait);
      TM_F3_TRACE_FIRST_READY();
      phase.full^=1u<<slot;
      if(warp>=Rows/16) {
        __syncwarp();
        if(lane==0)Base::Arrive(ring.empty+slot);
        continue;
      }
      unsigned q=Base::Shared(arena+slot*kSlotBytes);
      unsigned key=q+2*Rows*D,value=key+2*Seq*D;
      TM_F3_TRACE_BEGIN(kAttentionQk);
      float scores[16][4]{};
      #pragma unroll 1
      for(int k=0;k<4;++k) {
        unsigned query[4];
        Async::MatrixLoad4(query,q+Swizzled(warp*16+lane%16,k*16+(lane/16)*8));
        #pragma unroll
        for(int pair=0;pair<8;++pair) {
          unsigned b[4];int j=lane/8;
          Async::MatrixLoad4(b,key+Swizzled(pair*16+(j/2)*8+lane%8,k*16+(j%2)*8));
          Async::Mma16x8x16(scores[2*pair],query,b[0],b[1]);
          Async::Mma16x8x16(scores[2*pair+1],query,b[2],b[3]);
        }
      }
      TM_F3_TRACE_END(kAttentionQk);
      TM_F3_TRACE_BEGIN(kAttentionSoftmax);
      float maximum[2]={-INFINITY,-INFINITY},sum[2]={0.f,0.f};
      #pragma unroll
      for(int block=0;block<16;++block) {
        maximum[0]=fmaxf(maximum[0],fmaxf(scores[block][0],scores[block][1]));
        maximum[1]=fmaxf(maximum[1],fmaxf(scores[block][2],scores[block][3]));
      }
      #pragma unroll
      for(int row=0;row<2;++row) {
        maximum[row]=fmaxf(maximum[row],__shfl_xor_sync(0xffffffffu,maximum[row],1));
        maximum[row]=fmaxf(maximum[row],__shfl_xor_sync(0xffffffffu,maximum[row],2));
      }
      unsigned probability[8][4];
      #pragma unroll
      for(int block=0;block<16;++block) {
        float v[4];
        #pragma unroll
        for(int e=0;e<4;++e) {
          v[e]=Async::Exp2((scores[block][e]-maximum[e/2])*(.125f*1.4426950408889634f));
          sum[e/2]+=v[e];
        }
        probability[block/2][2*(block%2)]=Pack(v[0],v[1]);
        probability[block/2][2*(block%2)+1]=Pack(v[2],v[3]);
      }
      #pragma unroll
      for(int row=0;row<2;++row) {
        sum[row]+=__shfl_xor_sync(0xffffffffu,sum[row],1);
        sum[row]+=__shfl_xor_sync(0xffffffffu,sum[row],2);
      }
      TM_F3_TRACE_END(kAttentionSoftmax);
      TM_F3_TRACE_BEGIN(kAttentionPv);
      float output[8][4]{};
      #pragma unroll
      for(int pair=0;pair<4;++pair) {
        #pragma unroll
        for(int k=0;k<8;++k) {
          unsigned b[4];
          Async::MatrixLoad4Trans(b,value+Swizzled(k*16+((lane/8)%2)*8+lane%8,
                                 pair*16+((lane/8)/2)*8));
          Async::Mma16x8x16(output[2*pair],probability[k],b[0],b[1]);
          Async::Mma16x8x16(output[2*pair+1],probability[k],b[2],b[3]);
        }
      }
      TM_F3_TRACE_END(kAttentionPv);
      TM_F3_TRACE_BEGIN(kStore);
#if TILEMEGA_F3_ATTENTION_VECTOR_STORE
      #pragma unroll
      for(int group=0;group<2;++group) {
        #pragma unroll
        for(int half=0;half<2;++half) {
          unsigned words[4];
          #pragma unroll
          for(int b=0;b<4;++b) {
            int block=group*4+b;
            words[b]=Pack(Async::DivideNormalization(output[block][2*half],sum[half]),
                Async::DivideNormalization(output[block][2*half+1],sum[half]));
          }
          // A quad owns one row. Transpose its four column pairs so each
          // destination lane writes eight consecutive BF16 values.
          int quad=lane&3;
          unsigned x=__shfl_xor_sync(0xffffffffu,quad&1?words[0]:words[1],1);
          unsigned y=__shfl_xor_sync(0xffffffffu,quad&1?words[2]:words[3],1);
          if(quad&1) {words[0]=x;words[2]=y;} else {words[1]=x;words[3]=y;}
          x=__shfl_xor_sync(0xffffffffu,quad&2?words[0]:words[2],2);
          y=__shfl_xor_sync(0xffffffffu,quad&2?words[1]:words[3],2);
          if(quad&2) {words[0]=x;words[1]=y;} else {words[2]=x;words[3]=y;}
          int row=query_row+warp*16+lane/4+8*half,col=group*32+8*quad;
          Async::Store128(p.output+(std::size_t(image)*Seq+row)*Heads*D+head*D+col,
              words[0],words[1],words[2],words[3]);
        }
      }
#else
      #pragma unroll
      for(int block=0;block<8;++block) {
        #pragma unroll
        for(int e=0;e<4;e+=2) {
          int row=query_row+warp*16+lane/4+8*(e/2),col=block*8+2*(lane%4);
          unsigned packed=Pack(Async::DivideNormalization(output[block][e],sum[e/2]),
              Async::DivideNormalization(output[block][e+1],sum[e/2]));
          Async::Store32(p.output+(std::size_t(image)*Seq+row)*Heads*D+head*D+col,packed);
        }
      }
#endif
      TM_F3_TRACE_END(kStore);
      __syncwarp();
      if((tilemega::backend::Sm90F3Async::ThreadIndex()&31)==0)Base::Arrive(ring.empty+slot);
    }
  }
};

}  // namespace tilemega::backend
