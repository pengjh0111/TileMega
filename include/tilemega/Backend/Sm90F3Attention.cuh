// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90F3Async.cuh>
#include <tilemega/Backend/Sm90F3Ring.cuh>
#include <cutlass/bfloat16.h>
#include <cmath>
#include <cuda.h>

namespace tilemega::backend {

struct Sm90F3AttentionOperands {
  CUtensorMap const* query;
  CUtensorMap const* key;
  CUtensorMap const* value;
  cutlass::bfloat16_t* output;
};

template<int Batch,int Seq=128,int D=64,int Heads=12>
struct Sm90F3Attention {
  using Element=cutlass::bfloat16_t;
  using Async=Sm90F3Async;
  using Base=Sm90F3Async::Base;
  static_assert(Batch>=1 && Seq==128 && D==64 && Heads==12);
  static constexpr int kSlots=2,kTasks=Batch*Heads;
  static constexpr int kSlotElements=3*Seq*D,kSlotBytes=kSlotElements*2;
  static constexpr int kSharedBytes=kSlots*kSlotBytes,kLayoutBytes=kSharedBytes;
  static constexpr int kSideOffset=0;
  static constexpr bool kCanPreissue=false;

  __device__ static __forceinline__ int LocalTasks() {
    return kTasks/int(gridDim.x)+(int(blockIdx.x)<kTasks%int(gridDim.x));
  }
  __device__ static __forceinline__ unsigned Swizzled(int row,int col) {
    return row*128+((((col/8)^(row%8))*16))+(col%8)*2;
  }
  __device__ static __forceinline__ unsigned Pack(float a,float b) {
    return unsigned(Element(a).raw())|(unsigned(Element(b).raw())<<16);
  }

  __device__ static __noinline__ void Produce(Sm90F3AttentionOperands const& p,
      Sm90F3Ring& ring,char* arena,Sm90F3Phase& phase) {
    if(threadIdx.x==256) {
      Base::ProxyAsyncGlobalFence();
      for(int local=0;local<LocalTasks();++local) {
        int task=int(blockIdx.x)+local*int(gridDim.x),slot=local%kSlots;
        int image=task/Heads,head=task%Heads;
        Base::Wait(ring.empty+slot,phase.Bit(phase.empty,slot));
        phase.empty^=1u<<slot;
        Base::ExpectTx(ring.full+slot,kSlotBytes);
        char* panel=arena+slot*kSlotBytes;
        Base::Tensor3D(panel,p.query,0,head,image*Seq,ring.full+slot);
        Base::Tensor3D(panel+2*Seq*D,p.key,0,head,image*Seq,ring.full+slot);
        Base::Tensor3D(panel+4*Seq*D,p.value,0,head,image*Seq,ring.full+slot);
      }
    }
    __syncwarp();
  }

  // Immediate softmax packing keeps 121 registers, no spills in P-UNIT.
  // Outlining still spills at the device-call register ceiling.
  __device__ static __forceinline__ void Consume(Sm90F3AttentionOperands const& p,
      Sm90F3Ring& ring,char* arena,Sm90F3Phase& phase) {
    int warp=int(threadIdx.x)/32,lane=int(threadIdx.x)%32;
    for(int local=0;local<LocalTasks();++local) {
      int task=int(blockIdx.x)+local*int(gridDim.x),slot=local%kSlots;
      int image=task/Heads,head=task%Heads;
      Base::Wait(ring.full+slot,phase.Bit(phase.full,slot));
      phase.full^=1u<<slot;
      unsigned q=Base::Shared(arena+slot*kSlotBytes);
      unsigned key=q+2*Seq*D,value=key+2*Seq*D;
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
          v[e]=exp2f((scores[block][e]-maximum[e/2])*(.125f*1.4426950408889634f));
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
      #pragma unroll
      for(int block=0;block<8;++block) {
        #pragma unroll
        for(int e=0;e<4;++e) {
          int row=warp*16+lane/4+8*(e/2),col=block*8+2*(lane%4)+(e&1);
          p.output[(std::size_t(image)*Seq+row)*Heads*D+head*D+col]=
              Element(output[block][e]/sum[e/2]);
        }
      }
      __syncwarp();
      if(lane==0)Base::Arrive(ring.empty+slot);
    }
  }
};

}  // namespace tilemega::backend
