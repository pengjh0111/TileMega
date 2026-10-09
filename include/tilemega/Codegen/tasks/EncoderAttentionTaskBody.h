// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ServingAttentionWarp.h>
#include <tilemega/Codegen/executor/Async.cuh>
#include <cstdint>
#include <cmath>

namespace tilemega::codegen {
struct EncoderAttentionOperands {
  // [batch, token, head, Q/K/V, 64], with each head's triple contiguous.
  cutlass::bfloat16_t const* qkv=nullptr;
  cutlass::bfloat16_t* context=nullptr;
  std::int64_t const* key_padding=nullptr;
  unsigned batch=0,heads=0;
};

template<class Arch,int Sequence,int QueryRows=64,bool Masked=false>
struct EncoderAttentionTaskBody {
  static_assert(Sequence==128 || Sequence==384 || Sequence==512);
  static_assert(QueryRows==64 || QueryRows==128);
  using E=cutlass::bfloat16_t;
  using QK=backend::ServingAttentionWarp<Arch,16,64>;
  using PV=backend::ServingAttentionWarp<Arch,64,16,true>;
  using Async=executor::Async<Arch>;
  static constexpr int kThreads=128,kHeadDim=64,kKvTile=64;
  struct SharedStorage {
    alignas(16) E query[16*64];
    union {
      struct {alignas(16) E key[2][64*64],value[2][64*64];} pipeline;
      alignas(16) float partial[4*16*64];
    } storage;
    float maxima[4][16],sums[4][16],row_max[16],row_sum[16],alpha[16];
  };
  static constexpr int kSharedBytes=sizeof(SharedStorage);
  static_assert(kSharedBytes==35520);
  __host__ __device__ static unsigned Count(EncoderAttentionOperands const& p) {
    return p.batch*p.heads*((Sequence+QueryRows-1)/QueryRows);
  }
  __device__ static void Query(EncoderAttentionOperands const& p,SharedStorage& shared,
                               unsigned image,unsigned head,unsigned begin) {
    for(unsigned vector=executor::ComputeThread();vector<16*64/8;vector+=kThreads) {
      unsigned row=vector/8,column=vector%8*8;
      auto* source=p.qkv+((std::size_t(image)*Sequence+begin+row)*p.heads+head)*3*64+column;
      Async::Copy16Bytes(shared.query+typename QK::LayoutA{}(row,column),source,16);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  __device__ static void LoadKV(EncoderAttentionOperands const& p,SharedStorage& shared,
      unsigned slot,unsigned image,unsigned head,unsigned begin) {
    for(unsigned vector=executor::ComputeThread();vector<64*64/8;vector+=kThreads) {
      unsigned row=vector/8,column=vector%8*8;
      auto* source=p.qkv+((std::size_t(image)*Sequence+begin+row)*p.heads+head)*3*64+64+column;
      auto* key=shared.storage.pipeline.key[slot]+(row/16)*16*64+typename QK::LayoutB{}(row%16,column);
      auto* value=shared.storage.pipeline.value[slot]+(row/16)*16*64+typename PV::LayoutB{}(column,row%16);
      Async::Copy16Bytes(key,source,16);Async::Copy16Bytes(value,source+64,16);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  __device__ static void Run(EncoderAttentionOperands const& p,unsigned task,SharedStorage& shared) {
    using namespace cute;
    using namespace executor;
    if(!p.qkv || !p.context || !p.batch || !p.heads || (Masked && !p.key_padding)) {
      asm volatile("trap;");return;
    }
    constexpr unsigned query_blocks=(Sequence+QueryRows-1)/QueryRows;
    unsigned query_begin=(task%query_blocks)*QueryRows;
    unsigned head=(task/query_blocks)%p.heads,image=task/(query_blocks*p.heads);
    if(image>=p.batch)return;
    if constexpr(Masked) {
      if(ComputeThread()==0) {
        unsigned count=0;
        for(unsigned token=0;token<Sequence;++token)count+=p.key_padding[image*Sequence+token]!=0;
        shared.row_sum[0]=float(count);
      }
      ComputeSync();bool valid=shared.row_sum[0]>0;ComputeSync();
      if(!valid) {asm volatile("trap;");return;}
    }
    int warp=ComputeThread()/32,lane=ComputeThread()%32;
    auto score_coords=typename QK::Mma{}.get_slice(lane).partition_C(make_identity_tensor(Shape<_16,_16>{}));
    auto output_coords=typename PV::Mma{}.get_slice(lane).partition_C(make_identity_tensor(Shape<_16,_64>{}));
    for(unsigned query=query_begin;query<min(query_begin+QueryRows,unsigned(Sequence));query+=16) {
      Query(p,shared,image,head,query);
      if(ComputeThread()<16) {shared.row_max[ComputeThread()]=-INFINITY;shared.row_sum[ComputeThread()]=0;}
      auto output=PV::Accumulator();LoadKV(p,shared,0,image,head,0);
      for(unsigned begin=0,step=0;begin<Sequence;begin+=64,++step) {
        unsigned slot=step%2;
        asm volatile("cp.async.wait_group 0;" ::: "memory");ComputeSync();
        if(begin+64<Sequence)LoadKV(p,shared,slot^1,image,head,begin+64);
        auto score=QK::Accumulator();
        QK::QK(shared.query,shared.storage.pipeline.key[slot]+warp*16*64,score);
        float maximum[2]={-INFINITY,-INFINITY};
        #pragma unroll
        for(int i=0;i<size(score);++i) {
          int row=get<0>(score_coords(i)),column=warp*16+get<1>(score_coords(i));
          bool valid=true;
          if constexpr(Masked)valid=p.key_padding[image*Sequence+begin+column]!=0;
          score(i)=valid?score(i)*(1.4426950408889634f/8):-INFINITY;
          maximum[row/8]=fmaxf(maximum[row/8],score(i));
        }
        #pragma unroll
        for(int r=0;r<2;++r) {
          maximum[r]=fmaxf(maximum[r],__shfl_xor_sync(0xffffffff,maximum[r],1));
          maximum[r]=fmaxf(maximum[r],__shfl_xor_sync(0xffffffff,maximum[r],2));
          if((lane&3)==0)shared.maxima[warp][lane/4+8*r]=maximum[r];
        }
        ComputeSync();
        if(ComputeThread()<16) {
          int row=ComputeThread();float maximum=shared.row_max[row];
          for(unsigned w=0;w<4;++w)maximum=fmaxf(maximum,shared.maxima[w][row]);
          shared.alpha[row]=isfinite(shared.row_max[row])?exp2f(shared.row_max[row]-maximum):0;
          shared.row_max[row]=maximum;
        }
        ComputeSync();float sums[2]={};
        #pragma unroll
        for(int i=0;i<size(score);++i) {
          int row=get<0>(score_coords(i));
          score(i)=isfinite(score(i))?exp2f(score(i)-shared.row_max[row]):0;
          sums[row/8]+=score(i);
        }
        #pragma unroll
        for(int r=0;r<2;++r) {
          sums[r]+=__shfl_xor_sync(0xffffffff,sums[r],1);
          sums[r]+=__shfl_xor_sync(0xffffffff,sums[r],2);
          if((lane&3)==0)shared.sums[warp][lane/4+8*r]=sums[r];
        }
        ComputeSync();
        if(ComputeThread()<16) {
          int row=ComputeThread();float sum=shared.row_sum[row]*shared.alpha[row];
          for(unsigned w=0;w<4;++w)sum+=shared.sums[w][row];
          shared.row_sum[row]=sum;
        }
        #pragma unroll
        for(int i=0;i<size(output);++i)output(i)*=shared.alpha[int(get<0>(output_coords(i)))];
        // Probabilities round to BF16 at the PV MMA input. Maxima, running
        // sums, rescaling and output accumulation remain FP32 until store.
        PV::PV(score,score_coords,shared.storage.pipeline.value[slot]+warp*16*64,output);
        ComputeSync();
      }
      asm volatile("cp.async.wait_group 0;" ::: "memory");ComputeSync();
      #pragma unroll
      for(int i=0;i<size(output);++i)
        shared.storage.partial[(warp*16+int(get<0>(output_coords(i))))*64+int(get<1>(output_coords(i)))]=output(i);
      ComputeSync();
      for(unsigned vector=ComputeThread();vector<16*64/8;vector+=kThreads) {
        unsigned row=vector/8,column=vector%8*8;
        alignas(16) E stored[8];
        #pragma unroll
        for(unsigned d=0;d<8;++d) {
          float sum=0;
          for(unsigned w=0;w<4;++w)sum+=shared.storage.partial[(w*16+row)*64+column+d];
          stored[d]=E(sum/shared.row_sum[row]);
        }
        auto* destination=p.context+((std::size_t(image)*Sequence+query+row)*p.heads+head)*64+column;
        *reinterpret_cast<uint4*>(destination)=*reinterpret_cast<uint4 const*>(stored);
      }
      ComputeSync();
    }
  }
};
} // namespace tilemega::codegen
