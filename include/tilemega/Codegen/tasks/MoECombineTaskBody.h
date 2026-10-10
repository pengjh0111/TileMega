// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Target/ArchDispatch.h>
#include <cutlass/bfloat16.h>
#include <cstdint>

namespace tilemega::codegen {
struct MoeCombineOperands {
  cutlass::bfloat16_t const* partials=nullptr;  // [token, rank, channel]
  cutlass::bfloat16_t const* weights=nullptr;   // [token, rank]
  cutlass::bfloat16_t const* residual=nullptr;  // [token, channel]
  cutlass::bfloat16_t* output=nullptr;
  float* row_stats=nullptr;                    // [token, channel tile, 2]
  unsigned tokens=0,channels=0;
  std::uint64_t token_stride=0,rank_stride=0,row_stride=0,stats_stride=0;
};

template<class Arch,int TopK=8,int TokenTile=1,int ChannelTile=128>
struct MoECombineTaskBody {
  static_assert(arch::Caps<Arch>::kBf16TensorCore);
  static_assert(TopK>0 && TopK<=32 && TokenTile>0 && ChannelTile>0 && ChannelTile%16==0);
  static constexpr int kThreads=128,kSharedBytes=TokenTile*ChannelTile*sizeof(float);
  struct SharedStorage {float output[TokenTile*ChannelTile];};
  __host__ __device__ static unsigned Count(MoeCombineOperands const& p) {
    return ((p.tokens+TokenTile-1)/TokenTile)*((p.channels+ChannelTile-1)/ChannelTile);
  }
  __device__ static void Run(MoeCombineOperands const& p,unsigned task,SharedStorage& s) {
    if(!p.partials || !p.weights || !p.residual || !p.output || !p.channels ||
       !p.token_stride || !p.rank_stride || !p.row_stride || (p.row_stats && !p.stats_stride)) {
      asm volatile("trap;");return;
    }
    unsigned tiles=(p.channels+ChannelTile-1)/ChannelTile;
    unsigned first_token=(task/tiles)*TokenTile,first_channel=(task%tiles)*ChannelTile;
    for(unsigned i=executor::ComputeThread();i<TokenTile*ChannelTile;i+=kThreads) {
      unsigned token=first_token+i/ChannelTile,channel=first_channel+i%ChannelTile;
      float result=0;
      if(token<p.tokens && channel<p.channels) {
        float sum=0;
        #pragma unroll
        for(unsigned rank=0;rank<TopK;++rank) {
          float weight=float(p.weights[token*TopK+rank]);
          float partial=float(p.partials[token*p.token_stride+rank*p.rank_stride+channel]);
          // HF grouped_mm rounds each weighted expert row to BF16, then
          // accumulates the K rows in FP32 and rounds once before residual.
          float product=float(cutlass::bfloat16_t(weight*partial));
          sum=__fadd_rn(sum,product);
        }
        auto moe=cutlass::bfloat16_t(sum);
        auto output=cutlass::bfloat16_t(__fadd_rn(float(moe),float(p.residual[token*p.row_stride+channel])));
        p.output[token*p.row_stride+channel]=output;result=float(output);
      }
      s.output[i]=result;
    }
    executor::ComputeSync();
    if(p.row_stats) {
      unsigned lane=executor::ComputeThread()%32,warp=executor::ComputeThread()/32;
      for(unsigned row=warp;row<TokenTile;row+=4) {
        if(first_token+row>=p.tokens)continue;
        float sum=0,square=0;
        for(unsigned column=lane;column<ChannelTile;column+=32) {
          float value=s.output[row*ChannelTile+column];sum+=value;square+=value*value;
        }
        for(unsigned shift=16;shift;shift/=2) {
          sum+=__shfl_xor_sync(0xffffffff,sum,shift);
          square+=__shfl_xor_sync(0xffffffff,square,shift);
        }
        if(!lane) {
          auto offset=(first_token+row)*p.stats_stride+(task%tiles)*2;
          p.row_stats[offset]=sum;p.row_stats[offset+1]=square;
        }
      }
    }
    executor::ComputeSync();
  }
};
} // namespace tilemega::codegen
