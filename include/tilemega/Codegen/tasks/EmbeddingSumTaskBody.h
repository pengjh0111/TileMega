// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Maca/DevicePrimitives.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Target/ArchDispatch.h>
#include <tilemega/Target/Maca/Bf16.h>
#include <cstdint>

namespace tilemega::codegen {
struct EmbeddingSumOperands {
  std::int64_t const* input_ids=nullptr;
  std::int64_t const* token_type_ids=nullptr;
  tilemega::Bf16 const* word=nullptr;
  tilemega::Bf16 const* token_type=nullptr;
  tilemega::Bf16 const* position=nullptr;
  tilemega::Bf16* output=nullptr;
  float* row_stats=nullptr;
  std::uint64_t rows=0;
  std::uint32_t sequence=0,vocabulary=0,token_types=0,positions=0;
};

// BERT adds word + token type, rounds to BF16, then adds position and rounds
// again. Statistics consume that stored value, rather than a fused FP32 sum.
template<class Arch,int Width>
struct EmbeddingSumTaskBody {
  static_assert(arch::Caps<Arch>::kBf16TensorCore && Width>0);
  static constexpr int kThreads=128,kSharedBytes=8*sizeof(float);
  __device__ static void Run(EmbeddingSumOperands const& p,std::uint64_t row,
                             float* shared) {
    using namespace executor;
    if(row>=p.rows)return;
    if(!p.input_ids || !p.token_type_ids || !p.word || !p.token_type ||
       !p.position || !p.output || !p.sequence || p.sequence>p.positions) {
      maca::Trap();return;
    }
    auto word=p.input_ids[row],type=p.token_type_ids[row];
    if(word<0 || std::uint64_t(word)>=p.vocabulary || type<0 ||
       std::uint64_t(type)>=p.token_types) {maca::Trap();return;}
    auto position=row%p.sequence;
    float sum=0,square=0;
    for(int column=ComputeThread();column<Width;column+=kThreads) {
      tilemega::Bf16 first(float(p.word[word*Width+column])+
          float(p.token_type[type*Width+column]));
      tilemega::Bf16 value(float(first)+float(p.position[position*Width+column]));
      p.output[row*Width+column]=value;
      sum+=float(value);square+=float(value)*float(value);
    }
    if(p.row_stats) {
      int lane=ComputeThread()%32,warp=ComputeThread()/32;
      for(int shift=16;shift;shift/=2) {
        sum+=__shfl_xor_sync(0xffffffff,sum,shift);
        square+=__shfl_xor_sync(0xffffffff,square,shift);
      }
      if(lane==0) {shared[warp]=sum;shared[warp+4]=square;}
      ComputeSync();
      if(ComputeThread()==0) {
        float total=0,total_square=0;
        for(int i=0;i<4;++i) {total+=shared[i];total_square+=shared[i+4];}
        p.row_stats[2*row]=total;p.row_stats[2*row+1]=total_square;
      }
      // A worker can immediately reuse this same scratch for its next row.
      ComputeSync();
    }
  }
};
} // namespace tilemega::codegen
