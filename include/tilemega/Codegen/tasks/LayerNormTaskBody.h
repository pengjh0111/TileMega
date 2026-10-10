// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/DmTensorAddress.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Target/ArchDispatch.h>
#include <cutlass/bfloat16.h>
#include <cmath>

namespace tilemega::codegen {
struct LayerNormOperands {
  cutlass::bfloat16_t const* input=nullptr;
  cutlass::bfloat16_t const* gamma=nullptr;
  cutlass::bfloat16_t const* beta=nullptr;
  cutlass::bfloat16_t* output=nullptr;
  DmBufferLayout input_layout{},output_layout{};
  std::uint64_t rows=0;
  float epsilon=0;
  // One complete pair per output row, after the BF16 output rounding.
  float* row_stats=nullptr;
};

// Each warp retains one row in registers. Centered variance avoids subtracting
// nearly equal second moments; the affine expression rounds only at the store.
template<class Arch,int Width,int RowsPerTask=4>
struct LayerNormTaskBody {
  static_assert(arch::Caps<Arch>::kBf16TensorCore && Width>0 && Width<=4096);
  static_assert(RowsPerTask>0 && RowsPerTask%4==0);
  static constexpr int kThreads=128,kSharedBytes=0;
  __device__ static void Run(LayerNormOperands const& p,std::uint64_t task) {
    using namespace executor;
    if(!p.input || !p.gamma || !p.beta || !p.output || !(p.epsilon>0)) {
      asm volatile("trap;");return;
    }
    int lane=ComputeThread()%32,warp=ComputeThread()/32;
    for(auto row=task*RowsPerTask+warp;
        row<p.rows && row<(task+1)*RowsPerTask;row+=4) {
      float values[(Width+31)/32],sum=0;
      #pragma unroll
      for(int i=0;i<(Width+31)/32;++i) {
        int column=lane+i*32;
        values[i]=column<Width?float(p.input[
            backend::DmTensorAddress<DmWriteKind::kDense>::Offset(
                p.input_layout,row,column,Width)]):0;
        sum+=values[i];
      }
      for(int shift=16;shift;shift/=2)sum+=__shfl_xor_sync(0xffffffff,sum,shift);
      float mean=sum/Width,variance=0;
      #pragma unroll
      for(int i=0;i<(Width+31)/32;++i)if(lane+i*32<Width) {
        float centered=values[i]-mean;variance+=centered*centered;
      }
      for(int shift=16;shift;shift/=2)variance+=__shfl_xor_sync(0xffffffff,variance,shift);
      float inverse=rsqrtf(variance/Width+p.epsilon),out_sum=0,out_square=0;
      #pragma unroll
      for(int i=0;i<(Width+31)/32;++i) {
        int column=lane+i*32;
        if(column>=Width)continue;
        float value=((values[i]-mean)*inverse)*float(p.gamma[column])+float(p.beta[column]);
        cutlass::bfloat16_t rounded(value);
        p.output[backend::DmTensorAddress<DmWriteKind::kDense>::Offset(
            p.output_layout,row,column,Width)]=rounded;
        out_sum+=float(rounded);out_square+=float(rounded)*float(rounded);
      }
      if(p.row_stats) {
        for(int shift=16;shift;shift/=2) {
          out_sum+=__shfl_xor_sync(0xffffffff,out_sum,shift);
          out_square+=__shfl_xor_sync(0xffffffff,out_square,shift);
        }
        if(lane==0) {p.row_stats[2*row]=out_sum;p.row_stats[2*row+1]=out_square;}
      }
    }
  }
};
} // namespace tilemega::codegen
