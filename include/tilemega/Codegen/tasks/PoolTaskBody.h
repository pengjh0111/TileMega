// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Target/ArchDispatch.h>
#include <cutlass/bfloat16.h>
#include <cmath>

namespace tilemega::codegen {
struct PoolOperands {
  cutlass::bfloat16_t const* input=nullptr;
  cutlass::bfloat16_t* output=nullptr;
  ConvDesc window{};
  DmBufferLayout input_layout{},output_layout{};
};

template<class Arch,int RowsPerTask=16,int ChannelsPerTask=64>
struct PoolTaskBody {
  static_assert(arch::Caps<Arch>::kBf16TensorCore && RowsPerTask>0);
  static_assert(ChannelsPerTask>=32 && ChannelsPerTask<=256 && ChannelsPerTask%32==0);
  static constexpr int kThreads=128,kSharedBytes=0;
  __host__ __device__ static unsigned Count(PoolOperands const& p) {
    auto const& w=p.window;
    return ((w.n*w.p*w.q+RowsPerTask-1)/RowsPerTask)*
        ((w.c+ChannelsPerTask-1)/ChannelsPerTask);
  }
  __device__ static void Run(PoolOperands const& p,unsigned task) {
    using namespace executor;
    auto const& w=p.window;auto const& in=p.input_layout;auto const& out=p.output_layout;
    unsigned channel_blocks=(w.c+ChannelsPerTask-1)/ChannelsPerTask;
    unsigned first_row=(task/channel_blocks)*RowsPerTask;
    unsigned first_channel=(task%channel_blocks)*ChannelsPerTask;
    // Four adjacent channels per lane keep NHWC windows coalesced. Spatial
    // bounds are semantic: padding never participates, even if halo is dirty.
    for(unsigned vector=ComputeThread();vector<RowsPerTask*ChannelsPerTask/4;vector+=kThreads) {
      unsigned row=first_row+vector/(ChannelsPerTask/4);
      unsigned column=first_channel+(vector%(ChannelsPerTask/4))*4;
      if(row>=w.n*w.p*w.q || column>=w.c)continue;
      unsigned image=row/(w.p*w.q),pixel=row%(w.p*w.q);
      float values[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};
      for(unsigned r=0;r<w.r;++r)for(unsigned s=0;s<w.s;++s) {
        int y=int(pixel/w.q*w.stride_h+r*w.dilation_h)-int(w.pad_h);
        int x=int(pixel%w.q*w.stride_w+s*w.dilation_w)-int(w.pad_w);
        if(y<0 || y>=int(w.h) || x<0 || x>=int(w.w))continue;
        auto base=image*in.strides[0]+(y+in.halo_top)*in.strides[1]+
            (x+in.halo_left)*in.strides[2]+column;
        #pragma unroll
        for(unsigned lane=0;lane<4;++lane)if(column+lane<w.c)
          values[lane]=fmaxf(values[lane],float(p.input[base+lane]));
      }
      auto base=image*out.strides[0]+(pixel/w.q+out.halo_top)*out.strides[1]+
          (pixel%w.q+out.halo_left)*out.strides[2]+column;
      #pragma unroll
      for(unsigned lane=0;lane<4;++lane)if(column+lane<w.c)
        p.output[base+lane]=cutlass::bfloat16_t(values[lane]);
    }
  }
};
} // namespace tilemega::codegen
