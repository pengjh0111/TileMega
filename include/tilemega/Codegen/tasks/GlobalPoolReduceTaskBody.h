// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Target/ArchDispatch.h>
#include <cutlass/bfloat16.h>

namespace tilemega::codegen {
struct GlobalPoolReduceOperands {
  float const* partials=nullptr;
  float* output=nullptr;
  // [image, global producer M-tile, channel]. Entries outside an image's
  // tile interval are unwritten and must never be read by the reducer.
  DmBufferLayout partial_layout{};
  unsigned images=0,channels=0,image_rows=0,producer_tile_rows=0;
  unsigned output_stride=0;
  // Strip producers number their partials independently within each image.
  // Zero retains the global GEMM M-tile indexing convention.
  unsigned partial_rows_per_image=0;
  cutlass::bfloat16_t* rounded_output=nullptr;
};

template<class Arch,int ChannelsPerTask=128>
struct GlobalPoolReduceTaskBody {
  static_assert(arch::Caps<Arch>::kBf16TensorCore);
  static_assert(ChannelsPerTask>=32 && ChannelsPerTask<=256 && ChannelsPerTask%32==0);
  static constexpr int kThreads=128,kSharedBytes=0;
  __host__ __device__ static unsigned Count(GlobalPoolReduceOperands const& p) {
    return p.images*((p.channels+ChannelsPerTask-1)/ChannelsPerTask);
  }
  __device__ static void Run(GlobalPoolReduceOperands const& p,unsigned task) {
    using namespace executor;
    unsigned blocks=(p.channels+ChannelsPerTask-1)/ChannelsPerTask;
    if(!blocks || !p.images || !p.image_rows || !p.partials || !p.output ||
       (!p.partial_rows_per_image && !p.producer_tile_rows)) {asm volatile("trap;");return;}
    unsigned image=task/blocks,first_channel=(task%blocks)*ChannelsPerTask;
    unsigned first=p.partial_rows_per_image?0:image*p.image_rows/p.producer_tile_rows;
    unsigned end=p.partial_rows_per_image?p.partial_rows_per_image:
        ((image+1)*p.image_rows+p.producer_tile_rows-1)/p.producer_tile_rows;
    auto const& l=p.partial_layout;
    for(unsigned c=ComputeThread();c<ChannelsPerTask;c+=kThreads) {
      unsigned channel=first_channel+c;
      if(image>=p.images || channel>=p.channels)continue;
      float sum=0;
      // A fixed producer-tile order is shared by standalone and LA execution.
      for(unsigned part=first;part<end;++part)
        sum+=p.partials[image*l.strides[0]+part*l.strides[1]+channel*l.strides[2]];
      auto offset=image*(p.output_stride?p.output_stride:p.channels)+channel;
      float mean=sum/p.image_rows;p.output[offset]=mean;
      if(p.rounded_output)p.rounded_output[offset]=cutlass::bfloat16_t(mean);
    }
  }
};
} // namespace tilemega::codegen
