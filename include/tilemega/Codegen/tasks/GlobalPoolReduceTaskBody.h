// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Target/ArchDispatch.h>

namespace tilemega::codegen {
struct GlobalPoolReduceOperands {
  float const* partials=nullptr;
  float* output=nullptr;
  // [image, global producer M-tile, channel]. Entries outside an image's
  // tile interval are unwritten and must never be read by the reducer.
  DmBufferLayout partial_layout{};
  unsigned images=0,channels=0,image_rows=0,producer_tile_rows=0;
  unsigned output_stride=0;
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
    unsigned image=task/blocks,first_channel=(task%blocks)*ChannelsPerTask;
    unsigned first=image*p.image_rows/p.producer_tile_rows;
    unsigned end=((image+1)*p.image_rows+p.producer_tile_rows-1)/p.producer_tile_rows;
    auto const& l=p.partial_layout;
    for(unsigned c=ComputeThread();c<ChannelsPerTask;c+=kThreads) {
      unsigned channel=first_channel+c;
      if(image>=p.images || channel>=p.channels)continue;
      float sum=0;
      // A fixed producer-tile order is shared by standalone and LA execution.
      for(unsigned part=first;part<end;++part)
        sum+=p.partials[image*l.strides[0]+part*l.strides[1]+channel*l.strides[2]];
      p.output[image*(p.output_stride?p.output_stride:p.channels)+channel]=sum/p.image_rows;
    }
  }
};
} // namespace tilemega::codegen
