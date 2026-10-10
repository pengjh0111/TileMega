// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ConvIteration.h>
#include <limits>
#include <stdexcept>

namespace tilemega::codegen {
struct DmConvRuntime {
  backend::ConvIterationGeometry geometry;
  int storage_k=0,tiles=0;
  static DmConvRuntime Build(ConvDesc const& conv,DmBufferLayout const& layout,
      int rows,int columns,int logical_k,int tile_k,int split) {
    auto geometry=backend::ConvIterationGeometry::Build(conv,layout,tile_k);
    auto pixels=std::uint64_t(conv.n)*conv.p*conv.q;
    auto reduction=std::uint64_t(conv.c)*conv.r*conv.s;
    auto height=std::int64_t(conv.h)+2ll*conv.pad_h-
        std::int64_t(conv.dilation_h)*(conv.r-1)-1;
    auto width=std::int64_t(conv.w)+2ll*conv.pad_w-
        std::int64_t(conv.dilation_w)*(conv.s-1)-1;
    if(rows<=0 || columns<=0 || logical_k<=0 || split<=0 ||
       !conv.n || !conv.h || !conv.w || !conv.stride_h || !conv.stride_w ||
       !conv.dilation_h || !conv.dilation_w || height<0 || width<0 ||
       std::uint64_t(height/conv.stride_h+1)!=conv.p ||
       std::uint64_t(width/conv.stride_w+1)!=conv.q ||
       pixels!=std::uint64_t(rows) || columns!=int(conv.k) ||
       reduction!=std::uint64_t(logical_k) ||
       layout.logical[0]!=conv.n || layout.logical[1]!=conv.h ||
       layout.logical[2]!=conv.w || layout.fill!=DmFill::kZero ||
       layout.physical[0]<conv.n ||
       layout.physical[1]<std::uint64_t(conv.h)+layout.halo_top+layout.halo_bottom ||
       layout.physical[2]<std::uint64_t(conv.w)+layout.halo_left+layout.halo_right ||
       layout.halo_top<conv.pad_h || layout.halo_bottom<conv.pad_h ||
       layout.halo_left<conv.pad_w || layout.halo_right<conv.pad_w ||
       layout.strides[3]!=1 || layout.strides[2]%(geometry.channels==4?4:8) ||
       layout.strides[2]<layout.physical[3] ||
       layout.strides[1]<std::uint64_t(layout.physical[2])*layout.strides[2] ||
       layout.strides[0]<std::uint64_t(layout.physical[1])*layout.strides[1] ||
       geometry.iterations%split || geometry.iterations<std::uint64_t(split) ||
       geometry.iterations>std::uint64_t(std::numeric_limits<int>::max())/tile_k)
      throw std::invalid_argument("convolution invocation differs from its exact ownership");
    return {geometry,int(geometry.iterations*tile_k),int(geometry.iterations)};
  }
};
} // namespace tilemega::codegen
