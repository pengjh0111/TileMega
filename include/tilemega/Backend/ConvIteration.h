// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/DmDescriptors.h>
#include <cstdint>
#include <limits>
#include <stdexcept>

#if defined(__CUDACC__)
#define TILEMEGA_CONV_ITERATION_HD __host__ __device__
#else
#define TILEMEGA_CONV_ITERATION_HD
#endif
namespace tilemega::backend {
struct ConvIterationPoint {
  std::uint32_t r=0,s=0,c=0;
  bool valid=false;
};
// One index numbers issued MMA K tiles, including each filter position's
// channel tail. Split-K partitions these iterations, never flattened logical K.
struct ConvIterationGeometry {
  std::uint32_t rows=0,columns=0,channels=0,tile_k=0;
  std::uint64_t iterations=0;
  static std::uint64_t CountIterations(std::uint32_t cp,std::uint32_t r,
      std::uint32_t s,std::uint32_t tk) {
    if(!cp || !r || !s || !tk)
      throw std::invalid_argument("convolution K iteration needs positive extents");
    std::uint64_t positions=std::uint64_t(r)*s;
    if(positions>std::numeric_limits<std::uint64_t>::max()/cp)
      throw std::overflow_error("convolution reduction capacity overflows");
    return cp>=tk?positions*((std::uint64_t(cp)+tk-1)/tk):
        positions*cp/tk+(positions*cp%tk!=0);
  }
  static ConvIterationGeometry Build(codegen::ConvDesc const& conv,
      codegen::DmBufferLayout const& layout,std::uint32_t tk) {
    if(layout.kind!=codegen::DmLayout::kNHWC || layout.rank!=4 ||
        !conv.r || !conv.s || !conv.c || layout.logical[3]!=conv.c ||
        layout.physical[3]<conv.c || !layout.physical[3] ||
        (tk!=16 && tk!=32 && tk!=64 && tk!=128))
      throw std::invalid_argument("invalid convolution K-iteration geometry");
    auto cp=layout.physical[3];
    auto padded=conv.c<=4?(cp==4 || cp==8):
        cp==((std::uint64_t(conv.c)+7)/8)*8;
    if(!padded || (cp<tk && tk%cp))
      throw std::invalid_argument("convolution channels do not fit its vector packing");
    auto count=CountIterations(cp,conv.r,conv.s,tk);
    return {conv.r,conv.s,cp,tk,count};
  }
  TILEMEGA_CONV_ITERATION_HD ConvIterationPoint At(std::uint64_t iteration,
                                                  std::uint32_t lane) const {
    if(iteration>=iterations || lane>=tile_k)return {};
    std::uint64_t position;std::uint32_t channel;
    if(channels>=tile_k) {
      auto blocks=(std::uint64_t(channels)+tile_k-1)/tile_k;
      position=iteration/blocks;channel=(iteration%blocks)*tile_k+lane;
    } else {
      auto flat=iteration*tile_k;
      if(lane>std::numeric_limits<std::uint64_t>::max()-flat)return {};
      flat+=lane;
      position=flat/channels;channel=flat%channels;
    }
    if(position>=std::uint64_t(rows)*columns || channel>=channels)return {};
    return {std::uint32_t(position/columns),std::uint32_t(position%columns),channel,true};
  }
  TILEMEGA_CONV_ITERATION_HD std::uint64_t Iteration(std::uint32_t r,
      std::uint32_t s,std::uint32_t c) const {
    auto position=std::uint64_t(r)*columns+s;
    return channels>=tile_k?position*((std::uint64_t(channels)+tile_k-1)/tile_k)+c/tile_k:
        (position*channels+c)/tile_k;
  }
};
} // namespace tilemega::backend
#undef TILEMEGA_CONV_ITERATION_HD
