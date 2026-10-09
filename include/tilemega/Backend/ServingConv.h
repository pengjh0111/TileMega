// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ConvIteration.h>
#include <tilemega/Backend/ServingDmGemm.h>

namespace tilemega::backend {

template<class Arch,int TM,int TN,int TK,int Stages>
struct ServingConv {
  using Gemm=ServingDmGemm<Arch,TM,TN,TK,Stages>;
  using Element=typename Gemm::Element;
  using Async=typename Gemm::Async;
  static constexpr int kSharedBytes=Gemm::kSharedBytes;

  template<bool TiledB,class Operands>
  __device__ static float* Run(Operands const& p,codegen::ConvDesc const& conv,
      codegen::DmBufferLayout const& layout,ConvIterationGeometry geometry,
      int tile_m,int tile_n,std::uint64_t begin,int iterations,char* shared) {
    if(!p.a || !p.b || geometry.tile_k!=TK || begin>=geometry.iterations ||
       iterations<=0 || std::uint64_t(iterations)>geometry.iterations-begin ||
       !conv.p || !conv.q || (TiledB && !p.weight_base)) {
      asm volatile("trap;");return nullptr;
    }
    // Four packed BF16 channels form an 8-byte vector. The input pixel pitch
    // remains 16-byte aligned; two adjacent filter positions need two copies.
    if(geometry.channels==4)
      return RunPacked<4,TiledB>(p,conv,layout,geometry,tile_m,tile_n,begin,iterations,shared);
    return RunPacked<8,TiledB>(p,conv,layout,geometry,tile_m,tile_n,begin,iterations,shared);
  }

 private:
  template<int Pack,bool TiledB,class Operands>
  __device__ static float* RunPacked(Operands const& p,codegen::ConvDesc const& conv,
      codegen::DmBufferLayout const& layout,ConvIterationGeometry geometry,
      int tile_m,int tile_n,std::uint64_t begin,int iterations,char* shared) {
    using codegen::executor::ComputeThread;
    using namespace cute;
    constexpr int kAVectors=(TM*TK+128*Pack-1)/(128*Pack);
    auto thread=ComputeThread();
    std::int64_t row_base[kAVectors];
    for(int i=0;i<kAVectors;++i) {
      int vector=thread*Pack+i*128*Pack,row=tile_m*TM+vector/TK;
      if(vector>=TM*TK || row>=p.m) {row_base[i]=0;continue;}
      auto image=row/(conv.p*conv.q),pixel=row%(conv.p*conv.q);
      auto y=std::int64_t(pixel/conv.q)*conv.stride_h-conv.pad_h+layout.halo_top;
      auto x=std::int64_t(pixel%conv.q)*conv.stride_w-conv.pad_w+layout.halo_left;
      row_base[i]=std::int64_t(image)*layout.strides[0]+y*layout.strides[1]+x*layout.strides[2];
    }
    unsigned lane=(thread*Pack)%TK;
    ConvIterationCursor cursor(geometry,conv,layout,begin,lane);
    auto weight_pitch=std::uint64_t(conv.r)*conv.s*geometry.channels;
    auto* a=reinterpret_cast<Element*>(shared);
    auto* b=a+Stages*TM*TK;
    auto issue=[&](int iteration) {
      bool k_valid=cursor.Point().valid;
      for(int i=0;i<kAVectors;++i) {
        int v=thread*Pack+i*128*Pack,row=tile_m*TM+v/TK;
        if(v>=TM*TK)continue;
        bool valid=row<p.m && k_valid;
        auto* source=valid?p.a+row_base[i]+cursor.a_offset:p.a;
        auto* target=a+(iteration%Gemm::kSlots)*TM*TK+typename Gemm::LayoutA{}(v/TK,v%TK);
        if constexpr(Pack==4)Async::Copy8Bytes(target,source,valid?8:0);
        else Async::Copy16Bytes(target,source,valid?16:0);
      }
      if constexpr(TiledB) {
        auto* source=p.weight_base+(std::uint64_t(tile_n)*geometry.iterations+begin+iteration)*TN*TK;
        for(int v=thread*8;v<TN*TK;v+=128*8)
          Async::Copy16Bytes(b+(iteration%Gemm::kSlots)*TN*TK+v,source+v,16);
      }
      if constexpr(!TiledB) {
        for(int v=thread*Pack;v<TN*TK;v+=128*Pack) {
          int column=tile_n*TN+v/TK;
          bool valid=column<p.n && k_valid;
          auto* source=valid?p.b+std::uint64_t(column)*weight_pitch+cursor.b_offset:p.b;
          auto* target=b+(iteration%Gemm::kSlots)*TN*TK+typename Gemm::LayoutB{}(v/TK,v%TK);
          if constexpr(Pack==4)Async::Copy8Bytes(target,source,valid?8:0);
          else Async::Copy16Bytes(target,source,valid?16:0);
        }
      }
      cursor.Advance();
    };
    return Gemm::Pipeline(issue,iterations,shared);
  }
};
} // namespace tilemega::backend
