// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Maca/DevicePrimitives.h>
#include <tilemega/Codegen/tasks/DepthwiseConvTaskBody.h>
#include <tilemega/Backend/ServingDmGemm.h>
#include <tilemega/Backend/ServingDmEpilogue.h>

namespace tilemega::codegen {
#ifndef TILEMEGA_DM_FUSED_DISPATCH
template<class Runner>
__device__ inline bool DispatchDmFused(std::uint32_t,Runner const&) {return false;}
#endif
template<int Variant, class Runner>
__device__ inline bool DispatchDmVariantFused(std::uint32_t stage,
                                             Runner const& runner) {
#if defined(TILEMEGA_DM_TYPED_DISPATCH) && defined(TILEMEGA_DM_FUSED_DISPATCH)
  return DispatchDmFused<Variant>(stage, runner);
#else
  return DispatchDmFused(stage, runner);
#endif
}
struct DwPwFusedOperands {
  DepthwiseConvOperands depthwise{};
  ServingGemmOperands pointwise{};
};

// Each N tile recomputes the full depthwise A tile. The only optional global
// depthwise store belongs to N tile zero and K chunk zero, avoiding duplicate writers.
template<class Arch,int Channels,int TM,int TN,int TK,int Stages,
         class DwProgram,class PwSpec>
struct DwPwFusedTaskBody {
  static_assert(Channels>0 && Channels%8==0);
  static_assert(backend::DmEpilogueWalk<DwProgram>::kGates==0);
  using E=tilemega::Bf16;
  using Gemm=backend::ServingDmGemm<Arch,TM,TN,TK,Stages>;
  using Dw=DepthwiseConvTaskBody<Arch,1,32,DwProgram>;
  static constexpr int kThreads=128;
  static constexpr int kDepthwiseBytes=TM*Channels*sizeof(E);
  static constexpr int kSharedBytes=kDepthwiseBytes+Gemm::kSharedBytes;

  __device__ static void Depthwise(DwPwFusedOperands const& p,int tile_m,int tile_n,E* tile) {
    using namespace executor;
    auto const& dw=p.depthwise;auto const& c=dw.convolution;
    auto const& input=dw.input_layout;
    if(!dw.input || !dw.weight || !tile || !c.p || !c.q || c.c!=Channels || c.k!=Channels ||
       !c.r || !c.s || !c.stride_h || !c.stride_w || !c.dilation_h || !c.dilation_w ||
       input.kind!=DmLayout::kNHWC || input.rank!=4 || input.strides[3]!=1 ||
       input.strides[2]%8 || input.physical[3]<Channels || dw.weight_channel_pitch%8 ||
       dw.chain.count!=DwProgram::kCount || dw.chain.side_count ||
       dw.chain.store_rounding!=DmRounding::kBF16 ||
       p.pointwise.m!=std::uint64_t(c.n)*c.p*c.q || p.pointwise.k_total!=Channels ||
       p.pointwise.k_begin<0 || p.pointwise.k_begin%TK || p.pointwise.k_count<=0 ||
       p.pointwise.k_begin+p.pointwise.k_count>Channels ||
       p.pointwise.access.a!=DmAAccess::kDense || p.pointwise.access.b!=DmBAccess::kDense ||
       p.pointwise.access.a_scale!=kDmNoIndex ||
       (p.pointwise.b_row_stride && (p.pointwise.b_row_stride%8 || p.pointwise.b_row_stride<Channels))) {
      maca::Trap();return;
    }
    for(unsigned vector=ComputeThread();vector<TM*Channels/8;vector+=kThreads) {
      unsigned row=tile_m*TM+vector/(Channels/8),channel=vector%(Channels/8)*8;
      float accum[8]={};
      if(row<unsigned(p.pointwise.m)) {
        unsigned image=row/(c.p*c.q),pixel=row%(c.p*c.q);
        long y=long(pixel/c.q)*c.stride_h-c.pad_h+input.halo_top;
        long x=long(pixel%c.q)*c.stride_w-c.pad_w+input.halo_left;
        if(y<0 || x<0 || y+long((c.r-1)*c.dilation_h)>=long(input.physical[1]) ||
           x+long((c.s-1)*c.dilation_w)>=long(input.physical[2])) {
          maca::Trap();return;
        }
        for(unsigned r=0;r<c.r;++r)for(unsigned s=0;s<c.s;++s) {
          alignas(16) E values[8];
          auto source=dw.input+image*input.strides[0]+(y+r*c.dilation_h)*input.strides[1]+
              (x+s*c.dilation_w)*input.strides[2]+channel;
          *reinterpret_cast<uint4*>(values)=*reinterpret_cast<uint4 const*>(source);
          #pragma unroll
          for(unsigned lane=0;lane<8;++lane) {
            auto weight=dw.weight+((std::size_t(channel+lane)*c.r+r)*c.s+s)*dw.weight_channel_pitch;
            accum[lane]=fmaf(float(values[lane]),float(*weight),accum[lane]);
          }
        }
      }
      alignas(16) E stored[8];
      #pragma unroll
      for(unsigned lane=0;lane<8;++lane) {
        typename Dw::Visitor value{accum[lane],0,channel+lane,0,dw};
        backend::DmEpilogueWalk<DwProgram>::Run(value);
        // The pointwise MMA observes the same BF16 materialization as the
        // unfused depthwise stage, including its declared chain rounding.
        stored[lane]=row<unsigned(p.pointwise.m)?E(value.first):E(0);
      }
      *reinterpret_cast<uint4*>(tile+vector*8)=*reinterpret_cast<uint4 const*>(stored);
      if(dw.output && tile_n==0 && p.pointwise.k_begin==0 && row<unsigned(p.pointwise.m)) {
        auto const& out=dw.output_layout;
        auto offset=backend::DmTensorAddress<DmWriteKind::kDense>::Offset(out,row,channel,Channels);
        *reinterpret_cast<uint4*>(dw.output+offset)=*reinterpret_cast<uint4 const*>(stored);
      }
    }
    ComputeSync();
  }

  /// The pointwise GEMM's A operand is the depthwise tile in shared memory
  /// ([TM][Channels] BF16, rows past m already zero); B is the DM GEMM's
  /// weight source (rows or fragment-packed tiles).
  template<bool TiledB>
  struct PointwiseSource {
    using Inner=typename Gemm::template Source<TiledB,ServingGemmOperands>;
    E const* depthwise;
    Inner inner;
    __device__ PointwiseSource(E const* tile,ServingGemmOperands const& p,typename Gemm::Lane const& l,
                               int tile_m,int tile_n,bool& ok)
        :depthwise(tile),inner(p,l,tile_m,tile_n,ok) {}
    __device__ uint4 A(int am,int kc) const {
      return *reinterpret_cast<uint4 const*>(depthwise+(am*16+inner.l.r)*Channels+inner.p.k_begin+kc+8*inner.l.g);
    }
    __device__ uint4 ATail(int am,int kc,int count) const {
      return kc+8*inner.l.g<count?A(am,kc):uint4{0,0,0,0};
    }
    __device__ uint4 B(int j,int kc) const {return inner.B(j,kc);}
    __device__ uint4 BTail(int j,int kc,int count) const {return inner.BTail(j,kc,count);}
  };

  template<bool TiledB=false>
  __device__ static void Run(DwPwFusedOperands const& p,int tile_m,int tile_n,char* scratch) {
    using namespace executor;
    if(!scratch || !p.pointwise.b || (TiledB && !p.pointwise.weight_base) ||
       p.pointwise.k_count%8 || (TiledB && p.pointwise.k_total_full!=Channels)) {
      maca::Trap();return;
    }
    auto* depthwise=reinterpret_cast<E*>(scratch);
    auto* workspace=scratch+kDepthwiseBytes;
    Depthwise(p,tile_m,tile_n,depthwise);
    typename Gemm::Lane const l;
    bool ok=true;
    PointwiseSource<TiledB> source(depthwise,p.pointwise,l,tile_m,tile_n,ok);
    if(!ok) {maca::Trap();return;}
    auto* result=Gemm::Mma::Run(source,l,p.pointwise.k_count,p.pointwise.m-tile_m*TM,
                                p.pointwise.n-tile_n*TN,workspace);
    backend::ServingDmEpilogue<Arch,PwSpec,TM,TN,false>::RunFromTile(
        result,DmEpilogueOperands(p.pointwise),tile_m,tile_n);
  }
};
} // namespace tilemega::codegen
