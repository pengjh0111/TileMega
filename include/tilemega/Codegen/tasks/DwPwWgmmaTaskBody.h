// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/DwPwFusedTaskBody.h>
#include <tilemega/Backend/Sm90Gmma.h>
#include <tilemega/Codegen/executor/TensorMap.h>
namespace tilemega::codegen {
// Stream one K64 depthwise block directly into a canonical SW128 operand.
// Shared space depends on tile geometry, never on the model channel count.
template<class Arch,int Channels,int TM,int TN,int TK,int Stages,class DwProgram,class PwSpec>
struct DwPwWgmmaTaskBody {
  static_assert(arch::Caps<Arch>::kWgmma && Channels>=64 && Channels%64==0);
  static_assert(TM==64 && (TN==64 || TN==128) && TK==64 && Stages==2);
  static_assert(backend::DmEpilogueWalk<DwProgram>::kGates==0);
  using E=cutlass::bfloat16_t;
  using Dw=DepthwiseConvTaskBody<Arch,1,32,DwProgram>;
  using Body=backend::Sm90ForwardGmma<Arch,TM,TN,TK>;
  using Gmma=typename Body::Gmma;
  using Copy=executor::Async<Arch>;
  static constexpr int kSharedBytes=1024+4*TK*(TM+TN)>4*TM*TN?1024+4*TK*(TM+TN):4*TM*TN;
  __device__ static __forceinline__ void DepthwiseBlock(DwPwFusedOperands const& p,
      int tile_m,int tile_n,int iteration,E* tile) {
    using namespace executor;auto const& dw=p.depthwise;auto const& c=dw.convolution;
    auto const& input=dw.input_layout;
    auto layout=cute::as_position_independent_swizzle_layout(backend::Sm90KLayout<TM,TK>{});
    for(unsigned vector=ComputeThread();vector<TM*TK/8;vector+=128) {
      unsigned local_row=vector/(TK/8),row=tile_m*TM+local_row;
      unsigned local_channel=vector%(TK/8)*8,channel=iteration*TK+local_channel;
      float accum[8]={};
      if(row<unsigned(p.pointwise.m)) {
        unsigned image=row/(c.p*c.q),pixel=row%(c.p*c.q);
        long y=long(pixel/c.q)*c.stride_h-c.pad_h+input.halo_top;
        long x=long(pixel%c.q)*c.stride_w-c.pad_w+input.halo_left;
        if(y<0 || x<0 || y+long((c.r-1)*c.dilation_h)>=long(input.physical[1]) ||
           x+long((c.s-1)*c.dilation_w)>=long(input.physical[2])){asm volatile("trap;");return;}
        for(unsigned r=0;r<c.r;++r)for(unsigned s=0;s<c.s;++s) {
          alignas(16) E values[8];auto* source=dw.input+image*input.strides[0]+
              (y+r*c.dilation_h)*input.strides[1]+(x+s*c.dilation_w)*input.strides[2]+channel;
          *reinterpret_cast<uint4*>(values)=backend::LoadGlobal16(source);
          #pragma unroll
          for(unsigned lane=0;lane<8;++lane) {
            auto* weight=dw.weight+((std::size_t(channel+lane)*c.r+r)*c.s+s)*dw.weight_channel_pitch;
            accum[lane]=fmaf(float(values[lane]),float(*weight),accum[lane]);
          }
        }
      }
      alignas(16) E stored[8];
      #pragma unroll
      for(unsigned lane=0;lane<8;++lane) {
        typename Dw::Visitor value{accum[lane],0,channel+lane,0,dw};
        backend::DmEpilogueWalk<DwProgram>::Run(value);
        stored[lane]=row<unsigned(p.pointwise.m)?E(value.first):E(0);
      }
      *reinterpret_cast<uint4*>(tile+layout(local_row,local_channel))=*reinterpret_cast<uint4 const*>(stored);
      if(dw.output && tile_n==0 && row<unsigned(p.pointwise.m)) {
        auto offset=backend::DmTensorAddress<DmWriteKind::kDense>::Offset(dw.output_layout,row,channel,Channels);
        backend::StoreGlobal16(dw.output+offset,*reinterpret_cast<uint4 const*>(stored));
      }
    }
    ComputeSync();Copy::FenceProxyAsyncShared();ComputeSync();
  }
  template<bool TiledB=false>
  __device__ static __forceinline__ void Run(DwPwFusedOperands const& p,int tile_m,int tile_n,char* scratch) {
    using namespace cute;using namespace executor;
    auto const& dw=p.depthwise;auto const& c=dw.convolution;auto const& in=dw.input_layout;auto const& pw=p.pointwise;
    if(!scratch || (Copy::Shared(scratch)&1023) || !dw.input || !dw.weight || !pw.b ||
       (!TiledB && !pw.tensor_map) || (TiledB && !pw.weight_base) || c.c!=Channels || c.k!=Channels ||
       !c.p || !c.q || !c.r || !c.s || !c.stride_h || !c.stride_w || !c.dilation_h || !c.dilation_w ||
       in.rank!=4 || in.kind!=DmLayout::kNHWC || in.strides[3]!=1 || in.strides[2]%8 || in.physical[3]<Channels ||
       dw.weight_channel_pitch<8 || dw.weight_channel_pitch%8 || dw.chain.count!=DwProgram::kCount ||
       dw.chain.side_count || dw.chain.store_rounding!=DmRounding::kBF16 ||
       pw.m!=std::uint64_t(c.n)*c.p*c.q || pw.k_total!=Channels || pw.k_begin!=0 || pw.k_count!=Channels ||
       (pw.b_row_stride && pw.b_row_stride!=Channels) ||
       pw.access.a!=DmAAccess::kDense || pw.access.b!=DmBAccess::kDense || pw.access.a_scale!=kDmNoIndex) {
      asm volatile("trap;");return;
    }
    auto* barriers=reinterpret_cast<std::uint64_t*>(scratch);
    auto* activation=reinterpret_cast<E*>(scratch+1024);auto* weight=activation+2*TM*TK;
    ComputeSync();if(ComputeThread()==0){Copy::Init(barriers,1);Copy::Init(barriers+1,1);Copy::InitFence();}
    ComputeSync();
    auto issue=[&](int it) {
      if(ComputeThread()!=0)return;auto* barrier=barriers+it%2;
      Copy::ExpectTx(barrier,TN*TK*sizeof(E));
      if constexpr(TiledB)Copy::Bulk(weight+(it%2)*TN*TK,
          pw.weight_base+(std::size_t(tile_n)*(Channels/TK)+it)*TN*TK,TN*TK*sizeof(E),barrier);
      else Copy::Tensor2D(weight+(it%2)*TN*TK,pw.tensor_map,it*TK,tile_n*TN,barrier);
    };
    Copy::ProxyAsyncGlobalFence();issue(0);if(Channels/TK>1)issue(1);
    auto accum=partition_fragment_C(typename Body::Mma{},Shape<Int<TM>,Int<TN>>{});clear(accum);Gmma::Fence(accum);
    for(int it=0;it<Channels/TK;++it) {
      DepthwiseBlock(p,tile_m,tile_n,it,activation+(it%2)*TM*TK);
      Copy::Wait(barriers+it%2,(it/2)&1);ComputeSync();
      Body::Accumulate(weight+(it%2)*TN*TK,activation+(it%2)*TM*TK,accum);
      Gmma::template Wait<0>();ComputeSync();if(it+2<Channels/TK)issue(it+2);
    }
    auto* result=Body::Materialize(accum,scratch);
    backend::ServingDmEpilogue<Arch,PwSpec,TM,TN,false>::RunFromTile(
        result,DmEpilogueOperands(pw),tile_m,tile_n);
  }
};
} // namespace tilemega::codegen
