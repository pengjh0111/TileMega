// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/DepthwiseConvTaskBody.h>
#include <tilemega/Codegen/tasks/PagedGemmTaskBody.h>

namespace tilemega::codegen {
#ifndef TILEMEGA_DM_FUSED_DISPATCH
template<class Runner>
__device__ inline bool DispatchDmFused(std::uint32_t,Runner const&) {return false;}
#endif
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
  using E=cutlass::bfloat16_t;
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
      asm volatile("trap;");return;
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
          asm volatile("trap;");return;
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

  template<int TileRows,int TileColumns>
  struct SharedActivation {
    E const* depthwise;
    int begin,count;
    __device__ void CopyValues(int iteration,E* target) const {
      using Layout=typename backend::ServingDmGemm<Arch,TileRows,TileColumns,TK,2>::LayoutA;
      for(unsigned vector=executor::ComputeThread();vector<TileRows*TK/8;vector+=kThreads) {
        unsigned row=vector/(TK/8),column=vector%(TK/8)*8;
        #pragma unroll
        for(unsigned lane=0;lane<8;++lane) {
          unsigned k=iteration*TK+column+lane;
          target[Layout{}(row,column+lane)]=k<unsigned(count)?depthwise[row*Channels+begin+k]:E(0);
        }
      }
    }
    __device__ void Copy(int iteration,E* target,int) const {
      CopyValues(iteration,target);
      // The paged mainloop tracks one async group per activation slot. Its
      // subsequent compute join publishes these shared-to-shared stores.
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
  };

  template<bool TiledB=false>
  __device__ static void Run(DwPwFusedOperands const& p,int tile_m,int tile_n,char* scratch) {
    using namespace executor;
    if(!scratch || !p.pointwise.b || (TiledB && !p.pointwise.weight_base)) {
      asm volatile("trap;");return;
    }
    auto* depthwise=reinterpret_cast<E*>(scratch);
    auto* workspace=scratch+kDepthwiseBytes;
    Depthwise(p,tile_m,tile_n,depthwise);
    SharedActivation<TM,TN> reader{depthwise,p.pointwise.k_begin,p.pointwise.k_count};
    auto* activation=reinterpret_cast<E*>(workspace);
    auto* weight=activation+Stages*TM*TK;
    auto issue=[&](int iteration) {
      reader.CopyValues(iteration,activation+(iteration%Gemm::kSlots)*TM*TK);
      if constexpr(TiledB) {
        auto* source=p.pointwise.weight_base+
            (std::size_t(tile_n)*((Channels+TK-1)/TK)+p.pointwise.k_begin/TK+iteration)*TN*TK;
        for(unsigned vector=ComputeThread();vector<TN*TK/8;vector+=kThreads)
          executor::Async<Arch>::Copy16Bytes(weight+(iteration%Gemm::kSlots)*TN*TK+vector*8,source+vector*8,16);
      }else {
        unsigned pitch=p.pointwise.b_row_stride?p.pointwise.b_row_stride:Channels;
        for(unsigned vector=ComputeThread();vector<TN*TK/8;vector+=kThreads) {
          unsigned n=tile_n*TN+vector/(TK/8),k=iteration*TK+vector%(TK/8)*8;
          bool valid=n<unsigned(p.pointwise.n) && k<unsigned(p.pointwise.k_count);
          unsigned count=valid?min(8u,unsigned(p.pointwise.k_count)-k):0;
          auto* source=valid?p.pointwise.b+std::size_t(n)*pitch+p.pointwise.k_begin+k:p.pointwise.b;
          executor::Async<Arch>::Copy16Bytes(weight+(iteration%Gemm::kSlots)*TN*TK+
              typename Gemm::LayoutB{}(vector/(TK/8),vector%(TK/8)*8),source,count*sizeof(E));
        }
      }
    };
    auto* result=Gemm::Pipeline(issue,(p.pointwise.k_count+TK-1)/TK,workspace);
    backend::ServingDmEpilogue<Arch,PwSpec,TM,TN,false>::RunFromTile(
        result,DmEpilogueOperands(p.pointwise),tile_m,tile_n);
  }

  template<int PageBytes,int Pages,bool ForceSm80=false>
  static constexpr int PagedSharedBytes() {
    return kDepthwiseBytes+PagedGemmTaskBody<Arch,TM,TN,TK,PageBytes,Pages,ForceSm80>::kDmWorkspaceBytes;
  }
  template<int PageBytes,int Pages,bool ForceSm80=false>
  __device__ static void RunPaged(DwPwFusedOperands const& p,int tile_m,int tile_n,
      typename PagedGemmTaskBody<Arch,TM,TN,TK,PageBytes,Pages,ForceSm80>::Ring const& ring,
      std::uint64_t& sequence,char* scratch) {
    using Paged=PagedGemmTaskBody<Arch,TM,TN,TK,PageBytes,Pages,ForceSm80>;
    if(!scratch) {asm volatile("trap;");return;}
    auto* depthwise=reinterpret_cast<E*>(scratch);
    Depthwise(p,tile_m,tile_n,depthwise);
    SharedActivation<TM,TN> reader{depthwise,p.pointwise.k_begin,p.pointwise.k_count};
    Paged::template RunDmActivation<PwSpec>(p.pointwise,tile_m,tile_n,ring,sequence,
        scratch+kDepthwiseBytes,reader,typename Paged::NoPhaseGate{});
  }
};
} // namespace tilemega::codegen
