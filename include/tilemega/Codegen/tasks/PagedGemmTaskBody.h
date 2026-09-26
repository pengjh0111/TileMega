// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <tilemega/Codegen/tasks/ServingRMSNormTaskBody.h>
#include <tilemega/Codegen/executor/PageRing.cuh>
#include <tilemega/Solver/PageLayout.h>

namespace tilemega::codegen {
// Decode uses the same MMA ownership and epilogue as the collective, but its
// B pipeline is the worker's cross-task page ring, not per-task storage.
template<class Arch,int TileM,int TileN,int TileK,int PageBytes,int Pages,
         bool ForceSm80=false>
struct PagedGemmTaskBody {
  using Config=backend::ServingGemmConfig<Arch,TileM,TileN,TileK,2>;
  using Element=typename Config::Element;
  using Mma=typename Config::TiledMma;
  using Ring=executor::PageRing<PageBytes,Pages,Arch,ForceSm80>;
  using Async=typename Ring::Copy;
  static constexpr int kBBytes=TileN*TileK*sizeof(Element);
  static constexpr int kGroupPages=kBBytes>PageBytes?kBBytes/PageBytes:1;
  static constexpr int kGroupStages=PageBytes>kBBytes?PageBytes/kBBytes:1;
  static constexpr int kActivationSlots=2;
  static constexpr int kActivationBytes=kActivationSlots*TileM*TileK*sizeof(Element);
  static constexpr int kScratchBytes=4*TileM*TileN;
  static_assert(solver::PageLayout::StageFits(PageBytes,TileN,TileK));
  static_assert(Pages>=kGroupPages,"page ring must hold one B stage");
  using LayoutA=decltype(cute::tile_to_shape(typename Config::SmemLayoutAtom{},
      cute::Shape<cute::Int<TileM>,cute::Int<TileK>>{}));
  using LayoutB=decltype(cute::tile_to_shape(typename Config::SmemLayoutAtom{},
      cute::Shape<cute::Int<TileN>,cute::Int<TileK>>{}));
  static_assert(cute::cosize_v<LayoutA>*sizeof(Element)==TileM*TileK*sizeof(Element));
  static_assert(cute::cosize_v<LayoutB>*sizeof(Element)==kBBytes);

  // Multi-page B stages must be contiguous for CuTe's ldmatrix views. At a
  // physical ring boundary both roles consume the same empty padding slots.
  // The padding is part of the static page stream and never allocates pages.
  __device__ static void AlignLoader(Ring const& ring,std::uint64_t& sequence) {
    int at=Ring::SlotIndex(sequence);
    if(at+kGroupPages<=Pages)return;
    for(;at<Pages;++at,++sequence) {
      ring.AcquireEmpty(sequence);ring.PublishCopies(sequence);
    }
  }
  __device__ static void AlignCompute(Ring const& ring,std::uint64_t& sequence) {
    int at=Ring::SlotIndex(sequence);
    if(at+kGroupPages<=Pages)return;
    for(;at<Pages;++at,++sequence) {
      ring.AwaitFull(sequence);ring.Release(sequence);
    }
  }

  __device__ static void Load(ServingGemmOperands const& p,int tile_n,
                              Ring const& ring,std::uint64_t& sequence) {
    int pitch=p.b_row_stride?p.b_row_stride:p.k_total;
    int iterations=(p.k_count+TileK-1)/TileK;
    for(int first=0;first<iterations;first+=kGroupStages) {
      AlignLoader(ring,sequence);
      for(int page=0;page<kGroupPages;++page)ring.AcquireEmpty(sequence+page);
      if constexpr(kGroupPages>1 && !Async::Caps::kTma) {
        // A CuTe swizzle changes which logical vectors land on each page.
        // Route each vector by its physical address, visiting it only once.
        // Every page is acquired before any copy and every full barrier is
        // published after all copies, so each barrier covers the whole group.
        for(int v=executor::LoaderLane()*8;v<TileN*TileK;v+=executor::kLoaderThreads*8) {
          int n=v/TileK,k=v%TileK;
          int byte=LayoutB{}(n,k)*sizeof(Element);
          int page=byte/PageBytes;
          int global_n=tile_n*TileN+n,local_k=first*TileK+k;
          bool valid=global_n<p.n && local_k<p.k_count;
          auto* src=valid?p.b+std::int64_t(global_n)*pitch+p.k_begin+local_k:p.b;
          Async::Copy16Bytes(ring.Page(sequence+page)+byte%PageBytes,src,
              valid?min(8,p.k_count-local_k)*sizeof(Element):0);
        }
        for(int page=0;page<kGroupPages;++page)ring.PublishCopies(sequence+page);
        sequence+=kGroupPages;
        continue;
      }
      for(int page=0;page<kGroupPages;++page) {
        if constexpr(Async::Caps::kTma) {
          if(p.tensor_map && p.k_count%TileK==0) {
            constexpr int box_rows=TileN<PageBytes/128?TileN:PageBytes/128;
            auto* barrier=&ring.slots[Ring::SlotIndex(sequence+page)].full;
            if(executor::LoaderLane()==0) {
              unsigned bytes=0;
              for(int stage=0;stage<kGroupStages && first+stage<iterations;++stage)
                for(int k=0;k<TileK;k+=64)for(int n=0;n<TileN;n+=box_rows)
                  if((stage*kBBytes+LayoutB{}(n,k)*2)/PageBytes==page)bytes+=box_rows*128;
              Async::ExpectTx(barrier,bytes);
              for(int stage=0;stage<kGroupStages && first+stage<iterations;++stage)
                for(int k=0;k<TileK;k+=64)for(int n=0;n<TileN;n+=box_rows) {
                  int byte=stage*kBBytes+LayoutB{}(n,k)*2;
                  if(byte/PageBytes==page)Async::Tensor2D(
                      ring.Page(sequence+page)+byte%PageBytes,p.tensor_map,
                      p.tensor_k_begin+(first+stage)*TileK+k,tile_n*TileN+n,barrier);
                }
            }else Async::Arrive(barrier);
            continue;
          }
        }
        for(int stage=0;stage<kGroupStages;++stage) {
          for(int v=executor::LoaderLane()*8;v<TileN*TileK;v+=executor::kLoaderThreads*8) {
            int n=v/TileK,k=v%TileK;
            int byte=stage*kBBytes+LayoutB{}(n,k)*sizeof(Element);
            if(byte/PageBytes!=page)continue;
            int global_n=tile_n*TileN+n,local_k=(first+stage)*TileK+k;
            bool valid=first+stage<iterations && global_n<p.n && local_k<p.k_count;
            auto* src=valid?p.b+std::int64_t(global_n)*pitch+p.k_begin+local_k:p.b;
            Async::Copy16Bytes(ring.Page(sequence+page)+byte%PageBytes,src,
                valid?min(8,p.k_count-local_k)*sizeof(Element):0);
          }
        }
        ring.PublishCopies(sequence+page);
      }
      sequence+=kGroupPages;
    }
  }

  __device__ static void LoadActivation(ServingGemmOperands const& p,int tile_m,
      int iteration,Element* shared,Element const* norm_input=nullptr,
      Element const* norm_weight=nullptr,float const* inverse_rms=nullptr) {
    int pitch=p.a_row_stride?p.a_row_stride:p.k_total;
    for(int v=ComputeThread()*8;v<TileM*TileK;v+=kComputeThreads*8) {
      int m=v/TileK,k=v%TileK,global_m=tile_m*TileM+m;
      int local_k=iteration*TileK+k;
      bool valid=global_m<p.m && local_k<p.k_count;
      auto* dest=shared+LayoutA{}(m,k);
      if(norm_input) {
        // The reduction and BF16 boundaries are shared with standalone norm.
        for(int e=0;e<8;++e)dest[e]=valid && local_k+e<p.k_count
            ? ServingRMSNormTaskBody::Transform(
                float(norm_input[std::int64_t(global_m)*pitch+p.k_begin+local_k+e]),
                inverse_rms[m],float(norm_weight[p.k_begin+local_k+e])):Element(0);
      }else {
        auto* src=valid?p.a+std::int64_t(global_m)*pitch+p.k_begin+local_k:p.a;
        Async::Copy16Bytes(dest,src,valid?min(8,p.k_count-local_k)*sizeof(Element):0);
      }
    }
    cute::cp_async_fence();
  }

  __device__ static void Run(ServingGemmOperands const& p,int tile_m,int tile_n,
      Ring const& ring,std::uint64_t& sequence,char* workspace,
      Element const* norm_input=nullptr,Element const* norm_weight=nullptr,
      float epsilon=0.0f) {
    using namespace cute;
    Mma mma;auto thread=mma.get_slice(ComputeThread());
    auto accum=partition_fragment_C(mma,Shape<Int<TileM>,Int<TileN>>{});clear(accum);
    auto* activation=reinterpret_cast<Element*>(workspace);
    float inverses[TileM];
    if(norm_input)for(int m=0;m<TileM;++m) {
      int row=tile_m*TileM+m;
      inverses[m]=row<p.m?ServingRMSNormTaskBody::RowInvRms(
          norm_input+std::int64_t(row)*(p.a_row_stride?p.a_row_stride:p.k_total),p.k_total,epsilon,
          reinterpret_cast<float*>(workspace)):0.0f;
      ComputeSync();
    }
    int iterations=(p.k_count+TileK-1)/TileK;
    if(iterations<=0)return;
    LoadActivation(p,tile_m,0,activation,norm_input,norm_weight,inverses);
    auto copy_a=make_tiled_copy_A(typename Config::SmemCopyAtom{},mma);
    auto copy_b=make_tiled_copy_B(typename Config::SmemCopyAtomB{},mma);
    for(int first=0;first<iterations;first+=kGroupStages) {
      AlignCompute(ring,sequence);
      for(int page=0;page<kGroupPages;++page)ring.AwaitFull(sequence+page);
      for(int stage=0;stage<kGroupStages && first+stage<iterations;++stage) {
        int it=first+stage;
        cute::cp_async_wait<0>();ComputeSync();
        auto sA=make_tensor(make_smem_ptr(activation+(it%kActivationSlots)*TileM*TileK),LayoutA{});
        auto sB=make_tensor(make_smem_ptr(reinterpret_cast<Element*>(ring.Page(sequence)+stage*kBBytes)),LayoutB{});
        auto rA=thread.partition_fragment_A(sA);auto rB=thread.partition_fragment_B(sB);
        auto src_a=copy_a.get_slice(ComputeThread()).partition_S(sA);
        auto src_b=copy_b.get_slice(ComputeThread()).partition_S(sB);
        auto dst_a=copy_a.get_slice(ComputeThread()).retile_D(rA);
        auto dst_b=copy_b.get_slice(ComputeThread()).retile_D(rB);
        if(it+1<iterations)LoadActivation(p,tile_m,it+1,
            activation+((it+1)%kActivationSlots)*TileM*TileK,norm_input,norm_weight,inverses);
        #pragma unroll
        for(int k=0;k<size<2>(rA);++k) {
          copy(typename Config::SmemCopyAtom{},src_a(_,_,k),dst_a(_,_,k));
          copy(typename Config::SmemCopyAtomB{},src_b(_,_,k),dst_b(_,_,k));
          gemm(mma,rA(_,_,k),rB(_,_,k),accum);
        }
      }
      for(int page=0;page<kGroupPages;++page)ring.Release(sequence+page);
      sequence+=kGroupPages;
    }
    auto finish=[&](auto op) {
      backend::ServingEpilogue<decltype(op)::value,TileM,TileN>::Run(
          accum,mma,workspace,tile_m,tile_n,p.m,p.n,p.output_stride,
          p.output,p.residual,p.partial,p.argmax_value,p.argmax_index);
    };
    switch(p.epilogue) {
      case backend::ServingEpilogueOp::kStore: finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kStore>{});break;
      case backend::ServingEpilogueOp::kResidual: finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kResidual>{});break;
      case backend::ServingEpilogueOp::kSwiGLU: finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kSwiGLU>{});break;
      case backend::ServingEpilogueOp::kArgmaxPartial: finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kArgmaxPartial>{});break;
      case backend::ServingEpilogueOp::kPartial: finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kPartial>{});break;
    }
  }
};
} // namespace tilemega::codegen
