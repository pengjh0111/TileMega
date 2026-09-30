// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <tilemega/Codegen/executor/PageRing.cuh>
#include <tilemega/Solver/PageLayout.h>

namespace tilemega::codegen {
#ifndef TILEMEGA_WEIGHT_LAYOUT_TILED
#define TILEMEGA_WEIGHT_LAYOUT_TILED 0
#endif
#ifndef TILEMEGA_EVICT_FIRST
#define TILEMEGA_EVICT_FIRST 0
#endif
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
  static_assert(kGroupPages==1,"paged decode uses one-page weight stages");
  static constexpr int kGroupStages=PageBytes>kBBytes?PageBytes/kBBytes:1;
  static constexpr int kActivationSlots=4;
  static constexpr int kActivationBytes=kActivationSlots*TileM*TileK*sizeof(Element);
  static constexpr int kScratchBytes=4*TileM*TileN+4*TileM;
  static_assert(solver::PageLayout::StageFits(PageBytes,TileN,TileK));
  static_assert(Pages>=kGroupPages,"page ring must hold one B stage");
  using LayoutA=decltype(cute::tile_to_shape(typename Config::SmemLayoutAtom{},
      cute::Shape<cute::Int<TileM>,cute::Int<TileK>>{}));
  using LayoutB=decltype(cute::tile_to_shape(typename Config::SmemLayoutAtom{},
      cute::Shape<cute::Int<TileN>,cute::Int<TileK>>{}));
  static_assert(cute::cosize_v<LayoutA>*sizeof(Element)==TileM*TileK*sizeof(Element));
  static_assert(cute::cosize_v<LayoutB>*sizeof(Element)==kBBytes);

  template<class BeforePage=executor::NoPageHook>
  __device__ static void LoadRow(ServingGemmOperands const& p,int tile_n,
                              Ring const& ring,std::uint64_t& sequence,
                              BeforePage const& before_page=BeforePage{}) {
    int pitch=p.b_row_stride?p.b_row_stride:p.k_total;
    int iterations=(p.k_count+TileK-1)/TileK;
    for(int first=0;first<iterations;first+=kGroupStages) {
      for(int page=0;page<kGroupPages;++page) {
        before_page(0);ring.AcquireEmpty(sequence+page);
      }
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
            // A single-page group covers every vector of every stage.
            // Keeping the runtime page test here adds a divide and branch to
            // each 16 B copy despite StageFits proving the page is always 0.
            if constexpr(kGroupPages>1) {
              if(byte/PageBytes!=page)continue;
            }
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

  template<class BeforePage=executor::NoPageHook>
  __device__ static void LoadTile(ServingGemmOperands const& p,int tile_n,
                                  Ring const& ring,std::uint64_t& sequence,
                              BeforePage const& before_page=BeforePage{}) {
    if(!p.weight_base || p.k_total_full<=0 || p.k_begin%TileK) {
      asm volatile("trap;");return;
    }
    int const kt=(p.k_total_full+TileK-1)/TileK;
    int const t0=p.k_begin/TileK;
    int const iters=(p.k_count+TileK-1)/TileK;
    char const* source=reinterpret_cast<char const*>(p.weight_base)+
        (std::size_t(tile_n)*kt+t0)*kBBytes;
    int const total=iters*kBBytes;
    std::uint64_t policy=Async::EvictFirst();
    for(int off=0;off<total;off+=PageBytes,++sequence) {
      int bytes=min(PageBytes,total-off);
      before_page(bytes);
      ring.AcquireEmpty(sequence);
      if constexpr(Async::Caps::kBulkCopy) {
        if constexpr(TILEMEGA_EVICT_FIRST)
          ring.PublishBulkHint(sequence,source+off,bytes,policy);
        else ring.PublishBulk(sequence,source+off,bytes);
      }else {
        char* destination=ring.Page(sequence);
        #pragma unroll 8
        for(int vector=executor::LoaderLane();vector*16<bytes;vector+=32)
          if constexpr(TILEMEGA_EVICT_FIRST)
            Async::Copy16Hint(destination+16*vector,source+off+16*vector,policy);
          else Async::Copy16Bytes(destination+16*vector,source+off+16*vector,16);
        ring.PublishCopies(sequence);
      }
    }
  }

  template<class BeforePage=executor::NoPageHook>
  __device__ static void Load(ServingGemmOperands const& p,int tile_n,
                              Ring const& ring,std::uint64_t& sequence,
                              BeforePage const& before_page=BeforePage{}) {
    if constexpr(TILEMEGA_WEIGHT_LAYOUT_TILED) LoadTile(p,tile_n,ring,sequence,before_page);
    else LoadRow(p,tile_n,ring,sequence,before_page);
  }

  __device__ static void LoadActivation(ServingGemmOperands const& p,int tile_m,
      int iteration,Element* shared) {
    int pitch=p.a_row_stride?p.a_row_stride:p.k_total;
    for(int v=ComputeThread()*8;v<TileM*TileK;v+=kComputeThreads*8) {
      int m=v/TileK,k=v%TileK,global_m=tile_m*TileM+m;
      int local_k=iteration*TileK+k;
      if(global_m>=p.m)continue;
      bool valid=local_k<p.k_count;
      auto* dest=shared+LayoutA{}(m,k);
      auto* src=valid?p.a+std::int64_t(global_m)*pitch+p.k_begin+local_k:p.a;
      Async::Copy16Bytes(dest,src,valid?min(8,p.k_count-local_k)*sizeof(Element):0);
    }
    cute::cp_async_fence();
  }

  __device__ static void ZeroInactiveRows(ServingGemmOperands const& p,int tile_m,
                                          Element* activation) {
    for(int vector=ComputeThread()*8;vector<kActivationSlots*TileM*TileK;
        vector+=kComputeThreads*8) {
      int row=(vector%(TileM*TileK))/TileK;
      if(tile_m*TileM+row<p.m)continue;
      auto* dest=activation+vector;
      *reinterpret_cast<uint4*>(dest)=make_uint4(0,0,0,0);
    }
    ComputeSync();
  }

  struct NoPhaseGate {
    bool enabled=false;
    __device__ bool Ready() const {return true;}
    __device__ void Wait(int) const {}
  };
  template<class Gate=NoPhaseGate>
  __device__ static void Run(ServingGemmOperands const& p,int tile_m,int tile_n,
      Ring const& ring,std::uint64_t& sequence,char* workspace,
      Gate gate={}) {
    using namespace cute;
    Mma mma;auto thread=mma.get_slice(ComputeThread());
    auto accum=partition_fragment_C(mma,Shape<Int<TileM>,Int<TileN>>{});clear(accum);
    auto* activation=reinterpret_cast<Element*>(workspace);
    int iterations=(p.k_count+TileK-1)/TileK;
    if(iterations<=0)return;
    ZeroInactiveRows(p,tile_m,activation);
    bool gated=gate.enabled && !gate.Ready();
    if(!gated)for(int initial=0;initial<kActivationSlots-1;++initial)
      if(initial<iterations)LoadActivation(p,tile_m,initial,
          activation+initial*TileM*TileK);
      else cute::cp_async_fence();
    auto copy_a=make_tiled_copy_A(typename Config::SmemCopyAtom{},mma);
    auto copy_b=make_tiled_copy_B(typename Config::SmemCopyAtomB{},mma);
    for(int first=0;first<iterations;first+=kGroupStages) {
      for(int page=0;page<kGroupPages;++page)ring.AwaitFull(sequence+page);
      for(int stage=0;stage<kGroupStages && first+stage<iterations;++stage) {
        int it=first+stage;
        bool direct=false;
        if(gated) {
          gate.Wait(p.k_begin/TileK+it);
          if(gate.Ready()) {
            gated=false;
            for(int prime=it;prime<it+kActivationSlots-1;++prime)
              if(prime<iterations)LoadActivation(p,tile_m,prime,
                  activation+(prime%kActivationSlots)*TileM*TileK);
              else cute::cp_async_fence();
          }else {
            LoadActivation(p,tile_m,it,
                activation+(it%kActivationSlots)*TileM*TileK);
            cute::cp_async_wait<0>();ComputeSync();direct=true;
          }
        }
        if(!direct){cute::cp_async_wait<kActivationSlots-2>();ComputeSync();}
        auto sA=make_tensor(make_smem_ptr(activation+(it%kActivationSlots)*TileM*TileK),LayoutA{});
        auto sB=make_tensor(make_smem_ptr(reinterpret_cast<Element*>(ring.Page(sequence)+stage*kBBytes)),LayoutB{});
        auto rA=thread.partition_fragment_A(sA);auto rB=thread.partition_fragment_B(sB);
        auto src_a=copy_a.get_slice(ComputeThread()).partition_S(sA);
        auto src_b=copy_b.get_slice(ComputeThread()).partition_S(sB);
        auto dst_a=copy_a.get_slice(ComputeThread()).retile_D(rA);
        auto dst_b=copy_b.get_slice(ComputeThread()).retile_D(rB);
        int ahead=it+kActivationSlots-1;
        if(!direct) {
          if(ahead<iterations)LoadActivation(p,tile_m,ahead,
              activation+(ahead%kActivationSlots)*TileM*TileK);
          else cute::cp_async_fence();
        }
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
          p.partial_stride,
          p.output,p.residual,p.partial,p.argmax_value,p.argmax_index,
          p.norm_ss,p.ss_out,p.norm_k,p.norm_eps);
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
