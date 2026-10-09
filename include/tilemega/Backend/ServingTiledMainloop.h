// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ServingMmaPipeline.h>
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cute/tensor.hpp>

namespace tilemega::backend {
// C-WL keeps the MMA shape and K order of the row-layout collective. Only
// global B copies change: the plan supplies an already-swizzled smem image.
template<class Arch,class Config,int TM,int TN,int TK,int Stages>
struct ServingTiledMainloop {
  using E=typename Config::Element;
  using Async=codegen::executor::Async<Arch>;
  using LA=decltype(cute::tile_to_shape(typename Config::SmemLayoutAtom{},
      cute::Shape<cute::Int<TM>,cute::Int<TK>>{}));
  using LB=decltype(cute::tile_to_shape(typename Config::SmemLayoutAtom{},
      cute::Shape<cute::Int<TN>,cute::Int<TK>>{}));
  static_assert(arch::Caps<Arch>::kCpAsync);
  // PTX exposes at most eight committed groups; retain the plan allocation
  // and residency while bounding the active copy pipeline to that limit.
  static constexpr int Slots=Stages<8?Stages:8;
  static_assert(cute::cosize_v<LA> ==TM*TK && cute::cosize_v<LB> ==TN*TK);
  template<class Operands,class Accum>
  __device__ static void Run(Operands const& p,int tm,int tn,char* shared,Accum& accum) {
    using namespace cute;
    using codegen::executor::ComputeThread;
    using codegen::executor::ComputeSync;
    auto* a=reinterpret_cast<E*>(shared);
    auto* b=a+Stages*TM*TK;
    int const iterations=(p.k_count+TK-1)/TK;
    int const kt=(p.k_total_full+TK-1)/TK;
    auto issue=[&](int it) {
      auto* sa=a+(it%Slots)*TM*TK;
      auto* sb=b+(it%Slots)*TN*TK;
      for(int v=ComputeThread()*8;v<TM*TK;v+=128*8) {
        int const row=tm*TM+v/TK,col=it*TK+v%TK;
        bool const valid=row<p.m && col<p.k_count;
        auto* src=valid?p.a+std::size_t(row)*p.a_row_stride+p.k_begin+col:p.a;
        Async::Copy16Bytes(sa+LA{}(v/TK,v%TK),src,
            valid?min(8,p.k_count-col)*int(sizeof(E)):0);
      }
      auto* src=p.weight_base+(std::size_t(tn)*kt+p.k_begin/TK+it)*TN*TK;
      // Packed N/K tails contain zeros, so no per-vector predicate or layout
      // function is needed on the B path.
      for(int v=ComputeThread()*8;v<TN*TK;v+=128*8)
        Async::Copy16Bytes(sb+v,src+v,16);
      cp_async_fence();
    };
    // Empty commits at either end preserve the group distance even for a
    // one-stage task. The slot is reused only after all MMA readers converge.
    for(int it=0;it<Slots-1;++it) {
      if(it<iterations)issue(it);else cp_async_fence();
    }
    typename Config::TiledMma mma;
    auto thread=mma.get_slice(ComputeThread());
    auto ca=make_tiled_copy_A(typename Config::SmemCopyAtom{},mma);
    auto cb=make_tiled_copy_B(typename Config::SmemCopyAtomB{},mma);
    for(int it=0;it<iterations;++it) {
      cp_async_wait<Slots-2>();ComputeSync();
#if TILEMEGA_TRACE_TASK
      if(it==0)codegen::FirstTileReadyProfile{p.profile}();
#endif
      int const ahead=it+Slots-1;
      if(ahead<iterations)issue(ahead);else cp_async_fence();
      auto sa=make_tensor(make_smem_ptr(a+(it%Slots)*TM*TK),LA{});
      auto sb=make_tensor(make_smem_ptr(b+(it%Slots)*TN*TK),LB{});
      auto ra=thread.partition_fragment_A(sa);auto rb=thread.partition_fragment_B(sb);
      auto as=ca.get_slice(ComputeThread()).partition_S(sa);
      auto bs=cb.get_slice(ComputeThread()).partition_S(sb);
      auto ad=ca.get_slice(ComputeThread()).retile_D(ra);
      auto bd=cb.get_slice(ComputeThread()).retile_D(rb);
#if TILEMEGA_MMA_REG_PIPE
      ServingMmaRegisterPipeline<Arch,Config>(mma,accum,ra,rb,as,bs,ad,bd);
#else
      #pragma unroll
      for(int k=0;k<size<2>(ra);++k) {
        copy(typename Config::SmemCopyAtom{},as(_,_,k),ad(_,_,k));
        copy(typename Config::SmemCopyAtomB{},bs(_,_,k),bd(_,_,k));
        gemm(mma,ra(_,_,k),rb(_,_,k),accum);
      }
#endif
    }
    // All copies must finish before the shared union becomes epilogue scratch.
    cp_async_wait<0>();ComputeSync();
  }
};
} // namespace tilemega::backend
