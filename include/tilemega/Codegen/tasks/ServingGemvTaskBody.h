// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ServingGemv.h>
#include <tilemega/Backend/Sm90GemvBulk.h>
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#ifndef TILEMEGA_GEMV_LOAD_BULK
#define TILEMEGA_GEMV_LOAD_BULK 0
#endif
static_assert(TILEMEGA_GEMV_LOAD_BULK==0 || TILEMEGA_GEMV_LOAD_BULK==1,
              "invalid bulk GEMV load policy");
static_assert(!(TILEMEGA_GEMV_LOAD_BULK && TILEMEGA_GEMV_LOAD_NC),
              "bulk and NC GEMV transports are mutually exclusive");
namespace tilemega::codegen {
template<class Arch,int TileN,int TileK,bool Tiled=false>
struct ServingGemvTaskBody {
  using Impl=std::conditional_t<TILEMEGA_GEMV_LOAD_BULK!=0,
      backend::Sm90GemvBulk<Arch,16,TileN,TileK>,
      backend::ServingGemv<Arch,16,TileN,TileK>>;
  static constexpr int kThreads=128,kSharedBytes=Impl::kSharedBytes;
  template<class Emit>
  __device__ static void PrefetchRanges(ServingGemmOperands const& p,int tile_n,Emit emit) {
    if constexpr(Tiled) {
      int kt=(p.k_total_full+TileK-1)/TileK;
      emit(executor::PrefetchRange{p.weight_base+
          (std::size_t(tile_n)*kt+p.k_begin/TileK)*TileN*TileK,
          unsigned(((p.k_count+TileK-1)/TileK)*TileN*TileK*2)});
    }else {
      int pitch=p.b_row_stride?p.b_row_stride:p.k_total;
      for(int n=tile_n*TileN;n<min(p.n,(tile_n+1)*TileN);++n)
        emit(executor::PrefetchRange{p.b+std::int64_t(n)*pitch+p.k_begin,unsigned(2*p.k_count)});
    }
  }
  __device__ static void Run(ServingGemmOperands const& p,int tile_m,int tile_n,char* shared) {
    auto* tile=Impl::template Dense<true>(p,tile_m,tile_n,shared,Tiled);
#if TILEMEGA_TRACE_TASK
    auto epilogue_begin=TaskProfileNow(p.profile);
#endif
    auto finish=[&](auto op) {
      backend::ServingEpilogue<decltype(op)::value,16,TileN,(TILEMEGA_SWIGLU_U<TileN/2?TILEMEGA_SWIGLU_U:TileN/2)>::template RunFromTile<true>(
          tile,tile_m,tile_n,p.m,p.n,p.output_stride,p.partial_stride,p.output,p.residual,
          p.partial,p.argmax_value,p.argmax_index,p.norm_ss,p.ss_out,p.norm_k,p.norm_eps);
    };
    using Op=backend::ServingEpilogueOp;
    switch(p.epilogue) {
      case Op::kStore:finish(std::integral_constant<Op,Op::kStore>{});break;
      case Op::kResidual:finish(std::integral_constant<Op,Op::kResidual>{});break;
      case Op::kPartial:finish(std::integral_constant<Op,Op::kPartial>{});break;
      case Op::kArgmaxPartial:finish(std::integral_constant<Op,Op::kArgmaxPartial>{});break;
      case Op::kSwiGLU:
        finish(std::integral_constant<Op,Op::kSwiGLU>{});
        break;
    }
#if TILEMEGA_TRACE_TASK
    if(p.profile && ComputeThread()==0)p.profile->epilogue_ns=TaskProfileNow(p.profile)-epilogue_begin;
#endif
  }
};
}
