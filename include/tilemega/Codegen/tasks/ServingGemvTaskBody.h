// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Mx/ServingGemvMx.h>
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
namespace tilemega::codegen {
/// xcore1000 SIMT GEMV TaskBody (M <= 4): Backend/Mx/ServingGemvMx.h fills
/// the FP32 tile, the shared epilogue rounds and stores it.
template<class Arch,int TileN,int TileK,bool Tiled=false>
struct ServingGemvTaskBody {
  static_assert(!Tiled, "tiled GEMV weights are not supported on xcore1000");
  using Impl=backend::ServingGemvMx<16,TileN,TileK>;
  static constexpr int kThreads=128,kSharedBytes=solver::ServingGemvSmemBytes(16,TileN);
  template<class Emit>
  __device__ static void PrefetchRanges(ServingGemmOperands const& p,int tile_n,Emit emit) {
    int pitch=p.b_row_stride?p.b_row_stride:p.k_total;
    for(int n=tile_n*TileN;n<min(p.n,(tile_n+1)*TileN);++n)
      emit(executor::PrefetchRange{p.b+std::int64_t(n)*pitch+p.k_begin,unsigned(2*p.k_count)});
  }
  __device__ static void Run(ServingGemmOperands const& p,int tile_m,int tile_n,char* shared) {
    auto* tile=Impl::Dense(p,tile_m,tile_n,shared);
#if TILEMEGA_TRACE_TASK
    auto epilogue_begin=TaskProfileNow(p.profile);
#endif
    auto finish=[&](auto op) {
      backend::ServingEpilogue<decltype(op)::value,16,TileN,(TILEMEGA_SWIGLU_U<TileN/2?TILEMEGA_SWIGLU_U:TileN/2)>::template RunFromTile<false>(
          tile,tile_m,tile_n,p.m,p.n,p.output_stride,p.partial_stride,p.output,p.residual,
          p.partial,p.argmax_value,p.argmax_index,p.norm_ss,p.ss_out,p.norm_k,p.norm_eps);
    };
    using Op=backend::ServingEpilogueOp;
    switch(p.epilogue) {
      case Op::kStore:finish(std::integral_constant<Op,Op::kStore>{});break;
      case Op::kResidual:finish(std::integral_constant<Op,Op::kResidual>{});break;
      case Op::kPartial:finish(std::integral_constant<Op,Op::kPartial>{});break;
      case Op::kArgmaxPartial:finish(std::integral_constant<Op,Op::kArgmaxPartial>{});break;
      case Op::kSwiGLU:finish(std::integral_constant<Op,Op::kSwiGLU>{});break;
    }
#if TILEMEGA_TRACE_TASK
    if(p.profile && ComputeThread()==0)p.profile->epilogue_ns=TaskProfileNow(p.profile)-epilogue_begin;
#endif
  }
};
}
