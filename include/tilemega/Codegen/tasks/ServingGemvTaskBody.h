// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ServingGemv.h>
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
namespace tilemega::codegen {
template<class Arch,int TileN,int TileK,bool Tiled=false>
struct ServingGemvTaskBody {
  using Impl=backend::ServingGemv<Arch,TileN,TileK,Tiled>;
  static constexpr int kThreads=128,kSharedBytes=Impl::kSharedBytes;
  __device__ static void Run(ServingGemmOperands const& p,int tile_m,int tile_n,char* shared) {
    if constexpr(TileN<32)if(p.epilogue==backend::ServingEpilogueOp::kSwiGLU || p.ss_out) {
      asm volatile("trap;");return;
    }
    auto* tile=reinterpret_cast<float*>(shared);
    Impl::Accumulate(p,tile_m,tile_n,tile);
    auto finish=[&](auto op) {
      backend::ServingEpilogue<decltype(op)::value,16,TileN>::template RunFromTile<true>(
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
        if constexpr(TileN>=32)finish(std::integral_constant<Op,Op::kSwiGLU>{});
        else asm volatile("trap;");
        break;
    }
  }
};
}
