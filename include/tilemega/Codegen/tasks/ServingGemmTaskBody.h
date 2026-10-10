// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Codegen/executor/Prefetch.cuh>

#include <tilemega/Backend/ServingEpilogue.h>
#include <tilemega/Backend/ServingGemm.h>
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Codegen/tasks/DmEpilogueDispatch.cuh>
#include <tilemega/Backend/ServingDmEpilogue.h>
#include <tilemega/Backend/ServingConv.h>
#include <tilemega/Backend/ServingGemv.h>
#ifndef TILEMEGA_MOE_GEMV
#define TILEMEGA_MOE_GEMV 0
#endif
#endif

#ifndef TILEMEGA_NONPAGED_TILED
#define TILEMEGA_NONPAGED_TILED 0
#endif
#if TILEMEGA_NONPAGED_TILED
#include <tilemega/Backend/ServingTiledMainloop.h>
#endif

#include <cute/tensor.hpp>
#include <type_traits>

namespace tilemega::codegen {

using codegen::executor::ComputeThread;
using codegen::executor::ComputeSync;
using codegen::executor::kComputeThreads;

struct ServingGemmOperands {
  cutlass::bfloat16_t const* a = nullptr;  // logical [M,K_total]
  cutlass::bfloat16_t const* b = nullptr;  // logical [N,K_total]
  cutlass::bfloat16_t const* residual = nullptr;
  cutlass::bfloat16_t* output = nullptr;
  float* partial = nullptr;
  float* argmax_value = nullptr;
  int* argmax_index = nullptr;
  int m = 0, n = 0, k_total = 0;
  int k_begin = 0, k_count = 0;
  int output_stride = 0;
  int partial_stride = 0;
  // Physical row pitches include zero-filled vector-alignment padding when
  // logical K has a residue.  Serving weights are packed once with this pitch.
  int a_row_stride = 0, b_row_stride = 0;
  cutlass::bfloat16_t const* weight_base = nullptr;
  int k_total_full = 0;
  float const* norm_ss = nullptr;
  float* ss_out = nullptr;
  int norm_k = 0;
  float norm_eps = 0.0f;
  backend::ServingEpilogueOp epilogue = backend::ServingEpilogueOp::kStore;
  void const* tensor_map = nullptr;
  int tensor_k_begin = 0;
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  DmGemmAccess access{};
  DmEpilogueChain chain{};
  ConvDesc const* convolutions = nullptr;
  backend::ConvIterationGeometry conv_iteration{};
  void const* binding = nullptr;
  void const* rows = nullptr;
  float const* a_scale = nullptr;
  DmBufferView dm_buffers{};
  backend::DmMoeRows moe{};
  bool dm_partial_rows = false;
#endif
};

#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
__device__ inline backend::DmEpilogueArguments DmEpilogueOperands(
    ServingGemmOperands const& p) {
  backend::DmEpilogueArguments result;
  result.buffers = p.dm_buffers;
  result.chain = p.chain;
  result.write = p.access.write;
  result.output = p.output;
  result.m = p.m;
  result.n = p.n;
  result.output_stride = p.output_stride;
  result.norm_width = p.norm_k;
  result.norm_eps = p.norm_eps;
  result.image_rows = p.access.rows_per_batch;
  result.moe=p.moe;
  result.partial_rows=p.dm_partial_rows;
  result.routing_topk=p.access.routing_topk;
  return result;
}
#endif

template <class Arch, int TileM, int TileN, int TileK, int Stages>
struct ServingGemmTaskBody {
  using Config = backend::ServingGemmConfig<Arch, TileM, TileN, TileK, Stages>;
  using Mainloop = typename Config::Mainloop;
  static constexpr int kThreads = Config::kThreads;
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  static constexpr int kSharedBytes=solver::DmServingBF16SmemBytes(TileM,TileN,TileK,Stages);
#else
  static constexpr int kSharedBytes = Config::kSharedBytes;
#endif
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  static constexpr int kTileColumns = TileN;
#endif

  template<class Emit>
  __device__ static void PrefetchRanges(ServingGemmOperands const& p,int tile_n,Emit emit) {
#if TILEMEGA_NONPAGED_TILED
    int const kt=(p.k_total_full+TileK-1)/TileK;
    emit(executor::PrefetchRange{p.weight_base+
        (std::size_t(tile_n)*kt+p.k_begin/TileK)*TileN*TileK,
        static_cast<unsigned>(((p.k_count+TileK-1)/TileK)*TileN*TileK*2)});
#else
    int pitch=p.b_row_stride?p.b_row_stride:p.k_total;
    for(int n=tile_n*TileN;n<min(p.n,(tile_n+1)*TileN);++n)
      emit(executor::PrefetchRange{p.b+static_cast<long long>(n)*pitch+p.k_begin,
                                  static_cast<unsigned>(2*p.k_count)});
#endif
  }
  __device__ static void Run(ServingGemmOperands const& p, int tile_m,
                             int tile_n, char* shared) {
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
    if(p.access.a==DmAAccess::kIm2Col || p.access.a==DmAAccess::kRowGather ||
       p.access.b==DmBAccess::kExpertIndirect || p.a_scale) {
      if(p.epilogue!=backend::ServingEpilogueOp::kPartial || !p.partial) {
        asm volatile("trap;");return;
      }
      auto partial=p;
      partial.output=reinterpret_cast<cutlass::bfloat16_t*>(p.partial);
      partial.output_stride=p.partial_stride;
      partial.chain={};partial.chain.store_rounding=DmRounding::kFP32;
      partial.access.write={};
      partial.dm_partial_rows=true;
      using Spec=DmEpilogueSpec<DmEpilogueProgram<>,DmWriteKind::kDense,1,DmRounding::kFP32>;
      RunDm<Spec>(partial,tile_m,tile_n,shared);
      return;
    }
    if constexpr(TileN==16 || TileK<64) {
      auto* tile=Config::Dense(p,tile_m,tile_n,shared,TILEMEGA_NONPAGED_TILED!=0);
      auto finish=[&](auto op) {
        backend::ServingEpilogue<decltype(op)::value,TileM,TileN>::template RunFromTile<false>(
            tile,tile_m,tile_n,p.m,p.n,p.output_stride,p.partial_stride,
            p.output,p.residual,p.partial,p.argmax_value,p.argmax_index,
            p.norm_ss,p.ss_out,p.norm_k,p.norm_eps);
      };
      switch(p.epilogue) {
        case backend::ServingEpilogueOp::kStore:finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kStore>{});break;
        case backend::ServingEpilogueOp::kResidual:finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kResidual>{});break;
        case backend::ServingEpilogueOp::kSwiGLU:
          if constexpr(TileN%32==0)finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kSwiGLU>{});
          else asm volatile("trap;");break;
        case backend::ServingEpilogueOp::kArgmaxPartial:finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kArgmaxPartial>{});break;
        case backend::ServingEpilogueOp::kPartial:finish(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kPartial>{});break;
      }
    }else {
#endif
    using namespace cute;
    if (!p.a || !p.b || p.k_begin < 0 || p.k_count <= 0 ||
        p.k_begin + p.k_count > p.k_total) {
      asm volatile("trap;");
      return;
    }
    int a_pitch = p.a_row_stride ? p.a_row_stride : p.k_total;
    int b_pitch = p.b_row_stride ? p.b_row_stride : p.k_total;
    int copy_k_count = (p.k_count + 7) & ~7;
    if (a_pitch < p.k_total || b_pitch < p.k_total ||
        a_pitch % 8 != 0 || b_pitch % 8 != 0 || p.k_begin % 8 != 0 ||
        p.k_begin + copy_k_count > a_pitch ||
        p.k_begin + copy_k_count > b_pitch) {
      asm volatile("trap;");
      return;
    }
    constexpr auto tile_shape = typename Mainloop::TileShape{};
    typename Mainloop::TiledMma mma;
    auto accum = partition_fragment_C(mma, take<0, 2>(tile_shape));
    clear(accum);
#if TILEMEGA_NONPAGED_TILED
    backend::ServingTiledMainloop<Arch,Config,TileM,TileN,TileK,Stages>::Run(
        p,tile_m,tile_n,shared,accum);
#else
    auto dA = make_stride(int64_t(a_pitch), _1{},
                          int64_t(p.m) * a_pitch);
    auto dB = make_stride(int64_t(b_pitch), _1{},
                          int64_t(p.n) * b_pitch);
    auto a = make_tensor(make_gmem_ptr(p.a + p.k_begin),
                         make_shape(p.m, copy_k_count, 1), dA);
    auto b = make_tensor(make_gmem_ptr(p.b + p.k_begin),
                         make_shape(p.n, copy_k_count, 1), dB);
    auto coordinate = make_coord(tile_m, tile_n, _, 0);
    auto gA = local_tile(a(_, _, 0), tile_shape,
                         take<0, 3>(coordinate), Step<_1, X, _1>{});
    auto gB = local_tile(b(_, _, 0), tile_shape,
                         take<0, 3>(coordinate), Step<X, _1, _1>{});
    auto residue = make_tuple(p.m - size<0>(gA) * tile_m,
                              p.n - size<0>(gB) * tile_n,
                              copy_k_count - size<1>(gA) * size<2>(gA));
    auto k_iter = make_coord_iterator(shape<2>(gA));
    Mainloop{}(accum, gA, gB, accum, k_iter, size<2>(gA), residue,
               ComputeThread(), shared);
#endif
    auto finish = [&](auto op) {
      backend::ServingEpilogue<decltype(op)::value, TileM, TileN>::Run(
          accum, mma, shared, tile_m, tile_n, p.m, p.n,
          p.output_stride, p.partial_stride, p.output, p.residual, p.partial,
          p.argmax_value, p.argmax_index,p.norm_ss,p.ss_out,p.norm_k,p.norm_eps);
    };
    switch (p.epilogue) {
      case backend::ServingEpilogueOp::kStore:
        finish(std::integral_constant<backend::ServingEpilogueOp,
               backend::ServingEpilogueOp::kStore>{}); break;
      case backend::ServingEpilogueOp::kResidual:
        finish(std::integral_constant<backend::ServingEpilogueOp,
               backend::ServingEpilogueOp::kResidual>{}); break;
      case backend::ServingEpilogueOp::kSwiGLU:
        finish(std::integral_constant<backend::ServingEpilogueOp,
               backend::ServingEpilogueOp::kSwiGLU>{}); break;
      case backend::ServingEpilogueOp::kArgmaxPartial:
        finish(std::integral_constant<backend::ServingEpilogueOp,
               backend::ServingEpilogueOp::kArgmaxPartial>{}); break;
      case backend::ServingEpilogueOp::kPartial:
        finish(std::integral_constant<backend::ServingEpilogueOp,
               backend::ServingEpilogueOp::kPartial>{}); break;
    }
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
    }
#endif
  }
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  template <class Spec>
  __device__ static void RunDm(ServingGemmOperands const& p, int tile_m,
                             int tile_n, char* shared) {
    auto resolved=p;
    if(!backend::ResolveDmMoeTile(resolved,tile_m,TileM))return;
    RunDmResolved<Spec>(resolved,tile_m,tile_n,shared);
  }
  template <class Spec>
  __device__ static void RunDmResolved(ServingGemmOperands const& p, int tile_m,
                                     int tile_n, char* shared) {
#if TILEMEGA_MOE_GEMV
    if constexpr(TileN<=32 && TileK>=32) {
      using Gemv=backend::ServingGemv<Arch,TileM,TileN,TileK>;
      if(Gemv::Supported(p,tile_m)) {
        auto* tile=Gemv::Dense(p,tile_m,tile_n,shared,TILEMEGA_NONPAGED_TILED!=0);
        backend::ServingDmEpilogue<Arch,Spec,TileM,TileN,false>::RunFromTile(
            tile,DmEpilogueOperands(p),tile_m,tile_n);
        return;
      }
    }
#endif
    if((p.a_scale && p.access.a==DmAAccess::kDense) ||
       p.access.a==DmAAccess::kRowGather || p.access.b==DmBAccess::kExpertIndirect) {
      auto* tile=backend::ServingDmGemm<Arch,TileM,TileN,TileK,Stages>::Dense(
          p,tile_m,tile_n,shared,TILEMEGA_NONPAGED_TILED!=0);
      backend::ServingDmEpilogue<Arch,Spec,TileM,TileN,false>::RunFromTile(
          tile,DmEpilogueOperands(p),tile_m,tile_n);
      return;
    }
    if(p.access.a==DmAAccess::kIm2Col) {
      if(!p.convolutions || p.access.conv==kDmNoIndex ||
         p.k_begin%TileK || p.k_count%TileK || !p.conv_iteration.iterations) {
        asm volatile("trap;");return;
      }
      auto const& conv=p.convolutions[p.access.conv];
      if(!p.dm_buffers.layouts || conv.input_layout>=p.dm_buffers.count) {
        asm volatile("trap;");return;
      }
      auto* tile=backend::ServingConv<Arch,TileM,TileN,TileK,Stages>::template
          Run<TILEMEGA_NONPAGED_TILED!=0>(p,conv,p.dm_buffers.layouts[conv.input_layout],
              p.conv_iteration,tile_m,tile_n,p.k_begin/TileK,p.k_count/TileK,shared);
      backend::ServingDmEpilogue<Arch,Spec,TileM,TileN,false>::RunFromTile(
          tile,DmEpilogueOperands(p),tile_m,tile_n);
      return;
    }
    if constexpr(TileN==16 || TileK<64) {
      auto* tile=Config::Dense(p,tile_m,tile_n,shared,TILEMEGA_NONPAGED_TILED!=0);
      backend::ServingDmEpilogue<Arch,Spec,TileM,TileN,false>::RunFromTile(
          tile,DmEpilogueOperands(p),tile_m,tile_n);
    }else {
    using namespace cute;
    if (!p.a || !p.b || p.k_begin < 0 || p.k_count <= 0 ||
        p.k_begin + p.k_count > p.k_total) {
      asm volatile("trap;");
      return;
    }
    int a_pitch = p.a_row_stride ? p.a_row_stride : p.k_total;
    int b_pitch = p.b_row_stride ? p.b_row_stride : p.k_total;
    int copy_k_count = (p.k_count + 7) & ~7;
    if (a_pitch < p.k_total || b_pitch < p.k_total ||
        a_pitch % 8 != 0 || b_pitch % 8 != 0 || p.k_begin % 8 != 0 ||
        p.k_begin + copy_k_count > a_pitch ||
        p.k_begin + copy_k_count > b_pitch) {
      asm volatile("trap;");
      return;
    }
    constexpr auto tile_shape = typename Mainloop::TileShape{};
    typename Mainloop::TiledMma mma;
    auto accum = partition_fragment_C(mma, take<0, 2>(tile_shape));
    clear(accum);
#if TILEMEGA_NONPAGED_TILED
    backend::ServingTiledMainloop<Arch,Config,TileM,TileN,TileK,Stages>::Run(
        p,tile_m,tile_n,shared,accum);
#else
    auto dA = make_stride(int64_t(a_pitch), _1{},
                          int64_t(p.m) * a_pitch);
    auto dB = make_stride(int64_t(b_pitch), _1{},
                          int64_t(p.n) * b_pitch);
    auto a = make_tensor(make_gmem_ptr(p.a + p.k_begin),
                         make_shape(p.m, copy_k_count, 1), dA);
    auto b = make_tensor(make_gmem_ptr(p.b + p.k_begin),
                         make_shape(p.n, copy_k_count, 1), dB);
    auto coordinate = make_coord(tile_m, tile_n, _, 0);
    auto gA = local_tile(a(_, _, 0), tile_shape,
                         take<0, 3>(coordinate), Step<_1, X, _1>{});
    auto gB = local_tile(b(_, _, 0), tile_shape,
                         take<0, 3>(coordinate), Step<X, _1, _1>{});
    auto residue = make_tuple(p.m - size<0>(gA) * tile_m,
                              p.n - size<0>(gB) * tile_n,
                              copy_k_count - size<1>(gA) * size<2>(gA));
    auto k_iter = make_coord_iterator(shape<2>(gA));
    Mainloop{}(accum, gA, gB, accum, k_iter, size<2>(gA), residue,
               ComputeThread(), shared);
#endif
    backend::ServingDmEpilogue<Arch, Spec, TileM, TileN>::Run(
        accum, mma, shared, DmEpilogueOperands(p), tile_m, tile_n);
    }
  }
#endif

};

}  // namespace tilemega::codegen
