// Preserve the independently validated serving and DM runtime contracts.
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
#include <tilemega/Codegen/tasks/DmServingGemmTaskBody.h>
#else
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/ServingTaskProfile.h>

#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Codegen/executor/Prefetch.cuh>

#include <tilemega/Backend/ServingEpilogue.h>
#include <tilemega/Backend/ServingGemm.h>
#if TILEMEGA_TRACE_TASK
#include <tilemega/Backend/ServingProfiledMainloop.h>
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
#if TILEMEGA_TRACE_TASK
  ServingTaskProfile* profile=nullptr;
#endif
};

template <class Arch, int TileM, int TileN, int TileK, int Stages>
struct ServingGemmTaskBody {
  using Config = backend::ServingGemmConfig<Arch, TileM, TileN, TileK, Stages>;
  using Mainloop = typename Config::Mainloop;
  static constexpr int kThreads = Config::kThreads;
  static constexpr int kSharedBytes = Config::kSharedBytes;

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
#if TILEMEGA_TRACE_TASK
    cutlass::gemm::collective::ProfiledServingMainloop<Mainloop>{}(accum, gA,
#else
    Mainloop{}(accum, gA,
#endif
               gB, accum, k_iter, size<2>(gA), residue,
               ComputeThread(), shared
#if TILEMEGA_TRACE_TASK
               , FirstTileReadyProfile{p.profile}
#endif
               );
#endif
#if TILEMEGA_TRACE_TASK
    auto epilogue_begin=TaskProfileNow(p.profile);
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
#if TILEMEGA_TRACE_TASK
    if(p.profile)p.profile->epilogue_ns=TaskProfileNow(p.profile)-epilogue_begin;
#endif
  }
};

}  // namespace tilemega::codegen

#endif
