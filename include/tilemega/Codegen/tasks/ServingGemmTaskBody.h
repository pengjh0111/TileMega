// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/ServingTaskProfile.h>

#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Codegen/executor/Prefetch.cuh>

#include <tilemega/Backend/ServingEpilogue.h>
#include <tilemega/Backend/Mx/ServingGemmConfig.h>
#include <tilemega/Backend/Mx/ServingGemmMma.h>
#include <tilemega/Target/Maca/DevicePrimitives.h>
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
#if defined(TILEMEGA_MOE_GEMV) && TILEMEGA_MOE_GEMV
#error "the MoE GEMV body is not ported to xcore1000; build MoE plans with --moe-gemv 0"
#endif
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Codegen/tasks/DmEpilogueDispatch.cuh>
#include <tilemega/Backend/ServingDmEpilogue.h>
#include <tilemega/Backend/ServingDmGemm.h>
#include <tilemega/Backend/ServingConv.h>
#include <tilemega/Solver/DmGemmTraits.h>
#endif

#ifndef TILEMEGA_NONPAGED_TILED
#define TILEMEGA_NONPAGED_TILED 0
#endif

#include <type_traits>

namespace tilemega::codegen {

using codegen::executor::ComputeThread;
using codegen::executor::ComputeSync;
using codegen::executor::kComputeThreads;

struct ServingGemmOperands {
  tilemega::Bf16 const* a = nullptr;  // logical [M,K_total]
  tilemega::Bf16 const* b = nullptr;  // logical [N,K_total]
  tilemega::Bf16 const* residual = nullptr;
  tilemega::Bf16* output = nullptr;
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
  tilemega::Bf16 const* weight_base = nullptr;
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

/// xcore1000 serving GEMM TaskBody: the MMA mainloop of Backend/Mx fills the
/// FP32 tile and the shared epilogue rounds and stores it.
/// Epilogue dispatch shared by the MMA and GEMV bodies. Not inlined: inlined
/// into the body, the five RunFromTile instantiations raised the GEMV kernel
/// to 256 registers with a 924-byte spill and the MMA kernel to 222 with 184
/// bytes, while the mainloops alone need 192 and 80 with no stack (measured,
/// cucc -resource-usage). A call per tile is cheap next to its mainloop.
template <int TileM, int TileN, int SwigluU = TILEMEGA_SWIGLU_U>
__device__ __noinline__ void ServingEpilogueDispatch(float* tile, ServingGemmOperands const& p,
                                                     int tile_m, int tile_n) {
  auto finish = [&](auto op) {
    backend::ServingEpilogue<decltype(op)::value, TileM, TileN, SwigluU>::template RunFromTile<false>(
        tile, tile_m, tile_n, p.m, p.n, p.output_stride, p.partial_stride,
        p.output, p.residual, p.partial, p.argmax_value, p.argmax_index,
        p.norm_ss, p.ss_out, p.norm_k, p.norm_eps);
  };
  using Op = backend::ServingEpilogueOp;
  switch (p.epilogue) {
    case Op::kStore: finish(std::integral_constant<Op, Op::kStore>{}); break;
    case Op::kResidual: finish(std::integral_constant<Op, Op::kResidual>{}); break;
    case Op::kSwiGLU: finish(std::integral_constant<Op, Op::kSwiGLU>{}); break;
    case Op::kArgmaxPartial: finish(std::integral_constant<Op, Op::kArgmaxPartial>{}); break;
    case Op::kPartial: finish(std::integral_constant<Op, Op::kPartial>{}); break;
  }
}

template <class Arch, int TileM, int TileN, int TileK, int Stages>
struct ServingGemmTaskBody {
  using Config = backend::ServingGemmConfig<Arch, TileM, TileN, TileK, Stages>;
  using Mainloop = typename Config::Mainloop;
  using Mma = backend::ServingGemmMma<TileM, TileN, TileK, Stages>;
  static constexpr int kThreads = Config::kThreads;
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  static constexpr int kSharedBytes = solver::DmServingBF16SmemBytes(TileM, TileN, TileK, Stages);
  static constexpr int kTileColumns = TileN;
#else
  static constexpr int kSharedBytes = Config::kSharedBytes;
#endif
  static_assert(kThreads == kComputeThreads);

  template<class Emit>
  __device__ static void PrefetchRanges(ServingGemmOperands const& p,int tile_n,Emit emit) {
    int pitch=p.b_row_stride?p.b_row_stride:p.k_total;
    for(int n=tile_n*TileN;n<min(p.n,(tile_n+1)*TileN);++n)
      emit(executor::PrefetchRange{p.b+static_cast<long long>(n)*pitch+p.k_begin,
                                  static_cast<unsigned>(2*p.k_count)});
  }
  __device__ static void Run(ServingGemmOperands const& p, int tile_m,
                             int tile_n, char* shared) {
    if (!p.a || !p.b || p.k_begin < 0 || p.k_count <= 0 ||
        p.k_begin + p.k_count > p.k_total) {
      maca::Trap();
      return;
    }
    int a_pitch = p.a_row_stride ? p.a_row_stride : p.k_total;
    int b_pitch = p.b_row_stride ? p.b_row_stride : p.k_total;
    // 16-byte operand loads need 8-element aligned rows and K offsets.
    // Fragment-packed weights are addressed by whole K tiles.
    bool const tiled_bad = TILEMEGA_NONPAGED_TILED != 0 && (!p.weight_base || p.k_begin % TileK != 0);
    if (tiled_bad || a_pitch < p.k_total || b_pitch < p.k_total ||
        a_pitch % 8 != 0 || b_pitch % 8 != 0 || p.k_begin % 8 != 0) {
      maca::Trap();
      return;
    }
#if TILEMEGA_TRACE_TASK
    if(p.profile)FirstTileReadyProfile{p.profile}();
#endif
    float* tile = Mma::template Dense<TILEMEGA_NONPAGED_TILED != 0>(p, tile_m, tile_n, shared);
#if TILEMEGA_TRACE_TASK
    auto epilogue_begin=TaskProfileNow(p.profile);
#endif
    ServingEpilogueDispatch<TileM, TileN>(tile, p, tile_m, tile_n);
#if TILEMEGA_TRACE_TASK
    if(p.profile)p.profile->epilogue_ns=TaskProfileNow(p.profile)-epilogue_begin;
#endif
  }
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  template <class Spec>
  __device__ static void RunDm(ServingGemmOperands const& p, int tile_m, int tile_n, char* shared) {
    auto resolved = p;
    if (!backend::ResolveDmMoeTile(resolved, tile_m, TileM)) return;
    RunDmResolved<Spec>(resolved, tile_m, tile_n, shared);
  }
  /// Every DM GEMM form produces the FP32 tile for the DM epilogue: implicit
  /// GEMM over NHWC (im2col), or [M,K] rows (dense, gathered, SCA-scaled,
  /// expert-indirect B after ResolveDmMoeTile).
  template <class Spec>
  __device__ static void RunDmResolved(ServingGemmOperands const& p, int tile_m, int tile_n, char* shared) {
    constexpr bool kTiled = TILEMEGA_NONPAGED_TILED != 0;
    float* tile = nullptr;
    if (p.access.a == DmAAccess::kIm2Col) {
      if (!p.convolutions || p.access.conv == kDmNoIndex || p.k_begin % TileK || p.k_count % TileK ||
          !p.conv_iteration.iterations || (kTiled && !p.weight_base)) {
        maca::Trap();
        return;
      }
      auto const& conv = p.convolutions[p.access.conv];
      if (!p.dm_buffers.layouts || conv.input_layout >= p.dm_buffers.count) {
        maca::Trap();
        return;
      }
      auto const& layout = p.dm_buffers.layouts[conv.input_layout];
      auto const& geometry = p.conv_iteration;
      std::uint64_t const begin = p.k_begin / TileK;
      int const iterations = p.k_count / TileK;
      if (geometry.tile_k != TileK || begin + iterations > geometry.iterations || !conv.p || !conv.q) {
        maca::Trap();
        return;
      }
      tile = backend::ServingConv<Arch, TileM, TileN, TileK, Stages>::template Run<kTiled>(
          p, conv, layout, geometry, tile_m, tile_n, begin, iterations, shared);
    } else {
      // Argument checks live in ServingDmGemm::Dense.
      tile = backend::ServingDmGemm<Arch, TileM, TileN, TileK, Stages>::Dense(p, tile_m, tile_n, shared, kTiled);
    }
    backend::ServingDmEpilogue<Arch, Spec, TileM, TileN, false>::RunFromTile(tile, DmEpilogueOperands(p), tile_m,
                                                                            tile_n);
  }
#endif
};

}  // namespace tilemega::codegen
