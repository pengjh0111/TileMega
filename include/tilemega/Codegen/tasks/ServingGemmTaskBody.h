// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/ServingEpilogue.h>
#include <tilemega/Backend/ServingGemm.h>

#include <cute/tensor.hpp>

namespace tilemega::codegen {

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
  // Physical row pitches include zero-filled vector-alignment padding when
  // logical K has a residue.  Serving weights are packed once with this pitch.
  int a_row_stride = 0, b_row_stride = 0;
};

template <class Arch, int TileM, int TileN, int TileK, int Stages,
          backend::ServingEpilogueOp Op>
struct ServingGemmTaskBody {
  using Config = backend::ServingGemmConfig<Arch, TileM, TileN, TileK, Stages>;
  using Mainloop = typename Config::Mainloop;
  static constexpr int kThreads = Config::kThreads;
  static constexpr int kSharedBytes = Config::kSharedBytes;

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
    auto dA = make_stride(int64_t(a_pitch), _1{},
                          int64_t(p.m) * a_pitch);
    auto dB = make_stride(int64_t(b_pitch), _1{},
                          int64_t(p.n) * b_pitch);
    auto a = make_tensor(make_gmem_ptr(p.a + p.k_begin),
                         make_shape(p.m, copy_k_count, 1), dA);
    auto b = make_tensor(make_gmem_ptr(p.b + p.k_begin),
                         make_shape(p.n, copy_k_count, 1), dB);
    constexpr auto tile_shape = typename Mainloop::TileShape{};
    auto coordinate = make_coord(tile_m, tile_n, _, 0);
    auto gA = local_tile(a(_, _, 0), tile_shape,
                         take<0, 3>(coordinate), Step<_1, X, _1>{});
    auto gB = local_tile(b(_, _, 0), tile_shape,
                         take<0, 3>(coordinate), Step<X, _1, _1>{});
    auto residue = make_tuple(p.m - size<0>(gA) * tile_m,
                              p.n - size<0>(gB) * tile_n,
                              copy_k_count - size<1>(gA) * size<2>(gA));
    typename Mainloop::TiledMma mma;
    auto accum = partition_fragment_C(mma, take<0, 2>(tile_shape));
    clear(accum);
    auto k_iter = make_coord_iterator(shape<2>(gA));
    Mainloop{}(accum, gA, gB, accum, k_iter, size<2>(gA), residue,
               int(threadIdx.x), shared);
    backend::ServingEpilogue<Op, TileM, TileN>::Run(
        accum, mma, shared, tile_m, tile_n, p.m, p.n,
        p.output_stride, p.output, p.residual, p.partial,
        p.argmax_value, p.argmax_index);
  }
};

}  // namespace tilemega::codegen
