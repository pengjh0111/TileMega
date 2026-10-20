// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <catlass/catlass.hpp>
#include <catlass/arch/arch.hpp>
#include <catlass/arch/resource.hpp>
#include <catlass/gemm/block/block_mmad.hpp>
#include <catlass/gemm/dispatch_policy.hpp>
#include <catlass/gemm/gemm_type.hpp>
#include <catlass/layout/layout.hpp>
#include <tla/tensor.hpp>
#include <tla/layout.hpp>

namespace tilemega::ascend {
// The caller supplies one CTG tile coordinate. This body never swizzles or
// queries BlockIdx; ordinary K accumulation stays in this AIC's BlockMmad.
template<class Input, class Output> class GemmTile {
  using Arch = Catlass::Arch::Ascend950;
  using Row = Catlass::layout::RowMajor;
  using Policy = Catlass::Gemm::MmadPingpong<Arch, true, false, 1, false, 2, 2, 2, 2>;
  using L1 = tla::Shape<tla::Int<32>, tla::Int<64>, tla::Int<64>>;
  using L0 = tla::Shape<tla::Int<32>, tla::Int<64>, tla::Int<32>>;
  using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, Input, Row, Input, Row, Output, Row, void>;
  using Mmad = Catlass::Gemm::Block::BlockMmadTla<Policy, L1, L0, Input, Input, Output, void, Copy>;
  Catlass::Arch::Resource<Arch> resource_;
  Mmad mm_;
 public:
  __aicore__ inline GemmTile() : mm_(resource_) {}
  __aicore__ inline void RunTask(GM_ADDR a, GM_ADDR b, GM_ADDR c, unsigned m, unsigned n,
                                 unsigned k, unsigned tile_m, unsigned tile_n) {
    AscendC::GlobalTensor<Input> ga, gb;
    AscendC::GlobalTensor<Output> gc;
    ga.SetGlobalBuffer((__gm__ Input*)a); gb.SetGlobalBuffer((__gm__ Input*)b);
    gc.SetGlobalBuffer((__gm__ Output*)c);
    auto ta = tla::MakeTensor(ga, tla::MakeLayout<Input, Row>(m, k), Catlass::Arch::PositionGM{});
    auto tb = tla::MakeTensor(gb, tla::MakeLayout<Input, Row>(k, n), Catlass::Arch::PositionGM{});
    auto tc = tla::MakeTensor(gc, tla::MakeLayout<Output, Row>(m, n), Catlass::Arch::PositionGM{});
    const unsigned row = tile_m*32, col = tile_n*64;
    const unsigned rm = m-row < 32 ? m-row : 32, rn = n-col < 64 ? n-col : 64;
    auto ba = tla::GetTile(ta, tla::MakeCoord(row, 0u), tla::MakeShape(rm, k));
    auto bb = tla::GetTile(tb, tla::MakeCoord(0u, col), tla::MakeShape(k, rn));
    auto bc = tla::GetTile(tc, tla::MakeCoord(row, col), tla::MakeShape(rm, rn));
    mm_(ba, bb, bc, Catlass::GemmCoord(rm, rn, k));
  }
};
} // namespace tilemega::ascend
