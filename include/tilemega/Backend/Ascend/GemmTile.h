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
#include <type_traits>

namespace tilemega::ascend {
// The caller supplies one CTG tile coordinate. This body never swizzles or
// queries BlockIdx; ordinary K accumulation stays in this AIC's BlockMmad.
template<class Input, class Output, unsigned TM=32, unsigned TN=64,
         unsigned TK=64, bool WeightNK=false> class GemmTile {
  using Arch = Catlass::Arch::Ascend950;
  using Row = Catlass::layout::RowMajor;
  using WeightLayout = std::conditional_t<WeightNK,Catlass::layout::ColumnMajor,Row>;
  using Policy = Catlass::Gemm::MmadPingpong<Arch, true, false, 1, false, 2, 2, 2, 2>;
  using L1 = tla::Shape<tla::Int<TM>, tla::Int<TN>, tla::Int<TK>>;
  using L0 = tla::Shape<tla::Int<TM>, tla::Int<TN>, tla::Int<32>>;
  using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, Input, Row, Input, WeightLayout, Output, Row, void>;
  using Mmad = Catlass::Gemm::Block::BlockMmadTla<Policy, L1, L0, Input, Input, Output, void, Copy>;
  Catlass::Arch::Resource<Arch> resource_;
  Mmad mm_;
 public:
  using SharedResource = Catlass::Arch::Resource<Arch>;
  using TileMmad = Mmad;
  __aicore__ inline GemmTile() : mm_(resource_) {}
  __aicore__ inline void RunTask(GM_ADDR a, GM_ADDR b, GM_ADDR c, unsigned m, unsigned n,
                                 unsigned k, unsigned tile_m, unsigned tile_n) {
    RunBorrowed(mm_,a,b,c,m,n,k,tile_m,tile_n);
  }
  // The caller must keep exactly one borrowed MMAD alive in this resource.
  // Its destructor drains internal flags before another dtype reuses the banks.
  __aicore__ static inline void RunBorrowed(TileMmad& mm,GM_ADDR a,GM_ADDR b,GM_ADDR c,
      unsigned m,unsigned n,unsigned k,unsigned tile_m,unsigned tile_n) {
    AscendC::GlobalTensor<Input> ga, gb;
    AscendC::GlobalTensor<Output> gc;
    ga.SetGlobalBuffer((__gm__ Input*)a); gb.SetGlobalBuffer((__gm__ Input*)b);
    gc.SetGlobalBuffer((__gm__ Output*)c);
    auto ta = tla::MakeTensor(ga, tla::MakeLayout<Input, Row>(m, k), Catlass::Arch::PositionGM{});
    auto tb = tla::MakeTensor(gb, tla::MakeLayout<Input, WeightLayout>(k, n), Catlass::Arch::PositionGM{});
    auto tc = tla::MakeTensor(gc, tla::MakeLayout<Output, Row>(m, n), Catlass::Arch::PositionGM{});
    const unsigned row = tile_m*TM, col = tile_n*TN;
    const unsigned rm = m-row < TM ? m-row : TM, rn = n-col < TN ? n-col : TN;
    auto ba = tla::GetTile(ta, tla::MakeCoord(row, 0u), tla::MakeShape(rm, k));
    auto bb = tla::GetTile(tb, tla::MakeCoord(0u, col), tla::MakeShape(k, rn));
    auto bc = tla::GetTile(tc, tla::MakeCoord(row, col), tla::MakeShape(rm, rn));
    mm(ba, bb, bc, Catlass::GemmCoord(rm, rn, k));
  }
};
} // namespace tilemega::ascend
