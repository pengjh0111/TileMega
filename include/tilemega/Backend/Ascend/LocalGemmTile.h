// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Ascend/GemmTile.h>

namespace tilemega::ascend {
template<class Copy> struct CopyToAiv0 : Copy {
  template<class Dst,class Src>
  __aicore__ inline void operator()(Dst const& dst,Src const& src,uint8_t unit=0) {
    Copy::operator()(dst,src,false,unit);
  }
};

// The caller owns the same-group AIV0 slot and its ready/consumed protocol.
// No scheduler or completion publication is hidden in this tile function.
template<class Input,class Output,unsigned TM=16,unsigned TN=128,unsigned TK=128>
struct LocalGemmTile {
  using Arch=Catlass::Arch::Ascend950;
  using Row=Catlass::layout::RowMajor;
  using Column=Catlass::layout::ColumnMajor;
  using CopyBase=Catlass::Gemm::Tile::PackedTileCopyTlaToUB<Arch,Input,Row,Input,Column,Output,Row>;
  struct Copy : CopyBase {
    template<class C> using CopyL0CToDst=CopyToAiv0<typename CopyBase::template CopyL0CToDst<C>>;
  };
  using Policy=Catlass::Gemm::MmadPingpong<Arch,true,false,1,false,2,2,2,2>;
  using L1=tla::Shape<tla::Int<TM>,tla::Int<TN>,tla::Int<TK>>;
  using L0=tla::Shape<tla::Int<TM>,tla::Int<TN>,tla::Int<32>>;
  using SharedResource=Catlass::Arch::Resource<Arch>;
  using TileMmad=Catlass::Gemm::Block::BlockMmadTla<Policy,L1,L0,Input,Input,Output,void,Copy>;

  __aicore__ static inline bool RunBorrowed(TileMmad& mm,GM_ADDR a,GM_ADDR b,
      AscendC::LocalTensor<Output> destination,unsigned m,unsigned n,unsigned k,
      unsigned tile_m,unsigned tile_n) {
    if(!m || !n || !k || k%16 || tile_m>=(m+TM-1)/TM || tile_n>=(n+TN-1)/TN)return false;
    const unsigned row=tile_m*TM,col=tile_n*TN;
    const unsigned rm=m-row<TM?m-row:TM,rn=n-col<TN?n-col:TN;
    AscendC::GlobalTensor<Input> ga,gb;ga.SetGlobalBuffer((__gm__ Input*)a);gb.SetGlobalBuffer((__gm__ Input*)b);
    auto ta=tla::MakeTensor(ga,tla::MakeLayout<Input,Row>(m,k),Catlass::Arch::PositionGM{});
    auto tb=tla::MakeTensor(gb,tla::MakeLayout<Input,Column>(k,n),Catlass::Arch::PositionGM{});
    auto ba=tla::GetTile(ta,tla::MakeCoord(row,0),tla::MakeShape(rm,k));
    auto bb=tla::GetTile(tb,tla::MakeCoord(0,col),tla::MakeShape(k,rn));
    // Fixpipe's BF16 tail occupies a padded physical column range; only rn
    // columns belong to the logical output and the caller must not expose pad.
    auto tc=tla::MakeTensor(destination,tla::MakeLayout<Output,Row>(rm,TN),Catlass::Arch::PositionUB{});
    constexpr unsigned alignment=32/sizeof(Output);
    auto bc=tla::GetTile(tc,tla::MakeCoord(0,0),tla::MakeShape(rm,(rn+alignment-1)/alignment*alignment));
    mm(ba,bb,bc,Catlass::GemmCoord(rm,rn,k));return true;
  }
};
} // namespace tilemega::ascend
