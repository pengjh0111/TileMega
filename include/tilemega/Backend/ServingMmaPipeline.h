// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/ArchDispatch.h>
#include <cute/tensor.hpp>

#ifndef TILEMEGA_MMA_REG_PIPE
#define TILEMEGA_MMA_REG_PIPE 0
#endif

namespace tilemega::backend {
// The shared tile is already ready and remains owned until this returns.
// Load k+1 before MMA(k); only two operand slices are live, and no load crosses
// a page/stage boundary. MMA order and the epilogue's rounding points are fixed.
template<class Arch,class Config,class Mma,class Accum,class RA,class RB,
         class SA,class SB,class DA,class DB>
__device__ __forceinline__ void ServingMmaRegisterPipeline(
    Mma const& mma,Accum& accum,RA& ra,RB& rb,
    SA const& sa,SB const& sb,DA& da,DB& db) {
  using namespace cute;
  static_assert(arch::Caps<Arch>::kBf16TensorCore);
  constexpr int Steps=decltype(size<2>(ra))::value;
  static_assert(Steps>0 && Steps==decltype(size<2>(rb))::value);
  copy(typename Config::SmemCopyAtom{},sa(_,_,Int<0>{}),da(_,_,Int<0>{}));
  copy(typename Config::SmemCopyAtomB{},sb(_,_,Int<0>{}),db(_,_,Int<0>{}));
  for_each(make_int_sequence<Steps>{},[&](auto k) {
    if constexpr(decltype(k)::value+1<Steps) {
      constexpr int Next=decltype(k)::value+1;
      copy(typename Config::SmemCopyAtom{},sa(_,_,Int<Next>{}),da(_,_,Int<Next>{}));
      copy(typename Config::SmemCopyAtomB{},sb(_,_,Int<Next>{}),db(_,_,Int<Next>{}));
    }
    gemm(mma,ra(_,_,k),rb(_,_,k),accum);
  });
}
} // namespace tilemega::backend
