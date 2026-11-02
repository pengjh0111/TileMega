// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Ascend/GemmTile.h>
#include <tilemega/Target/Ascend/DevicePrimitives.h>
#include <cstddef>

namespace tilemega::ascend {struct GemmSessionPlacement {}; }
// CANN9.1 has no device overload for standard placement new. This tagged
// overload only begins a lifetime in existing scalar storage; it allocates none.
__aicore__ inline void* operator new(std::size_t,tilemega::ascend::GemmSessionPlacement,void* p) noexcept {return p;}

namespace tilemega::ascend {
// One active recipe owns these banks/flags. FinishRecipe must precede any
// other MMAD using the resource. No operand residency or asynchronous external
// completion is implied: every SubmitTile still waits for its FIX payload.
template<class Gemm> class GemmSession {
  using Mmad=typename Gemm::TileMmad;
  union Slot {
    unsigned char empty;
    Mmad mm;
    __aicore__ inline Slot():empty(0) {}
    __aicore__ inline ~Slot() {}
  } slot_;
  typename Gemm::SharedResource& resource_;
  bool active_=false;
 public:
  __aicore__ inline explicit GemmSession(typename Gemm::SharedResource& resource):resource_(resource) {}
  GemmSession(GemmSession const&)=delete;
  GemmSession& operator=(GemmSession const&)=delete;
  __aicore__ inline void BeginRecipe() {
    if(!active_) {new (GemmSessionPlacement{},&slot_.mm) Mmad(resource_);active_=true;}
  }
  __aicore__ inline void SubmitTile(GM_ADDR a,GM_ADDR b,GM_ADDR c,unsigned m,unsigned n,
      unsigned k,unsigned tile_m,unsigned tile_n) {
    BeginRecipe();Gemm::RunBorrowed(slot_.mm,a,b,c,m,n,k,tile_m,tile_n);
    CompleteDma<AscendC::HardEvent::FIX_S>();
  }
  __aicore__ inline void FinishRecipe() {
    if(active_) {slot_.mm.~Mmad();active_=false;}
  }
  __aicore__ inline ~GemmSession() {FinishRecipe();}
};
} // namespace tilemega::ascend
