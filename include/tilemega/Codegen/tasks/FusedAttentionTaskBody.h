// SPDX-License-Identifier: BSD-3-Clause
#pragma once
// Prefill attention TaskBody. On xcore1000 the implementation is the MMA body
// of Codegen/tasks/mx/PrefillAttentionMma.h; the template signature is the
// one the harness and the solver key on.
#include <tilemega/Codegen/tasks/mx/PrefillAttentionMma.h>
#include <tilemega/Target/ArchDispatch.h>

#include <type_traits>

// The plan's kv_layout (MX4 prompt §3.4): 1 = `vt` (K rows, V^T), the layout
// of xcore1000 dense plans whose decode is the MX-4 static program; 0 = rows.
#ifndef TILEMEGA_KV_LAYOUT_VT
#define TILEMEGA_KV_LAYOUT_VT 0
#endif

namespace tilemega::codegen {

template<class Arch,int kHeadDim,int kQPerKV,int kTokens,
         int kQRows,int kKvTile,bool kQkNorm>
struct FusedAttentionTaskBody {
  static_assert(std::is_same_v<Arch,arch::Xcore1000>);
  using Impl=mx::PrefillAttentionMma<kHeadDim,kQPerKV,kTokens,kQRows,kKvTile,kQkNorm,
                                     TILEMEGA_KV_LAYOUT_VT!=0>;
  using Element=tilemega::Bf16;
  using SharedStorage=typename Impl::SharedStorage;
  static constexpr int kQueryExtent=Impl::kQueryExtent;
  template<class Emit>
  __device__ static void PrefetchRanges(ServingAttentionOperands const& p,int batch,int group,
                                      int block,unsigned history_mask,Emit emit) {
    int begin=block*p.block_extent,end=min(p.past,(block+1)*p.block_extent);
    if(begin>=end)return;
    auto offset=(static_cast<long long>(batch)*p.heads_kv+group)*p.capacity*kHeadDim+
                static_cast<long long>(begin)*kHeadDim;
    unsigned bytes=2u*(end-begin)*kHeadDim;
    if(history_mask&1)emit(executor::PrefetchRange{p.key_cache+offset,bytes});
    // V^T keeps a position's values D rows apart: no contiguous range.
    if((history_mask&2) && !TILEMEGA_KV_LAYOUT_VT)emit(executor::PrefetchRange{p.value_cache+offset,bytes});
  }
  __device__ static void Run(ServingAttentionOperands const& p,SharedStorage& s,
                             int b,int g,int qb,int c) {
    Impl::Run(p,s,b,g,qb,c);
  }
};

}  // namespace tilemega::codegen
