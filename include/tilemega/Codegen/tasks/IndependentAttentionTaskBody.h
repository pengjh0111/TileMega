// SPDX-License-Identifier: BSD-3-Clause
#pragma once
// Decode attention TaskBody (one token per request). On xcore1000 the `simt`
// implementation reads the KV cache directly (Codegen/tasks/mx/
// DecodeAttentionSimt.h); `mma16` is not implemented yet and is rejected.
#include <tilemega/Backend/ServingAttentionImpl.h>
#include <tilemega/Codegen/tasks/AttentionPageLayout.h>
#include <tilemega/Codegen/tasks/mx/DecodeAttentionSimt.h>
#include <tilemega/Codegen/tasks/mx/PrefillAttentionMma.h>
#include <tilemega/Target/ArchDispatch.h>

#include <type_traits>

namespace tilemega::codegen {

template<class Arch,int D,int Q,bool Norm,int PageBytes=8192,
    AttentionPagePolicy Policy=AttentionPagePolicy::WarpPrivate,int Buffers=2>
struct IndependentAttentionTaskBody {
  static_assert(std::is_same_v<Arch,arch::Xcore1000>);
  static_assert(TILEMEGA_ATTENTION_IMPL==0 || TILEMEGA_ATTENTION_IMPL==3,
      "xcore1000 decode attention implements attention_impl=mma16 or simt");
  // mma16 reuses PrefillAttentionMma at Tokens=1: that body already carries
  // the whole decode contract -- RoPE and q/k-norm on the current token, the
  // KV cache write by the owning block, per-block partial and LSE for the
  // merge, and a direct context write when cmax==1 -- so the two
  // implementations differ only in how S and PV are computed. One 16-row MMA
  // tile covers QPerKV <= 8 query rows; the rest of the tile is padding, and
  // the gain is in dropping the per-token shuffle reduction (46-53 ns a
  // shuffle, measured) rather than in arithmetic.
  //
  // It costs shared memory: ServingAttentionSharedBytes(D, KvTile) is
  // 11.5-38.5 KiB against the SIMT body's ~2 KiB, so D=128 with KvTile=64
  // (38.5 KiB) does not leave room for a second resident CTA.
  static constexpr int kKvTile = Q >= 8 ? 64 : 32;
  using Impl=std::conditional_t<TILEMEGA_ATTENTION_IMPL==3,
      mx::DecodeAttentionSimt<D,Q,Norm>,
      mx::PrefillAttentionMma<D,Q,1,16,kKvTile,Norm>>;
  using Element=tilemega::Bf16;
  using SharedStorage=typename Impl::SharedStorage;
  __device__ static void Run(ServingAttentionOperands const& p,SharedStorage& shared,
                            int batch,int group,int query_block,int block) {
    if constexpr (TILEMEGA_ATTENTION_IMPL==3) {
      (void)query_block;
      Impl::Run(p,shared,batch,group,block);
    } else {
      Impl::Run(p,shared,batch,group,query_block,block);
    }
  }
};

}  // namespace tilemega::codegen
