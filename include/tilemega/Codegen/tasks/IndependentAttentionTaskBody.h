// SPDX-License-Identifier: BSD-3-Clause
#pragma once
// Decode attention TaskBody (one token per request). On xcore1000 the `simt`
// implementation reads the KV cache directly (Codegen/tasks/mx/
// DecodeAttentionSimt.h); `mma16` is not implemented yet and is rejected.
#include <tilemega/Backend/ServingAttentionImpl.h>
#include <tilemega/Codegen/tasks/AttentionPageLayout.h>
#include <tilemega/Codegen/tasks/mx/DecodeAttentionSimt.h>
#include <tilemega/Target/ArchDispatch.h>

#include <type_traits>

namespace tilemega::codegen {

template<class Arch,int D,int Q,bool Norm,int PageBytes=8192,
    AttentionPagePolicy Policy=AttentionPagePolicy::WarpPrivate,int Buffers=2>
struct IndependentAttentionTaskBody {
  static_assert(std::is_same_v<Arch,arch::Xcore1000>);
  static_assert(TILEMEGA_ATTENTION_IMPL==3,
      "xcore1000 decode attention implements attention_impl=simt only");
  using Impl=mx::DecodeAttentionSimt<D,Q,Norm>;
  using Element=tilemega::Bf16;
  using SharedStorage=typename Impl::SharedStorage;
  __device__ static void Run(ServingAttentionOperands const& p,SharedStorage& shared,
                            int batch,int group,int query_block,int block) {
    (void)query_block;
    Impl::Run(p,shared,batch,group,block);
  }
};

}  // namespace tilemega::codegen
