// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/AttentionPageLayout.h>

namespace tilemega::solver {
constexpr bool ServingAttentionPageRowsLegal(int page_bytes,int head_dim) {
  return head_dim>0 && page_bytes>0 && page_bytes%(4*head_dim)==0 &&
      (page_bytes/(4*head_dim))%16==0;
}
constexpr bool ServingAttentionFrontierFits(int page_bytes,int pages,
                                            int head_dim,int kv_block) {
  if(kv_block<=0 || pages<=0 || !ServingAttentionPageRowsLegal(page_bytes,head_dim))return false;
  codegen::AttentionPageLayout layout(page_bytes/(4*head_dim),kv_block,
                                     codegen::AttentionPagePolicy::Packed);
  // Chunk length grows monotonically into this live frontier. Later waves
  // reuse slots and must not be charged as simultaneous page ownership.
  return layout.pages_in_flight<=pages;
}
} // namespace tilemega::solver
