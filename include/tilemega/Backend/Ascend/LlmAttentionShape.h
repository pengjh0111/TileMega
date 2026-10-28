// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/LlmPlanABI.h>

namespace tilemega::ascend {
struct LlmAttentionShape {
  unsigned batch,kv_group,first_position,positions,rows,keys,valid_keys,width;
};
__aicore__ inline bool DescribeLlmAttention(llm::Stage const& s,unsigned tile,
    llm::Metadata const& m,unsigned past,LlmAttentionShape& out) {
  if(s.kind!=13 || (s.variant!=4 && s.variant!=5 && s.variant!=6 && s.variant!=7 && s.variant!=8 && s.variant!=9) || s.workspace==llm::kAbsent ||
      m.capacity!=llm::attention::kCapacity || s.kv_block!=m.capacity ||
      (s.width!=64 && s.width!=128) || !s.extent || !s.group || s.group>4 ||
      (m.seq!=1 && m.seq!=64) || uint64_t(past)+m.seq>=m.capacity || (m.seq==64 && past))return false;
  if(s.variant==9) {
#if !defined(TILEMEGA_LLM_FP32_KV_CACHE) || !TILEMEGA_LLM_FP32_KV_CACHE
    return false;
#else
    if(s.operands[10]==llm::kAbsent || s.operands[11]==llm::kAbsent ||
        s.operands[10]==s.operands[11])return false;
#endif
  }
  unsigned blocks=m.seq==64?4:1,positions=m.seq==64?16:1;
  if(s.query_rows!=positions*s.group || tile>=s.tasks ||
      uint64_t(m.batch)*s.extent*blocks!=s.tasks)return false;
  unsigned qb=(tile/s.extent)%blocks,b=tile/(s.extent*blocks);
  out={b,tile%s.extent,m.seq==64?qb*16:past,positions,positions*s.group,
    m.seq==64?64:m.capacity,m.seq==64?64:past+1,s.width};
  return out.rows<=llm::attention::kMaxRows;
}
__aicore__ inline GM_ADDR LlmAttentionWork(GM_ADDR arena,llm::Stage const& s,unsigned owner_group) {
  return arena+s.workspace+uint64_t(owner_group)*llm::attention::kGroupBytes;
}
__aicore__ inline GM_ADDR LlmAttentionCachedKv(GM_ADDR arena,llm::Stage const& s,
    llm::Metadata const& m,LlmAttentionShape const& h,bool value) {
  return arena+s.operands[value?11:10]+(uint64_t(h.batch)*s.extent+h.kv_group)*m.capacity*h.width*4;
}
} // namespace tilemega::ascend
