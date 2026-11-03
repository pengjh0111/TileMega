// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/LlmPlanABI.h>
namespace tilemega::ascend::qk_phase {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 0
__aicore__
#endif
inline constexpr bool Shape(llm::Stage const& s) {
  return s.kind==13 && s.width==128 && s.group==2 && s.extent==8 &&
      s.query_rows==2 && s.tasks==8 && s.kv_block==1088 &&
      s.workspace!=llm::kAbsent && s.operands[10]!=llm::kAbsent &&
      s.operands[11]!=llm::kAbsent && s.operands[10]!=s.operands[11];
}
}
