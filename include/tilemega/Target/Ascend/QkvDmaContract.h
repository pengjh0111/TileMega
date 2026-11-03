// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/LlmPlanABI.h>

namespace tilemega::ascend::qkv_dma {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 0
__aicore__
#endif
inline constexpr bool Shape(llm::Stage const& s) {
  return s.kind==0 && s.rows==1 && s.columns==4096 && s.k==1024 &&
      s.epilogue==0 && s.extent<=1 && s.tasks==32 &&
      s.operands[0]!=llm::kAbsent && s.operands[1]!=llm::kAbsent &&
      s.operands[3]!=llm::kAbsent;
}
}
