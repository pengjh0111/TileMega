// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/LlmPlanABI.h>

namespace tilemega::ascend::batched_residual {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 0
__aicore__
#endif
inline constexpr bool Tile(unsigned rows,unsigned columns,unsigned left_stride,
    unsigned right_stride,unsigned output_stride) {
  return rows && rows<=16 && columns && columns<=128 &&
      left_stride>=columns && right_stride>=columns && output_stride>=columns &&
      left_stride<=16384 && right_stride<=16384 && output_stride<=16384 &&
      left_stride%16==0 && right_stride%16==0 && output_stride%16==0;
}
#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 0
__aicore__
#endif
inline constexpr bool Shape(llm::Stage const& s) {
  return s.kind==0 && s.rows==16 && s.columns==1024 && (s.k==2048 || s.k==3072) &&
      s.epilogue==1 && s.extent<=1 && s.tasks==8 && s.workspace!=llm::kAbsent &&
      s.operands[0]!=llm::kAbsent && s.operands[1]!=llm::kAbsent &&
      s.operands[2]!=llm::kAbsent && s.operands[3]!=llm::kAbsent;
}
} // namespace tilemega::ascend::batched_residual
