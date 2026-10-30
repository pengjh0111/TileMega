// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

namespace tilemega::ascend::argmax {
inline constexpr uint32_t kFloatIndexSentinel=1u<<24;
#if defined(__CCE_AICORE__)
#define TILEMEGA_ARGMAX_INLINE __aicore__ inline
#else
#define TILEMEGA_ARGMAX_INLINE inline
#endif
TILEMEGA_ARGMAX_INLINE constexpr bool Tile(unsigned rows,unsigned columns,unsigned stride,unsigned output_stride,
                    unsigned first,unsigned vocabulary) {
  return rows && rows<=16 && columns && columns<=256 && stride>=columns && stride<=UINT32_MAX/2 && stride%16==0 &&
      output_stride && vocabulary && vocabulary<kFloatIndexSentinel &&
      uint64_t(first)+columns<=vocabulary && uint64_t(rows)*((columns+15)/16*16)<=4096;
}
TILEMEGA_ARGMAX_INLINE constexpr bool Merge(unsigned partials,unsigned vocabulary) {
  return partials && partials<=2048 && vocabulary && vocabulary<kFloatIndexSentinel;
}
#undef TILEMEGA_ARGMAX_INLINE
} // namespace tilemega::ascend::argmax
