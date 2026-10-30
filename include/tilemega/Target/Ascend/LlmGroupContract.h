// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#ifndef TILEMEGA_LLM_G14
#define TILEMEGA_LLM_G14 0
#endif
static_assert(TILEMEGA_LLM_G14==0 || TILEMEGA_LLM_G14==1);

namespace tilemega::ascend::llm {
// G14 requires an explicitly emitted implementation catalogue. Existing G8
// images and the default target do not acquire a larger resource domain.
inline constexpr bool SupportsGroups(unsigned groups) {
  return groups==2 || groups==4 || groups==8 || (TILEMEGA_LLM_G14 && groups==14);
}
}
