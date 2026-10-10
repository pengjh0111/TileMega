// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

namespace tilemega::codegen {
struct DmReductionArrival {
  std::uint32_t task=0,expected=0;
};
struct DmReductionStage {
  std::uint32_t target=~std::uint32_t(0),task_offset=0,ticket_offset=0;
  std::uint32_t counted_edge=~std::uint32_t(0);
};
struct DmReductionView {
  DmReductionStage const* stages=nullptr;
  std::uint32_t const* offsets=nullptr;
  DmReductionArrival const* arrivals=nullptr;
  unsigned long long* tickets=nullptr;
  std::uint32_t ticket_count=0;
};
} // namespace tilemega::codegen
