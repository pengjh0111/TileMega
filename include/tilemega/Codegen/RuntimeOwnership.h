// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

namespace tilemega::codegen {
enum RuntimeOwnershipFlag : std::uint32_t {
  kRoPETileOwnership = 1u << 0,
  kKVTileOwnership = 1u << 1,
  kActivationTileOwnership = 1u << 2,
  kCombinerTileOwnership = 1u << 3,
};
}  // namespace tilemega::codegen
