// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstddef>

namespace tilemega::codegen::executor {

// CUDA's default per-CTA allowance includes both static and dynamic storage.
// Subtraction also keeps the comparison valid for oversized caller requests.
constexpr bool SharedMemoryRequiresOptIn(std::size_t dynamic_bytes,
                                         std::size_t static_bytes,
                                         std::size_t default_limit_bytes) {
  return static_bytes > default_limit_bytes ||
         dynamic_bytes > default_limit_bytes - static_bytes;
}

}  // namespace tilemega::codegen::executor
