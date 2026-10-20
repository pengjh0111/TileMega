// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <string_view>
namespace tilemega::arch {
constexpr int ArchIdForTag(std::string_view tag) {
  return tag == "ascend_3510" ? 3510 : 0;
}
} // namespace tilemega::arch
