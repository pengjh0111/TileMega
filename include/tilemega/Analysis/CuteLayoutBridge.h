// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.2 CuTe layout projection (Phase 3 stub).
#pragma once
#include <string>
namespace tilemega::analysis {
struct LayoutProjection { std::string expression; bool static_shape = false; };
class CuteLayoutBridge {
 public:
  LayoutProjection Project(std::string const& access_relation) const;
};
}  // namespace tilemega::analysis
