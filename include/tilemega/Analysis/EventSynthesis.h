// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 invariant I2 and §5.3 event synthesis (Phase 3 stub).
#pragma once
#include <string>
#include <vector>
namespace tilemega::analysis {
struct EventRequirement { std::string producer; std::string consumer; std::string count; };
class EventSynthesis {
 public:
  std::vector<EventRequirement> Synthesize(std::vector<std::string> const& edges) const;
};
}  // namespace tilemega::analysis
