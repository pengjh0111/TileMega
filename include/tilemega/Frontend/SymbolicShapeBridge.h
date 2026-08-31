// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definition 1 and §5.1 symbolic guards (Phase 1 stub).
#pragma once
#include <string>
#include <vector>
namespace tilemega::frontend {
struct SymbolicShape { std::vector<std::string> dimensions; std::vector<std::string> guards; };
class SymbolicShapeBridge {
 public:
  SymbolicShape Parse(std::string const& shape_environment) const;
};
}  // namespace tilemega::frontend
