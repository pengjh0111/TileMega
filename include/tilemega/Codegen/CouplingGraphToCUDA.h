// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 Coupling Graph lowering (Phase 2 stub).
#pragma once
#include <string>
namespace tilemega::codegen {
class CouplingGraphToCUDA { public: std::string Lower(std::string const& coupling_graph) const; };
}  // namespace tilemega::codegen
