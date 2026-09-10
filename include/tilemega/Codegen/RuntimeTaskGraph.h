// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <vector>

namespace tilemega::codegen {
struct RuntimeDependencyWindow {
  int producer, consumer;
  bool all;
  long div, scale, offset, count;
};
struct RuntimeTaskGraph {
  std::vector<int> stage_offsets;
  std::vector<std::vector<int>> successors;
  std::vector<int> preferred_worker;
  int baseline_max_queue = 0;
};
struct RuntimeExactDependencyDesc {
  char const* tasks = nullptr;         ///< [] -> [stage,task]
  char const* dependencies = nullptr;  ///< [consumer stage,task] -> [producer stage,task]
  char const* seq_parameter = nullptr;
  char const* past_parameter = nullptr;
};
// Consumes the already projected/expanded runtime stages, never logical op IDs.
RuntimeTaskGraph MaterializeRuntimeTaskGraph(std::vector<int> const& counts,
    std::vector<RuntimeDependencyWindow> const& dependencies, int workers);
RuntimeTaskGraph MaterializeExactRuntimeTaskGraph(std::vector<int> const& counts,
    RuntimeExactDependencyDesc const& dependencies,int seq,int past,int workers);
}  // namespace tilemega::codegen
