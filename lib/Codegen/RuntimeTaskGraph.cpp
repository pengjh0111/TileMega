// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Codegen/RuntimeWindow.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tilemega::codegen {
RuntimeTaskGraph MaterializeRuntimeTaskGraph(std::vector<int> const& counts,
    std::vector<RuntimeDependencyWindow> const& dependencies, int workers) {
  if (workers<=0) throw std::invalid_argument("runtime task graph needs positive workers");
  RuntimeTaskGraph graph;
  graph.stage_offsets.push_back(0);
  std::vector<int> lengths(workers);
  for (int count:counts) {
    if (count<0 || count>std::numeric_limits<int>::max()-graph.stage_offsets.back())
      throw std::invalid_argument("runtime task count outside representation");
    graph.stage_offsets.push_back(graph.stage_offsets.back()+count);
    for (int t=0;t<count;++t) {
      graph.preferred_worker.push_back(t%workers);
      ++lengths[t%workers];
    }
  }
  graph.successors.resize(graph.stage_offsets.back());
  for (auto const& edge:dependencies) {
    if (edge.producer<0 || edge.consumer<=edge.producer ||
        edge.consumer>=static_cast<int>(counts.size()) || edge.div<=0 || edge.count<0)
      throw std::invalid_argument("invalid projected dependency window");
    for (int c=0;c<counts[edge.consumer];++c) {
      auto bounds=RuntimeDependencyBounds(c,counts[edge.producer],edge.all,
          edge.div,edge.scale,edge.offset,edge.count);
      for (int p=bounds.first;p<bounds.past;++p)
        graph.successors[graph.stage_offsets[edge.producer]+p].push_back(
            graph.stage_offsets[edge.consumer]+c);
    }
  }
  for (auto& edges:graph.successors) {
    std::sort(edges.begin(),edges.end());
    edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
  }
  graph.baseline_max_queue=*std::max_element(lengths.begin(),lengths.end());
  return graph;
}
}  // namespace tilemega::codegen
