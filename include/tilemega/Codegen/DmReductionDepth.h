// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tilemega::codegen {
// Only MoETopK can be both a reducer and a producer of another reducer.
// Pool and expert-combine are terminal under BuildDmReductionProof's rules.
inline unsigned DmReductionDepthBound(std::vector<TaskKind> const& kinds,
    std::vector<std::pair<unsigned,unsigned>> const& edges, unsigned mask) {
  std::vector<std::vector<unsigned>> incoming(kinds.size());
  for (auto [p,c] : edges) {
    if (p>=kinds.size() || c>=kinds.size() || p>c)
      throw std::invalid_argument("invalid DM reduction depth dependency");
    if (p!=c) incoming[c].push_back(p);
  }
  unsigned bound=1;
  std::vector<unsigned> depth(kinds.size(),1);
  if (mask&2)
    for (unsigned c=0;c<kinds.size();++c) {
      if (kinds[c]!=TaskKind::kMoETopK) continue;
      for (auto p:incoming[c])
        if (kinds[p]==TaskKind::kMoETopK)
          depth[c]=std::max(depth[c],depth[p]+1);
      bound=std::max(bound,depth[c]);
    }
  // Preserve the existing six-reducer limit; the host proof checks the
  // selected chain, rather than rejecting a longer potential dependency path.
  return std::min(bound,6u);
}
} // namespace tilemega::codegen
