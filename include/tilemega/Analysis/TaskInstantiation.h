// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 (three-layer IR), §2.3 (Split), §2.4 (split-K semantics).
//
// L-task: L-sem instantiated at one granularity g. The result is the
// OperatorGraph the coupling derivation already consumes, so the derivation is
// unchanged; what changes is where the graph comes from. Every tile size
// enters here and nowhere earlier, which is what makes L-sem rebuildable and
// I1 checkable.
#pragma once

#include <map>
#include <string>
#include <vector>

#include <tilemega/Analysis/Semantics.h>
#include <tilemega/Analysis/TensorSpace.h>

namespace tilemega::analysis {

/// g: a tile per (operator, iteration dim), plus the reduction chunk sizes that
/// turn a declared splittable reduction into a partial/combine pair. A dim
/// with no entry keeps its whole extent, i.e. contributes no task coordinate.
struct Granularity {
  std::map<std::string, std::map<std::string, ClosedForm>> tiles;
  std::map<std::string, ClosedForm> reduction_chunk;

  Granularity& Tile(std::string op, std::string dim, ClosedForm value);
  Granularity& Split(std::string op, ClosedForm chunk);
  bool TileOf(std::string const& op, std::string const& dim,
              ClosedForm* value) const;
  bool ChunkOf(std::string const& op, ClosedForm* value) const;
  std::string Serialize() const;
};

/// Instantiate every op. A splittable reduction with a chunk in `g` becomes two
/// task spaces (partial contributions and an explicit combiner); anything else
/// becomes one.
OperatorGraph Instantiate(SemanticGraph const& graph, Granularity const& g);

}  // namespace tilemega::analysis
