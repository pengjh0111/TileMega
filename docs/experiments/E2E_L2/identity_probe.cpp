// SPDX-License-Identifier: BSD-3-Clause
//
// Part 5.2: with the TaskBody ABI's ownership entry in place, `kIdentity`
// needs one more thing -- an edge whose coupling relation C is contained in
// the identity, so that consumer task `b` depends only on producer task `b`.
// This probe counts how many of the derived edges of the two accepted models
// satisfy that necessary condition, using the isl relations the migrated
// derivation now produces (before the migration there was no containment
// operator to ask with).
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ReferenceModels.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace tilemega::analysis;

namespace {

/// Which TaskBody would execute this operator, and therefore which
/// `TaskOwnershipKind` its CTAs declare (§5.3, TaskBase.h).
///
/// The reference models name their operators after the TaskBody that runs
/// them, so this mirrors `RunStage`'s TaskKind dispatch in ModelHarness.cuh:
/// GEMMs (`w*`) and attention are `kTilePerBlock` (CTA `b` owns task `b`),
/// while RoPE / KVAppend / elementwise walk a flat element range with a
/// grid-stride loop and are `kElementChunk` (CTA `b` owns a slice that
/// depends on `gridDim`, not on the task space). RMSNorm is one token per
/// CTA, i.e. `kTilePerBlock`.
char const* OwnershipKind(std::string const& name) {
  auto starts = [&](char const* prefix) {
    return name.rfind(prefix, 0) == 0 ||
           name.find(std::string(".") + prefix) != std::string::npos;
  };
  if (starts("rope") || starts("kvappend") || starts("silu") ||
      starts("add"))
    return "kElementChunk";
  if (starts("w") || starts("attn") || starts("rmsnorm")) return "kTilePerBlock";
  return "unknown";
}

/// `{ [a,b,...] -> [a,b,...] }` over the same arity as `C`'s two tuples.
CouplingRelation IdentityLike(CouplingRelation const& C) {
  std::vector<std::string> in = C.DomainDimNames();
  std::string tuple;
  for (std::size_t i = 0; i < in.size(); ++i) {
    if (i) tuple += ",";
    tuple += "x" + std::to_string(i);
  }
  return CouplingRelation::FromIslText("{ [" + tuple + "] -> [" + tuple + "] }");
}

void Report(char const* label, OperatorGraph const& graph,
            ParamBinding const& known) {
  std::vector<CouplingEdge> edges = CouplingDerivation().Derive(graph, known);
  int identity = 0, arity_mismatch = 0, same_ownership = 0;
  for (auto const& edge : edges) {
    if (edge.C.empty()) continue;
    if (edge.C.DomainDimNames().size() != edge.C.RangeDimNames().size()) {
      // A consumer task space of a different rank than its producer's cannot
      // have CTA `b` depend on producer CTA `b` at all -- there is no
      // identity to be contained in. Reported so the edge list stays whole.
      ++arity_mismatch;
      std::printf("EDGE model=%s src=%s dst=%s identity=no-arity C=%s\n",
                  label, edge.src.name.c_str(), edge.dst.name.c_str(),
                  edge.C.ToString().c_str());
      continue;
    }
    bool is_identity = edge.C.IsSubset(IdentityLike(edge.C));
    char const* producer_owns = OwnershipKind(edge.src.name);
    char const* consumer_owns = OwnershipKind(edge.dst.name);
    bool same = std::string(producer_owns) == consumer_owns;
    if (is_identity) ++identity;
    if (is_identity && same) ++same_ownership;
    std::printf("EDGE model=%s src=%s dst=%s identity=%s owns=%s->%s "
                "kidentity_admissible=%s C=%s\n", label,
                edge.src.name.c_str(), edge.dst.name.c_str(),
                is_identity ? "yes" : "no", producer_owns, consumer_owns,
                (is_identity && same) ? "yes" : "no",
                edge.C.ToString().c_str());
  }
  std::printf("SUMMARY model=%s edges=%zu identity_candidates=%d "
              "arity_mismatch=%d kidentity_admissible=%d\n", label,
              edges.size(), identity, arity_mismatch, same_ownership);
}

}  // namespace

int main() {
  DecoderShape shape;
  ParamBinding known = DecoderShape::Table27Theta();
  for (auto const& [name, value] : DecoderShape::Table27G().values)
    known.Bind(name, value);
  Report("llama_decoder_layer", LlamaDecoderLayer(shape), known);
  Report("mha4", MhaModel(shape, /*layers=*/4), known);
  return 0;
}
