// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: invariant I2 -- a Relax may replace an edge only if the relaxed
// relation contains the exact one.  This test checks the containment predicate
// itself, in both directions, on relations the derivation actually produced.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ReferenceModels.h>
#include <tilemega/Analysis/TierClassifier.h>

#include <cstdlib>
#include <iostream>

using namespace tilemega::analysis;

namespace {

int failures = 0;

void Require(bool condition, char const* what, int line) {
  if (condition) return;
  std::cerr << "line " << line << ": " << what << '\n';
  ++failures;
}

#define REQUIRE(expression) Require((expression), #expression, __LINE__)

ParamBinding KnownBinding() {
  ParamBinding known = DecoderShape::Table27Theta();
  for (auto const& [name, value] : DecoderShape::Table27G().values)
    known.Bind(name, value);
  return known;
}

CouplingEdge Only(OperatorGraph const& graph, ParamBinding const& known) {
  std::vector<CouplingEdge> edges = CouplingDerivation().Derive(graph, known);
  if (edges.size() != 1) {
    std::cerr << "expected exactly one edge, got " << edges.size() << '\n';
    std::exit(1);
  }
  return edges.front();
}

}  // namespace

int main() {
  DecoderShape shape;
  ParamBinding known = KnownBinding();

  OperatorGraph relaxed_graph = GatherModel(shape, /*data_dependent=*/true);
  OperatorGraph exact_graph = GatherModel(shape, /*data_dependent=*/false);

  CouplingEdge relaxed = Only(relaxed_graph, known);
  CouplingEdge exact = Only(exact_graph, known);

  REQUIRE(!relaxed.exact);
  REQUIRE(exact.exact);
  REQUIRE(relaxed.tier == Tier::kDataDependent);
  REQUIRE(exact.tier == Tier::kAffine);

  // I2 holds in the direction the Relax operation uses ...
  REQUIRE(Contains(relaxed.C, exact.C));
  // ... and the predicate is not vacuous: the exact relation does not contain
  // the relaxed one, so substituting the other way round would be rejected.
  REQUIRE(!Contains(exact.C, relaxed.C));

  // Reflexive on every edge of a real model, and a relaxation of an edge is
  // never claimed to be contained in an unrelated edge's relation.
  OperatorGraph llama = LlamaDecoderLayer(shape);
  std::vector<CouplingEdge> llamaEdges = CouplingDerivation().Derive(llama, known);
  for (auto const& edge : llamaEdges) {
    OperatorNode const* p = llama.Find(edge.src.name);
    REQUIRE(p != nullptr);
    REQUIRE(Contains(edge.C, edge.C));
  }

  // Part 4: what isl can check about a Tier is a claimed relaxation's
  // *consequence*, not the Tier itself (see TierClassifier.h -- Tier is
  // provenance, and classifying by single-valuedness misclassifies §2.7's
  // own Tier 0 rows). A relaxed edge claims to have widened C to the whole
  // producer task space; that claim is machine-checkable, and if it were
  // false the widening would not be I2-safe.
  TierClassifier tiers;
  OperatorNode const& gatherProducer = *relaxed_graph.Find("produce");
  REQUIRE(tiers.RelaxationCoversProducer(relaxed.C, gatherProducer, known));
  // The exact edge does *not* cover the whole producer space -- which is
  // exactly why it is the narrow relation and the relaxed one is the wide
  // one, and confirms the check is not vacuously true.
  REQUIRE(!tiers.RelaxationCoversProducer(exact.C, gatherProducer, known));

  // A relaxed relation over a *different* producer is not accepted.
  CouplingEdge foreign = llamaEdges.front();
  REQUIRE(!Contains(relaxed.C, foreign.C));

  if (failures) {
    std::cerr << failures << " assertion(s) failed\n";
    return 1;
  }
  return 0;
}
