// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 three-layer IR, invariant I1, §2.3 Split, §2.4 split-K.
//
// The machine-executable form of I1 lives here: L-sem's serialization must be
// byte-identical under two different granularities. That replaces the old
// StructureKey() proxy, which compared a derived relation's shape rather than
// the semantics itself.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/TierClassifier.h>
#include <tilemega/Analysis/ReferenceModels.h>

#include <cstdlib>
#include <iostream>
#include <string>

using namespace tilemega::analysis;

namespace {

int failures = 0;

void Require(bool condition, std::string const& what, int line) {
  if (condition) return;
  std::cerr << "line " << line << ": " << what << '\n';
  ++failures;
}

#define REQUIRE(expression) Require((expression), #expression, __LINE__)
#define EQ(actual, expected)                                            \
  Require((actual) == (expected),                                       \
          std::string(#actual) + " == " + #expected + " (got: " +       \
              Show(actual) + ")",                                       \
          __LINE__)

std::string Show(std::string const& value) { return value; }
std::string Show(long value) { return std::to_string(value); }
std::string Show(std::size_t value) { return std::to_string(value); }

ParamBinding KnownBinding() {
  ParamBinding known = DecoderShape::Table27Theta();
  for (auto const& [name, value] : DecoderShape::Table27G().values)
    known.Bind(name, value);
  return known;
}

OperatorNode const* Node(OperatorGraph const& graph, std::string const& name) {
  return graph.Find(name);
}

}  // namespace

int main() {
  // --- I1: L-sem is byte-identical under two different g -------------------
  DecoderShape symbolic;
  DecoderShape concrete;
  concrete.Tm = ClosedForm::Constant(64);
  concrete.Tn = ClosedForm::Constant(32);
  concrete.Tkv = ClosedForm::Constant(256);

  ReferenceModel a = LlamaDecoderLayerSem(symbolic);
  ReferenceModel b = LlamaDecoderLayerSem(concrete);
  EQ(a.sem.Serialize(), b.sem.Serialize());
  REQUIRE(a.g.Serialize() != b.g.Serialize());
  REQUIRE(!a.sem.Serialize().empty());
  // The instantiations must differ, or the identity above would be vacuous.
  REQUIRE(Node(a.Task(), "add1")->tile[0].ToString() !=
          Node(b.Task(), "add1")->tile[0].ToString());
  // Nothing in L-sem names a tile size.
  for (char const* tile : {"Tm", "Tn", "Tkv"})
    REQUIRE(a.sem.Serialize().find(tile) == std::string::npos);

  // --- §2.3 memory effects ------------------------------------------------
  SemanticOp const* append = a.sem.Find("kvappend_k");
  REQUIRE(append != nullptr);
  REQUIRE(append->result_effect.kind == EffectKind::kReadWrite);
  EQ(append->result_effect.state_object, std::string("kv_cache"));
  EQ(append->result_effect.alias_set, std::string("kv_cache"));
  SemanticOp const* attn = a.sem.Find("attn_chunk");
  REQUIRE(attn != nullptr);
  EQ(attn->operands[1].effect.state_object, std::string("kv_cache"));
  REQUIRE(attn->operands[1].effect.kind == EffectKind::kRead);
  // A scratch result carries no state object.
  EQ(a.sem.Find("silu")->result_effect.state_object, std::string());

  // --- §2.4 split-K is a granularity choice, not a TaskBody feature -------
  // Attention declares its KV reduction splittable; §2.7's chunk/combine pair
  // is what Split produces, not two hand-written operators.
  REQUIRE(attn->reduction.splittable);
  EQ(attn->reduction.dim, std::string("kv"));
  EQ(attn->reduction.combiner, std::string("attn_combine"));
  OperatorGraph instantiated = a.Task();
  REQUIRE(Node(instantiated, "attn_chunk") != nullptr);
  REQUIRE(Node(instantiated, "attn_combine") != nullptr);
  EQ(Node(instantiated, "attn_chunk")->output.axes.size(), std::size_t(3));

  // The same operation applied to a GEMM. wo reduces over n_h*d = 4096
  // elements; splitting it into Kc = 4 chunks of 1024 gives each partial task
  // Tm x (1024/d) producer tasks, i.e. wait = Tm*n_h/Kc.
  ReferenceModel split = LlamaDecoderLayerSem(symbolic);
  split.g.Split("wo", ClosedForm::Constant(1024));
  OperatorGraph splitGraph = split.Task();
  REQUIRE(Node(splitGraph, "wo.combine") != nullptr);
  EQ(Node(splitGraph, "wo")->output.name, std::string("wo.partial"));
  EQ(Node(splitGraph, "wo")->output.axes.size(), std::size_t(3));

  ParamBinding eval = KnownBinding();
  eval.Bind("S", 512).Bind("L_s", 1024).Bind("past", 512);
  auto edges = CouplingDerivation().Derive(splitGraph, KnownBinding());
  CouplingEdge const* row7 = nullptr;
  for (auto const& edge : edges)
    if (edge.src.name == "attn_combine" && edge.dst.name == "wo") row7 = &edge;
  REQUIRE(row7 != nullptr);
  if (row7) {
    EQ(row7->metrics.wait.Eval(eval), 1024L);
    // §2.3 Coarsen by kappa = 4 on the same axis reaches the same wait: the
    // two operations agree where they overlap, which is the property that
    // makes split-K expressible as a CG transform rather than a new mechanism.
    auto unsplit = CouplingDerivation().Derive(a.Task(), KnownBinding());
    for (auto const& edge : unsplit)
      if (edge.src.name == "attn_combine" && edge.dst.name == "wo")
        EQ(edge.C.Coarsen({1, 4}).Card().Eval(eval), 1024L);
  }

  // --- degradation: an operator no pattern recognizes ---------------------
  // "One operator = one task space": the generic semantics tiles the result at
  // its full extent and reads every operand in full, so the consumer is a
  // single task waiting on all of the producer's. It is a coarsening, not an
  // error, and not a fabricated affine relation.
  OperatorGraph unknown = UnknownOperatorModel(symbolic);
  EQ(unknown.nodes.size(), std::size_t(2));
  REQUIRE(UnknownOperatorModelSem(symbolic).sem.Find("mystery")->generic);
  OperatorNode const* mystery = Node(unknown, "mystery");
  REQUIRE(mystery != nullptr);
  EQ(mystery->tile[0].ToString(), mystery->output.axes[0].extent.ToString());
  EQ(mystery->tile[1].ToString(), mystery->output.axes[1].extent.ToString());
  auto degraded = CouplingDerivation().Derive(unknown, KnownBinding());
  EQ(degraded.size(), std::size_t(1));
  CouplingEdge const& edge = degraded.front();
  // The relaxed relation is itself exactly countable -- what is conservative
  // is the declared access, not the counting. Part 3 splits those two apart.
  EQ(edge.metrics.wait.Eval(eval), 128L);
  EQ(edge.metrics.fanout.Eval(eval), 1L);
  REQUIRE(TierClassifier().RelaxationCoversProducer(edge.C, *Node(unknown, "produce"),
                                                    KnownBinding()));

  if (failures) {
    std::cerr << failures << " assertion(s) failed\n";
    return 1;
  }
  return 0;
}
