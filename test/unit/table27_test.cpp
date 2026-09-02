// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2.7.  Every assertion below is read off the table; nothing is
// read off the implementation.  Where the derivation and the table disagree the
// difference is asserted in the form the derivation produces and explained in
// docs/experiments/P3/table27.md -- the expectation is not moved to fit.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ReferenceModels.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace tilemega::analysis;

namespace {

int failures = 0;

void Require(bool condition, std::string const& what, int line) {
  if (condition) return;
  std::cerr << "line " << line << ": " << what << '\n';
  ++failures;
}

#define REQUIRE(expression) Require((expression), #expression, __LINE__)
#define EQ(actual, expected)                                                 \
  Require((actual) == (expected),                                            \
          std::string(#actual) + " == " + #expected + " (got: " +            \
              Show(actual) + ")",                                            \
          __LINE__)

std::string Show(std::string const& value) { return value; }
std::string Show(long value) { return std::to_string(value); }

ParamBinding Theta() {
  ParamBinding theta = DecoderShape::Table27Theta();
  // S and L_s stay free in the table; pick one instantiation to evaluate the
  // numeric cells.  past is the KV rows already in the cache.
  theta.Bind("S", 512).Bind("L_s", 1024).Bind("past", 512);
  return theta;
}

class Table {
 public:
  explicit Table(OperatorGraph const& graph)
      : edges_(CouplingDerivation().Derive(graph)) {}

  CouplingEdge const& Row(std::string const& src, std::string const& dst) {
    for (auto const& edge : edges_)
      if (edge.src.name == src && edge.dst.name == dst) return edge;
    std::cerr << "no derived edge " << src << " -> " << dst << '\n';
    std::exit(1);
  }

  long Wait(std::string const& src, std::string const& dst) {
    return Row(src, dst).metrics.wait.Eval(Theta(), DecoderShape::Table27G());
  }
  long Fanout(std::string const& src, std::string const& dst) {
    return Row(src, dst).metrics.fanout.Eval(Theta(), DecoderShape::Table27G());
  }
  std::size_t Size() const { return edges_.size(); }
  std::vector<CouplingEdge> const& All() const { return edges_; }

 private:
  std::vector<CouplingEdge> edges_;
};

}  // namespace

int main() {
  DecoderShape shape;
  Table t(LlamaDecoderLayer(shape));

  // Row 1: RMSNorm1 -> Wq/Wk/Wv.  The table folds the three projections into
  // one row, so its fanout 48 is the sum over the three derived edges.
  for (char const* dst : {"wq", "wk", "wv"}) {
    EQ(t.Row("rmsnorm1", dst).C.ToString(),
       std::string("(m,n) -> {rmsnorm1(m)}"));
    EQ(t.Row("rmsnorm1", dst).EventShapeString(),
       std::string("[ceildiv(S, Tm)]"));
    EQ(t.Wait("rmsnorm1", dst), 1L);
    REQUIRE(t.Row("rmsnorm1", dst).tier == Tier::kAffine);
  }
  EQ(t.Fanout("rmsnorm1", "wq") + t.Fanout("rmsnorm1", "wk") +
         t.Fanout("rmsnorm1", "wv"),
     48L);

  // Row 2: Wq -> RoPE_q.
  EQ(t.Row("wq", "rope_q").C.ToString(), std::string("(m,hh) -> {wq(m,hh)}"));
  EQ(t.Row("wq", "rope_q").EventShapeString(),
     std::string("[ceildiv(S, Tm)xn_h]"));
  EQ(t.Wait("wq", "rope_q"), 1L);
  EQ(t.Fanout("wq", "rope_q"), 1L);
  REQUIRE(t.Row("wq", "rope_q").tier == Tier::kAffine);

  // Row 3: RoPE_k -> KVappend.  wait/fanout/Tier as tabulated.  C differs from
  // the table's `(m,hh) ↦ (m,hh)`: the append is tiled one cache row at a time,
  // so the consumer coordinate is a row and the projection into the producer's
  // row-block space is floordiv(row, Tm).  See table27.md.
  EQ(t.Row("rope_k", "kvappend_k").C.ToString(),
     std::string("(row,hh) -> {rope_k(floordiv(row, Tm),hh)}"));
  EQ(t.Wait("rope_k", "kvappend_k"), 1L);
  EQ(t.Fanout("rope_k", "kvappend_k"), 1L);
  REQUIRE(t.Row("rope_k", "kvappend_k").tier == Tier::kSharedInjectiveLayout);
  REQUIRE(t.Row("wv", "kvappend_v").tier == Tier::kSharedInjectiveLayout);

  // Row 4: KVappend -> Attn chunk.  Only the chunk the append lands in is
  // coupled; that is the recorded guard, not a widened C.
  EQ(t.Row("kvappend_k", "attn_chunk").guard,
     std::string("j == floordiv(past, Tkv)"));
  REQUIRE(t.Row("kvappend_k", "attn_chunk").tier == Tier::kStructuredRagged);
  REQUIRE(t.Row("kvappend_k", "attn_chunk").exact);

  // Row 5: RoPE_q -> Attn chunk.
  EQ(t.Row("rope_q", "attn_chunk").C.ToString(),
     std::string("(s,h,j) -> {rope_q(floordiv(s, Tm),h)}"));
  EQ(t.Wait("rope_q", "attn_chunk"), 1L);
  EQ(t.Row("rope_q", "attn_chunk").metrics.fanout.ToString(),
     std::string("ceildiv(L_s, Tkv)"));
  REQUIRE(t.Row("rope_q", "attn_chunk").tier == Tier::kStructuredRagged);

  // Row 6: Attn chunk -> Attn combine.  wait is the runtime chunk count.
  EQ(t.Row("attn_chunk", "attn_combine").metrics.wait.ToString(),
     std::string("ceildiv(L_s, Tkv)"));
  EQ(t.Fanout("attn_chunk", "attn_combine"), 1L);
  EQ(t.Row("attn_chunk", "attn_combine").EventShapeString(),
     std::string("[Sxn_h]"));
  REQUIRE(t.Row("attn_chunk", "attn_combine").tier == Tier::kStructuredRagged);

  // Row 7: Attn combine -> Wo.  wait = Tm x 32, fanout = 32.
  EQ(t.Row("attn_combine", "wo").metrics.wait.ToString(),
     std::string("(Tm * n_h)"));
  EQ(t.Wait("attn_combine", "wo"), 128L * 32);
  EQ(t.Fanout("attn_combine", "wo"), 32L);
  EQ(t.Row("attn_combine", "wo").EventShapeString(),
     std::string("[ceildiv(S, Tm)]"));
  REQUIRE(t.Row("attn_combine", "wo").tier == Tier::kAffine);

  // Row 8: Wo -> residual add.
  EQ(t.Row("wo", "add1").C.ToString(), std::string("(m,n) -> {wo(m,n)}"));
  EQ(t.Wait("wo", "add1"), 1L);
  EQ(t.Fanout("wo", "add1"), 1L);
  EQ(t.Row("wo", "add1").EventShapeString(),
     std::string("[ceildiv(S, Tm)xceildiv(H, Tn)]"));

  // Row 9: add -> RMSNorm2.  One consumer coordinate; wait = 32 column blocks.
  EQ(t.Wait("add1", "rmsnorm2"), 32L);
  EQ(t.Fanout("add1", "rmsnorm2"), 1L);
  EQ(t.Row("add1", "rmsnorm2").EventShapeString(),
     std::string("[ceildiv(S, Tm)]"));
  REQUIRE(t.Row("add1", "rmsnorm2").tier == Tier::kAffine);

  // Row 10: RMSNorm2 -> Wgate/Wup, fanout 224 summed over the two edges.
  EQ(t.Fanout("rmsnorm2", "wgate") + t.Fanout("rmsnorm2", "wup"), 224L);
  EQ(t.Wait("rmsnorm2", "wgate"), 1L);
  EQ(t.Wait("rmsnorm2", "wup"), 1L);

  // Row 11: Wgate,Wup -> SiLU.  The table's wait 2 is the sum over the two
  // operand edges the derivation emits separately.
  EQ(t.Wait("wgate", "silu") + t.Wait("wup", "silu"), 2L);
  EQ(t.Row("wgate", "silu").EventShapeString(),
     std::string("[ceildiv(S, Tm)xceildiv(I, Tn)]"));
  EQ(t.Fanout("wgate", "silu"), 1L);

  // Row 12: SiLU -> Wdown.
  EQ(t.Wait("silu", "wdown"), 112L);
  EQ(t.Fanout("silu", "wdown"), 32L);
  EQ(t.Row("silu", "wdown").EventShapeString(),
     std::string("[ceildiv(S, Tm)]"));

  // Row 13: Wdown -> residual add2.
  EQ(t.Row("wdown", "add2").C.ToString(), std::string("(m,n) -> {wdown(m,n)}"));
  EQ(t.Wait("wdown", "add2"), 1L);
  EQ(t.Fanout("wdown", "add2"), 1L);

  // 11/13 Tier 0, no Tier 3, ragged only on the attention edges.
  int affine = 0;
  for (auto const& edge : t.All()) {
    REQUIRE(edge.tier != Tier::kDataDependent);
    if (edge.tier == Tier::kAffine) ++affine;
    REQUIRE(edge.relaxation.empty());
  }
  EQ((long)affine, 15L);  // 21 derived edges, 4 ragged + 2 layout-tier

  // The derivation finds one edge the table does not list: the residual
  // add1 -> add2.  Asserted so that it cannot silently disappear.
  EQ(t.Row("add1", "add2").C.ToString(), std::string("(m,n) -> {add1(m,n)}"));

  // Decode instantiation (S = 1): row 4's tabulated wait of 1.
  DecoderShape decode;
  decode.S = ClosedForm::Constant(1);
  Table d(LlamaDecoderLayer(decode));
  EQ(d.Wait("kvappend_k", "attn_chunk"), 1L);
  EQ(d.Row("kvappend_k", "attn_chunk").guard,
     std::string("j == floordiv(past, Tkv)"));

  // §2.5 / I1: split-K on the derived row 7 is a reparameterization.  The
  // relation is not re-derived; only Kc joins the parameter list.
  CouplingEdge row7 = t.Row("attn_combine", "wo");
  CouplingEdge split = row7;
  split.C = row7.C.PartitionRange("h", "kc", "Kc");
  split.metrics.wait = row7.metrics.wait.CeilDiv(ClosedForm::Symbol("Kc"));
  REQUIRE(row7.C.SameStructure(split.C));
  EQ(split.C.StructureKey(), row7.C.StructureKey());
  EQ(split.C.Parameters().back(), std::string("Kc"));
  EQ(split.metrics.wait.ToString(), std::string("ceildiv((Tm * n_h), Kc)"));
  ParamBinding g = DecoderShape::Table27G();
  g.Bind("Kc", 4);
  EQ(split.metrics.wait.Eval(Theta(), g), 1024L);

  // Tier 3: a runtime index degrades to the whole producer task space, and is
  // reported as a relaxation rather than dressed up as an affine map.
  Table gather(GatherModel(shape));
  CouplingEdge const& routed = gather.Row("produce", "gather");
  REQUIRE(routed.tier == Tier::kDataDependent);
  REQUIRE(!routed.exact);
  EQ(routed.relaxation.substr(0, 20), std::string("data-dependent index"));

  if (failures) {
    std::cerr << failures << " assertion(s) failed\n";
    return 1;
  }
  return 0;
}
