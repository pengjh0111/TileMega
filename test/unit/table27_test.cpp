// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2.7.  Every assertion below is read off the table; nothing is
// read off the implementation.  Where the derivation and the table disagree the
// difference is asserted in the form the derivation produces and explained in
// docs/experiments/P3/table27.md -- the expectation is not moved to fit.
//
// This is the isl-backed re-derivation (Part 3 migration). Re-running the
// same cross-validation this test always did surfaced one genuine new
// finding, not merely a representation change: row 3's tabulated fanout (1)
// is wrong at the granularity kvappend_k/kvappend_v actually use (one task
// per cache row, tile = 1) -- the true |C^-1(p0)| is Tm = 128, since each
// 128-row rope_k block feeds 128 individual row-tasks. The pre-migration
// ClosedForm heuristic ("a coordinate that occurs in C is pinned, contributes
// factor 1") silently assumed every occurring coordinate is recovered
// bijectively from the producer coordinate; that is false for the floordiv
// occurrence this edge has (many rows per block), and isl's genuine
// |C^-1(y)| count (barvinok, not a heuristic) catches it. See
// docs/experiments/P3/table27.md and TileMega_skeleton.md §2.7 for the
// corrected table entry. wait is unaffected (still 1: one row still needs
// exactly one producer block).
//
// Rows 4 and 5 carry the same defect and were found later, when the frontend
// wiring re-derived §2.7 from the exported model: their fanout cells had
// never been asserted here, so nothing caught them. Both are now pinned. All
// remaining rows' wait/fanout match the table exactly; the three corrections
// share one cause (a producer coarser than its consumer) and are not a
// wholesale re-derivation disagreement.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ReferenceModels.h>

#include <cstdlib>
#include <iostream>
#include <set>
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

// isl can only take a literal floor/ceildiv divisor (docs/experiments/
// P3_ISL/result.md), so every symbol a derivation might use as a tile size
// or GQA group divisor -- everything Table27Theta/Table27G fix -- must be
// bound before DeriveCoupling builds an isl_map. S/L_s/past stay free in
// the *derivation* (matching §2.7's own "S and L_s stay free"), but a fully
// numeric instantiation is still needed to Eval() a specific cell, exactly
// as the pre-migration test did with theta.Bind("S",...).
ParamBinding KnownBinding() {
  ParamBinding known = DecoderShape::Table27Theta();
  for (auto const& [name, value] : DecoderShape::Table27G().values)
    known.Bind(name, value);
  return known;
}

ParamBinding Theta() {
  ParamBinding known = KnownBinding();
  known.Bind("S", 512).Bind("L_s", 1024).Bind("past", 512);
  return known;
}

class Table {
 public:
  // `deriveKnown` binds only theta/g (S/L_s/past stay free, matching §2.7's
  // own "S and L_s stay free" and keeping C's printed form comparable to the
  // table's notation); `evalKnown` additionally fixes S/L_s/past, needed
  // only to Eval() a specific numeric cell.
  Table(OperatorGraph const& graph, ParamBinding deriveKnown, ParamBinding evalKnown)
      : evalKnown_(std::move(evalKnown)),
        edges_(CouplingDerivation().Derive(graph, deriveKnown)) {}

  CouplingEdge const& Row(std::string const& src, std::string const& dst) {
    for (auto const& edge : edges_)
      if (edge.src.name == src && edge.dst.name == dst) return edge;
    std::cerr << "no derived edge " << src << " -> " << dst << '\n';
    std::exit(1);
  }

  long Wait(std::string const& src, std::string const& dst) {
    return Row(src, dst).metrics.wait.Eval(evalKnown_);
  }
  long Fanout(std::string const& src, std::string const& dst) {
    return Row(src, dst).metrics.fanout.Eval(evalKnown_);
  }
  std::size_t Size() const { return edges_.size(); }
  std::vector<CouplingEdge> const& All() const { return edges_; }

 private:
  ParamBinding evalKnown_;
  std::vector<CouplingEdge> edges_;
};

}  // namespace

int main() {
  DecoderShape shape;
  Table t(LlamaDecoderLayer(shape), KnownBinding(), Theta());

  // Row 1: RMSNorm1 -> Wq/Wk/Wv.  The table folds the three projections into
  // one row, so its fanout 48 is the sum over the three derived edges.
  for (char const* dst : {"wq", "wk", "wv"}) {
    EQ(t.Row("rmsnorm1", dst).EventShapeString(),
       std::string("[ceildiv(S, Tm)]"));
    EQ(t.Wait("rmsnorm1", dst), 1L);
    REQUIRE(t.Row("rmsnorm1", dst).tier == Tier::kAffine);
  }
  EQ(t.Row("rmsnorm1", "wq").C.ToString(),
     std::string("[S] -> { [m, n] -> [p0 = m] : m >= 0 and 128m < S and 0 <= n <= 31 }"));
  EQ(t.Fanout("rmsnorm1", "wq") + t.Fanout("rmsnorm1", "wk") +
         t.Fanout("rmsnorm1", "wv"),
     48L);

  // Row 2: Wq -> RoPE_q.
  EQ(t.Row("wq", "rope_q").C.ToString(),
     std::string("[S] -> { [m, hh] -> [p0 = m, p1 = hh] : m >= 0 and 128m < S and "
                 "0 <= hh <= 31 }"));
  EQ(t.Row("wq", "rope_q").EventShapeString(),
     std::string("[ceildiv(S, Tm)xn_h]"));
  EQ(t.Wait("wq", "rope_q"), 1L);
  EQ(t.Fanout("wq", "rope_q"), 1L);
  REQUIRE(t.Row("wq", "rope_q").tier == Tier::kAffine);

  // Row 3: RoPE_k -> KVappend.  wait as tabulated.  C and fanout both differ
  // from the table, for the same underlying reason: the append is tiled one
  // cache row at a time (tile = 1), so the consumer coordinate is a row, not
  // a Tm-block, and the projection into the producer's row-block space is
  // floordiv(row, Tm) -- a many-to-one map. wait(row) = 1 still (one row
  // needs exactly one producer block); fanout(p0) = |{row mapping to p0}| is
  // genuinely Tm = 128 (each 128-row producer block feeds 128 row-tasks),
  // not the table's 1 -- see the file header and table27.md.
  EQ(t.Row("rope_k", "kvappend_k").C.ToString(),
     std::string("[S] -> { [row, hh] -> [p0, p1 = hh] : 0 <= row < S and 0 <= hh <= 7 "
                 "and -127 + row <= 128p0 <= row }"));
  EQ(t.Wait("rope_k", "kvappend_k"), 1L);
  EQ(t.Fanout("rope_k", "kvappend_k"), 128L);
  EQ(t.Fanout("wv", "kvappend_v"), 128L);
  REQUIRE(t.Row("rope_k", "kvappend_k").tier == Tier::kSharedInjectiveLayout);
  REQUIRE(t.Row("wv", "kvappend_v").tier == Tier::kSharedInjectiveLayout);

  // Row 4: KVappend -> Attn chunk.  Only the chunk the append lands in is
  // coupled; that is the recorded guard, not a widened C. The table's wait=1
  // is the decode (S=1) instantiation; table27.md documents the general
  // (prefill) wait as Tkv -- not re-asserted here as a specific number.
  // fanout was left unchecked until the frontend wiring re-derived it, which
  // is why the table's `S` survived: the consumer ranges over the G query
  // heads sharing one kv head, so it is G*S = 2048, right only for MHA.
  // table27.md note (g) counts both cells by hand from the table's own C.
  EQ(t.Row("kvappend_k", "attn_chunk").guard,
     std::string("j == floordiv(past, Tkv)"));
  EQ(t.Fanout("kvappend_k", "attn_chunk"), 2048L);
  REQUIRE(t.Row("kvappend_k", "attn_chunk").tier == Tier::kStructuredRagged);
  REQUIRE(t.Row("kvappend_k", "attn_chunk").exact);

  // Row 5: RoPE_q -> Attn chunk.  Same defect as row 3: rope_q emits Tm-row
  // blocks and attn_chunk consumes single tokens, so one block feeds
  // Tm * ceildiv(L_s,Tkv) = 128 * 8 chunks, not the table's 8.
  EQ(t.Wait("rope_q", "attn_chunk"), 1L);
  EQ(t.Fanout("rope_q", "attn_chunk"), 1024L);
  REQUIRE(t.Row("rope_q", "attn_chunk").tier == Tier::kStructuredRagged);

  // Row 6: Attn chunk -> Attn combine.  wait is the runtime chunk count,
  // ceildiv(L_s, Tkv) = ceildiv(1024, 128) = 8 at this instantiation.
  EQ(t.Wait("attn_chunk", "attn_combine"), 8L);
  EQ(t.Fanout("attn_chunk", "attn_combine"), 1L);
  EQ(t.Row("attn_chunk", "attn_combine").EventShapeString(),
     std::string("[Sxn_h]"));
  REQUIRE(t.Row("attn_chunk", "attn_combine").tier == Tier::kStructuredRagged);

  // Row 7: Attn combine -> Wo.  wait = Tm x 32, fanout = 32.
  EQ(t.Wait("attn_combine", "wo"), 128L * 32);
  EQ(t.Fanout("attn_combine", "wo"), 32L);
  EQ(t.Row("attn_combine", "wo").EventShapeString(),
     std::string("[ceildiv(S, Tm)]"));
  REQUIRE(t.Row("attn_combine", "wo").tier == Tier::kAffine);

  // Row 8: Wo -> residual add.
  EQ(t.Row("wo", "add1").C.ToString(),
     std::string("[S] -> { [m, n] -> [p0 = m, p1 = n] : m >= 0 and 128m < S and "
                 "0 <= n <= 31 }"));
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
  EQ(t.Row("wdown", "add2").C.ToString(),
     std::string("[S] -> { [m, n] -> [p0 = m, p1 = n] : m >= 0 and 128m < S and "
                 "0 <= n <= 31 }"));
  EQ(t.Wait("wdown", "add2"), 1L);
  EQ(t.Fanout("wdown", "add2"), 1L);

  // 15/21 Tier 0 (matching 11/13 table rows once split by operand, per
  // difference (a)), no Tier 3, ragged only on the attention edges.
  int affine = 0;
  for (auto const& edge : t.All()) {
    REQUIRE(edge.tier != Tier::kDataDependent);
    if (edge.tier == Tier::kAffine) ++affine;
    REQUIRE(edge.relaxation.empty());
  }
  EQ((long)affine, 15L);

  // Part 3: the tier is a derived summary, and it is lossy in exactly the
  // place §2.4 conflates two unrelated obligations. All four Tier-2 edges
  // here are ragged, but they split three ways once the attributes are kept
  // apart, and every one of them is *exactly* derived -- the tier alone
  // cannot say that.
  std::set<std::string> tier2;
  for (auto const& edge : t.All()) {
    REQUIRE(DeriveTier(edge.attributes) == edge.tier);
    if (edge.tier == Tier::kStructuredRagged)
      tier2.insert(edge.attributes.ToString());
  }
  EQ((long)tier2.size(), 3L);
  for (auto const& attributes : tier2)
    REQUIRE(attributes.find("+ exact +") != std::string::npos);
  REQUIRE(t.Row("rmsnorm1", "wq").attributes.ToString() ==
          "affine + symbolic_static + exact + none + constant");
  REQUIRE(t.Row("rope_k", "kvappend_k").attributes.ToString() ==
          "layout_mediated + symbolic_static + exact + none + constant");
  REQUIRE(t.Row("kvappend_k", "attn_chunk").attributes.ToString() ==
          "layout_mediated + runtime_dynamic + exact + prefix_sum + constant");

  // The derivation finds one edge the table does not list: the residual
  // add1 -> add2.  Asserted so that it cannot silently disappear.
  EQ(t.Row("add1", "add2").C.ToString(),
     std::string("[S] -> { [m, n] -> [p0 = m, p1 = n] : m >= 0 and 128m < S and "
                 "0 <= n <= 31 }"));

  // Decode instantiation (S = 1): row 4's tabulated wait of 1.
  DecoderShape decode;
  decode.S = ClosedForm::Constant(1);
  ParamBinding decodeKnown = KnownBinding();
  decodeKnown.Bind("L_s", 1024).Bind("past", 512);
  Table d(LlamaDecoderLayer(decode), KnownBinding(), decodeKnown);
  EQ(d.Wait("kvappend_k", "attn_chunk"), 1L);
  EQ(d.Row("kvappend_k", "attn_chunk").guard,
     std::string("j == floordiv(past, Tkv)"));

  // §2.3 Coarsen: C_kappa = floor(./kappa) o C on the producer (range) side.
  // Replaces the pre-migration split-K reparameterization test
  // (AffineRelation::PartitionRange/SameStructure/StructureKey/Parameters,
  // all deleted with AffineRelation -- there is no clean isl analogue of
  // "same structure, one more parameter" for an arbitrary quantified-range
  // split). Coarsen is the operation this migration actually delivers for
  // §2.3, and is what a real split-K reparameterization's wait/kappa
  // relationship reduces to: coarsening row 7's producer by kappa=4 on the
  // n_h-sized axis divides wait by 4, exactly like Kc=4 split-K would.
  CouplingEdge row7 = t.Row("attn_combine", "wo");
  QuasiPolynomial coarseWait =
      row7.C.Coarsen({1, 4}).Card();
  EQ(coarseWait.Eval(Theta()), 1024L);

  // Tier 3: a runtime index degrades to the whole producer task space, and is
  // reported as a relaxation rather than dressed up as an affine map.
  Table gather(GatherModel(shape), KnownBinding(), KnownBinding());
  CouplingEdge const& routed = gather.Row("produce", "gather");
  REQUIRE(routed.tier == Tier::kDataDependent);
  REQUIRE(routed.attributes.ToString() ==
          "data_dependent + symbolic_static + relaxed + tensor_values + uncountable");
  REQUIRE(!routed.exact);
  EQ(routed.relaxation.substr(0, 20), std::string("data-dependent index"));

  if (failures) {
    std::cerr << failures << " assertion(s) failed\n";
    return 1;
  }
  return 0;
}
