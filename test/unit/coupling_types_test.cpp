// SPDX-License-Identifier: BSD-3-Clause
// Exercises the isl-backed CouplingEdge/DerivedMetrics types produced by a
// real derivation (not hand-built ProducerMap trees -- that AffineRelation
// machinery is gone; CouplingRelation is a genuine isl_map, see
// CouplingRelation.h), and Coarsen, the operation the migration exists to
// make possible (§2.3's C_kappa; AffineRelation had no image/preimage/
// composition operator at all, so Coarsen was unimplementable before this).
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ReferenceModels.h>

#include <cstdlib>
#include <iostream>
#include <string>

using namespace tilemega::analysis;

namespace {

void Require(bool condition, char const* expression, int line) {
  if (condition) return;
  std::cerr << "requirement failed at line " << line << ": " << expression
            << '\n';
  std::exit(1);
}

#define REQUIRE(expression) Require((expression), #expression, __LINE__)

ParamBinding KnownBinding() {
  ParamBinding known = DecoderShape::Table27Theta();
  for (auto const& [name, value] : DecoderShape::Table27G().values)
    known.Bind(name, value);
  return known;
}

CouplingEdge Find(std::vector<CouplingEdge> const& edges, std::string const& src,
                  std::string const& dst) {
  for (auto const& edge : edges)
    if (edge.src.name == src && edge.dst.name == dst) return edge;
  std::cerr << "no edge " << src << " -> " << dst << '\n';
  std::exit(1);
}

}  // namespace

int main() {
  DecoderShape shape;
  ParamBinding known = KnownBinding();
  auto edges = CouplingDerivation().Derive(LlamaDecoderLayer(shape), known);

  // §2.7 row 1: rmsnorm1 -> wq, wait = 1 (one producer tile per consumer tile).
  auto rmsnormToWq = Find(edges, "rmsnorm1", "wq");
  REQUIRE(rmsnormToWq.metrics.wait.Eval(known) == 1);
  REQUIRE(rmsnormToWq.C.ToString().find("p0") != std::string::npos);

  // §2.7 row 7: attn_combine -> wo, wait = Tm * n_h = 4096.
  auto combineToWo = Find(edges, "attn_combine", "wo");
  REQUIRE(combineToWo.metrics.wait.Eval(known) == 4096);

  // §2.7 row 11: wgate -> silu and wup -> silu, wait = 1 each (the table's
  // combined "1 + 1 = 2" is the per-operand split, §2.7's difference (a)).
  auto gateToSilu = Find(edges, "wgate", "silu");
  auto upToSilu = Find(edges, "wup", "silu");
  REQUIRE(gateToSilu.metrics.wait.Eval(known) == 1);
  REQUIRE(upToSilu.metrics.wait.Eval(known) == 1);

  // §2.3 Coarsen: C_kappa = floor(./kappa) o C on the producer (range) side.
  // combine->wo's producer coordinate has two dims (see the derived C's
  // "[p0, p1]" range); coarsening only p1 (the n_h-sized head axis) by 4
  // groups 4 heads into one coarse event coordinate, so wait divides by 4.
  // Coarsen is a *reindexing* of the range (floor(producer/kappa)), not an
  // I2 relaxation in the original coordinate system, so it is not compared
  // to combineToWo.C via Contains -- I2 containment is exercised separately
  // in containment_test.cpp, on relations that share one coordinate system.
  //
  // This is derived with S already bound to a literal (unlike every check
  // above), not merely for a simpler assertion: composing Coarsen's floor
  // map with a relation whose producer coordinate is itself an *inequality*
  // range tied to the consumer (attn_combine->wo's `128m <= p0 <= 127+128m`,
  // not a plain equality) while S stays a genuine isl parameter hits an isl/
  // barvinok basis-reduction limitation -- confirmed empirically, isl prints
  // "unexpected missing (bounded) solution" (basis_reduction_tab.c) and the
  // resulting quasi-polynomial's pieces no longer cover the relation's own
  // (already-bound) domain, so Eval() correctly refuses to collapse it to a
  // scalar rather than silently returning a wrong number. With S bound
  // before Coarsen runs, the same computation is immediate and exact. This
  // is recorded as residual debt (TileMega_skeleton.md §1.5.1, P4.6): a
  // future Solver that wants to Coarsen with S still symbolic will need
  // either a different isl formulation of the inequality-range case or to
  // bind workload parameters first, as this test now does.
  ParamBinding boundKnown = known;
  boundKnown.Bind("S", 512);
  auto boundEdges = CouplingDerivation().Derive(LlamaDecoderLayer(shape), boundKnown);
  auto boundCombineToWo = Find(boundEdges, "attn_combine", "wo");
  QuasiPolynomial coarseWait = boundCombineToWo.C.Coarsen({1, 4}).Card();
  REQUIRE(coarseWait.Eval(boundKnown) == 1024);  // 4096 / 4

  return 0;
}
