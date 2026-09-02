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
  // This is the operation AffineRelation could not express at all (it had no
  // composition operator), so it is the migration's own reason to exist.
  QuasiPolynomial coarseWait = combineToWo.C.Coarsen({1, 4}).Card();
  REQUIRE(coarseWait.Eval(known) == 1024);  // 4096 / 4

  // Coarsening is a group homomorphism on the granularity, and the identity
  // at kappa = 1. Both were false at first: the fresh output names Coarsen
  // generated collided with the range names of an already-coarsened relation,
  // and isl reads the resulting `q1 = floord(q1, 2)` as a constraint on one
  // variable whose only solution is 0 -- silently collapsing that coordinate
  // to a point rather than halving it. Keeping these two laws asserted is
  // what makes a repeat of that failure impossible to miss.
  REQUIRE(combineToWo.C.Coarsen({1, 1}) == combineToWo.C);
  REQUIRE(combineToWo.C.Coarsen({1, 2}).Coarsen({1, 2}) ==
          combineToWo.C.Coarsen({1, 4}));

  // A coupling whose wait is genuinely a piecewise quasi-polynomial, which
  // is what makes barvinok load-bearing rather than merely equivalent to the
  // old closed form. Producer tiles rows by 96, consumer by 160: a consumer
  // block straddles 2 or 3 producer blocks depending on its position, with
  // period lcm(96,160)/160 = 3.
  //
  // Two things are being asserted at once. First, the edge is derived
  // *exactly* -- the two-sided overlap condition is affine, so isl carries
  // it and no relaxation is needed; the pre-migration path had no way to
  // express that condition or to count over it, so it relaxed this whole
  // class of edge to the full producer axis and raised the tier. Second,
  // the resulting wait genuinely has no single scalar value, and Eval says
  // so instead of silently returning one of the two.
  ParamBinding misaligned = KnownBinding();
  misaligned.Bind("S", 1536);
  auto misalignedEdges = CouplingDerivation().Derive(
      MisalignedTileModel(shape, /*producer_tile=*/96, /*consumer_tile=*/160),
      misaligned);
  REQUIRE(misalignedEdges.size() == 1);
  CouplingEdge const& straddle = misalignedEdges.front();
  REQUIRE(straddle.exact);
  REQUIRE(straddle.relaxation.empty());
  REQUIRE(straddle.tier == Tier::kAffine);
  bool refused = false;
  try {
    (void)straddle.metrics.wait.Eval(misaligned);
  } catch (std::out_of_range const&) {
    refused = true;  // correctly not collapsible to one number
  }
  REQUIRE(refused);
  // The piecewise form itself carries the period-3 structure, so a consumer
  // that wants the real per-position count can still read it.
  REQUIRE(straddle.metrics.wait.ToString().find("floor") != std::string::npos);

  // An aligned control with the same machinery: 96 into 192 divides exactly,
  // so wait is the constant 2 and does collapse.
  auto alignedEdges = CouplingDerivation().Derive(
      MisalignedTileModel(shape, /*producer_tile=*/96, /*consumer_tile=*/192),
      misaligned);
  REQUIRE(alignedEdges.front().exact);
  REQUIRE(alignedEdges.front().metrics.wait.Eval(misaligned) == 2);

  return 0;
}
