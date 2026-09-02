// SPDX-License-Identifier: BSD-3-Clause
// Part 3 acceptance (b): Coarsen (§2.3's C_kappa = floor(./kappa) o C) is
// implementable at all -- AffineRelation had no composition operator, so this
// is the operation the isl migration exists to unlock -- and P4.6's flagged
// question: does isl blow up on parameterised integer division as kappa and
// nesting grow? Reports expression size (isl text length, piece count) so the
// answer is measured, not asserted.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ReferenceModels.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace tilemega::analysis;

namespace {

ParamBinding Known(bool bind_workload) {
  ParamBinding known = DecoderShape::Table27Theta();
  for (auto const& [name, value] : DecoderShape::Table27G().values)
    known.Bind(name, value);
  if (bind_workload) known.Bind("S", 512).Bind("L_s", 1024).Bind("past", 512);
  return known;
}

CouplingEdge Find(std::vector<CouplingEdge> const& edges, std::string const& src,
                  std::string const& dst) {
  for (auto const& edge : edges)
    if (edge.src.name == src && edge.dst.name == dst) return edge;
  std::fprintf(stderr, "no edge %s -> %s\n", src.c_str(), dst.c_str());
  std::exit(1);
}

/// Rough expression-size proxies: isl's own canonical text length, and how
/// many disjuncts/pieces the result carries (the thing that would "explode").
std::size_t Pieces(std::string const& text) {
  std::size_t count = 1;
  for (char c : text)
    if (c == ';') ++count;
  return count;
}

void Report(char const* edge_name, CouplingEdge const& edge,
            std::vector<long> const& kappa, ParamBinding const& known) {
  CouplingRelation coarse = edge.C.Coarsen(kappa);
  QuasiPolynomial wait = coarse.Card();
  std::string kappa_text;
  for (std::size_t i = 0; i < kappa.size(); ++i) {
    if (i) kappa_text += ",";
    kappa_text += std::to_string(kappa[i]);
  }
  std::printf("| %s | {%s} | %zu | %zu | %zu | %zu | ", edge_name,
              kappa_text.c_str(), coarse.ToString().size(),
              Pieces(coarse.ToString()), wait.ToString().size(),
              Pieces(wait.ToString()));
  try {
    std::printf("%ld |\n", wait.Eval(known));
  } catch (std::exception const& error) {
    std::printf("(not a scalar: %s) |\n", error.what());
  }
}

}  // namespace

int main() {
  DecoderShape shape;
  ParamBinding known = Known(/*bind_workload=*/true);
  auto edges = CouplingDerivation().Derive(LlamaDecoderLayer(shape), known);

  std::printf("## Coarsen: C_kappa = floor(./kappa) o C\n\n");
  std::printf("Workload parameters bound (S=512, L_s=1024, past=512); see the\n"
              "symbolic-parameter note below for why.\n\n");
  std::printf("| edge | kappa | |C_kappa| text | C_kappa pieces | wait text | "
              "wait pieces | wait |\n");
  std::printf("|---|---|---:|---:|---:|---:|---:|\n");

  // Row 7 (attn_combine -> wo): wait = Tm * n_h = 4096, producer coordinate
  // is 2-D, so kappa can be swept on the n_h axis.
  CouplingEdge row7 = Find(edges, "attn_combine", "wo");
  for (long k : {1L, 2L, 4L}) Report("attn_combine->wo", row7, {1, k}, known);
  // Sweep the other (row-block) axis too, and both at once.
  for (long k : {2L, 4L}) Report("attn_combine->wo", row7, {k, 1}, known);
  Report("attn_combine->wo", row7, {4, 4}, known);

  // Row 1 (rmsnorm1 -> wq): 1-D producer coordinate, wait = 1 -- coarsening
  // cannot reduce it below 1, which is the expected floor.
  CouplingEdge row1 = Find(edges, "rmsnorm1", "wq");
  for (long k : {1L, 2L, 4L}) Report("rmsnorm1->wq", row1, {k}, known);

  // Row 12 (silu -> wdown): wait = 112 over a 2-D producer coordinate.
  CouplingEdge row12 = Find(edges, "silu", "wdown");
  for (long k : {1L, 2L, 4L}) Report("silu->wdown", row12, {1, k}, known);

  // I2: coarsening is a reindexing of the range, so C_kappa is not compared
  // to C by IsSubset (different coordinate systems). What must hold is that
  // C_kappa is itself a valid relation over the coarse space and that
  // successive coarsening composes: floor(floor(x/2)/2) == floor(x/4).
  CouplingRelation twice = row7.C.Coarsen({1, 2}).Coarsen({1, 2});
  CouplingRelation once = row7.C.Coarsen({1, 4});
  std::printf("\nfloor(floor(./2)/2) == floor(./4): %s\n",
              (twice == once) ? "yes" : "NO");
  std::printf("kappa=1 is the identity: %s\n",
              (row7.C.Coarsen({1, 1}) == row7.C) ? "yes" : "NO");

  // The symbolic-parameter limitation, measured rather than asserted.
  std::printf("\n## Same sweep with S left symbolic\n\n");
  ParamBinding symbolic = Known(/*bind_workload=*/false);
  auto symbolic_edges =
      CouplingDerivation().Derive(LlamaDecoderLayer(shape), symbolic);
  CouplingEdge symbolic_row7 = Find(symbolic_edges, "attn_combine", "wo");
  std::printf("| edge | kappa | |C_kappa| text | C_kappa pieces | wait text | "
              "wait pieces | wait |\n");
  std::printf("|---|---|---:|---:|---:|---:|---:|\n");
  for (long k : {1L, 2L, 4L})
    Report("attn_combine->wo (S symbolic)", symbolic_row7, {1, k}, symbolic);
  return 0;
}
