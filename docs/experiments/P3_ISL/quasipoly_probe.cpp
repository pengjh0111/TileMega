// SPDX-License-Identifier: BSD-3-Clause
// Part 3 acceptance (d): a coupling whose wait is genuinely a piecewise
// quasi-polynomial -- the case that makes barvinok load-bearing rather than
// merely a different spelling of the old closed form.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ReferenceModels.h>

#include <cstdio>

using namespace tilemega::analysis;

namespace {
ParamBinding Known(long rows) {
  ParamBinding known = DecoderShape::Table27Theta();
  for (auto const& [name, value] : DecoderShape::Table27G().values)
    known.Bind(name, value);
  known.Bind("S", rows);
  return known;
}

void Report(long producer_tile, long consumer_tile, long rows) {
  DecoderShape shape;
  ParamBinding known = Known(rows);
  auto edges = CouplingDerivation().Derive(
      MisalignedTileModel(shape, producer_tile, consumer_tile), known);
  CouplingEdge const& edge = edges.front();
  std::printf("### producer_tile=%ld consumer_tile=%ld (S=%ld)\n\n",
              producer_tile, consumer_tile, rows);
  std::printf("- exact: %s, relaxation: %s, tier: %s\n", edge.exact ? "yes" : "no",
              edge.relaxation.empty() ? "(none)" : edge.relaxation.c_str(),
              ToString(edge.tier).c_str());
  std::printf("- C    = `%s`\n", edge.C.ToString().c_str());
  std::printf("- wait = `%s`\n", edge.metrics.wait.ToString().c_str());
  try {
    std::printf("- wait as one scalar: %ld\n\n", edge.metrics.wait.Eval(known));
  } catch (std::exception const& error) {
    std::printf("- wait as one scalar: **refused** -- %s\n\n", error.what());
  }
}
}  // namespace

int main() {
  std::printf("## Piecewise quasi-polynomial wait\n\n");
  std::printf("A producer tiling rows by `producer_tile` feeding a consumer\n"
              "tiling the same axis by `consumer_tile`. When the tiles do not\n"
              "divide one another, the number of producer blocks a consumer\n"
              "block straddles varies periodically with the consumer\n"
              "coordinate, so wait is a genuine piecewise quasi-polynomial.\n\n");
  Report(96, 160, 1536);   // straddles 2 or 3, period lcm(96,160)/160 = 3
  Report(100, 128, 1600);  // straddles 2 or 3, a second instance
  Report(96, 192, 1536);   // aligned control: exactly 2, collapses to a scalar
  Report(128, 128, 1536);  // identical tiles: exactly 1
  return 0;
}
