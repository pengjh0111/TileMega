// SPDX-License-Identifier: BSD-3-Clause
//
// P4.6 ablation, analytic half: what does event granularity kappa buy, and
// what does it cost, on the coupling relations the derivation actually
// produces?  §2.3's C_kappa = floor(./kappa) o C is applied to every range
// (producer) dimension of every derived edge of both accepted models.
//
// Two numbers per (edge, kappa), and they move in opposite directions:
//
//   waits(kappa)    = |C_kappa(x)|, the events one consumer task polls.
//   overwait(kappa) = kappa*waits(kappa) - waits(1), the producer tasks the
//                     consumer is forced to wait for beyond the ones it reads.
//
// kappa = 1 is the finest event graph (overwait 0, waits maximal); kappa >=
// the producer extent is one event per stage, which is what the L2 megakernel
// runs today (waits = 1, overwait maximal).  The ablation is whether anything
// in between is better, and the answer has to come from these two columns plus
// the measured cost of a wait -- see run.sh for the hardware half.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ReferenceModels.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace tilemega::analysis;

namespace {

constexpr long kKappa[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 1024};

/// `waits` at this kappa, or -1 when the count is not a single value over the
/// consumer's task space (reported rather than averaged away).
///
/// Only the *last* producer dimension is coarsened, because that is the one
/// the megakernel can actually group: an event is owned by a CTA, and the CTA
/// index runs along the fastest-varying axis of the producer's task space.
/// Coarsening every axis at once would describe an event scheme the runtime
/// has no way to index.
long Waits(CouplingEdge const& edge, long kappa, ParamBinding const& known) {
  std::vector<long> per_dim(edge.C.RangeDimNames().size(), 1);
  if (per_dim.empty()) return -1;
  per_dim.back() = kappa;
  try {
    return QuasiPolynomial::Card(edge.C.Coarsen(per_dim)).Eval(known);
  } catch (std::exception const&) {
    return -1;
  }
}

void Report(char const* label, OperatorGraph const& graph,
            ParamBinding const& known) {
  std::vector<CouplingEdge> edges = CouplingDerivation().Derive(graph, known);
  std::vector<long> total(std::size(kKappa), 0);
  std::vector<long> over(std::size(kKappa), 0);
  int counted = 0, skipped = 0;
  for (auto const& edge : edges) {
    if (edge.C.empty()) continue;
    long const fine = Waits(edge, 1, known);
    if (fine < 0) {
      ++skipped;
      std::printf("EDGE model=%s src=%s dst=%s waits=varies\n", label,
                  edge.src.name.c_str(), edge.dst.name.c_str());
      continue;
    }
    ++counted;
    std::printf("EDGE model=%s src=%s dst=%s", label, edge.src.name.c_str(),
                edge.dst.name.c_str());
    for (std::size_t i = 0; i < std::size(kKappa); ++i) {
      long const w = Waits(edge, kKappa[i], known);
      if (w < 0) { std::printf(" k%ld=varies", kKappa[i]); continue; }
      total[i] += w;
      over[i] += kKappa[i] * w - fine;
      std::printf(" k%ld=%ld/%ld", kKappa[i], w, kKappa[i] * w - fine);
    }
    std::printf("\n");
  }
  std::printf("CURVE model=%s edges=%d skipped=%d\n", label, counted, skipped);
  for (std::size_t i = 0; i < std::size(kKappa); ++i)
    std::printf("POINT model=%s kappa=%ld waits=%ld overwait=%ld\n", label,
                kKappa[i], total[i], over[i]);
}

}  // namespace

int main() {
  DecoderShape shape;
  ParamBinding known = DecoderShape::Table27Theta();
  for (auto const& [name, value] : DecoderShape::Table27G().values)
    known.Bind(name, value);
  known.Bind("S", 512).Bind("L_s", 1024).Bind("past", 512);
  Report("gqa2", LlamaStack(shape, /*layers=*/2), known);
  Report("mha4", MhaModel(shape, /*layers=*/4), known);
  return 0;
}
