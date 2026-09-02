// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/CandidateGenerator.h>

#include <algorithm>
#include <tuple>
#include <utility>

namespace tilemega::solver {
namespace {

// The native shape of the SIMT collective: a 16x16 thread MMA and the
// shortest cp.async row the copy layout admits. Every candidate is this tile
// doubled along some axes, which is why the walk needs no shape table.
constexpr int kNativeTileMN = 16;
constexpr int kNativeTileK = 8;
constexpr int kNativeStages = 2;

int CeilDiv(int a, int b) { return b > 0 ? (a + b - 1) / b : 0; }

}  // namespace

CandidateGenerator::CandidateGenerator(TargetSpec target, Envelope envelope)
    : target_(std::move(target)), envelope_(envelope) {}

std::vector<BackendCandidate> CandidateGenerator::Enumerate(Stats* stats) const {
  std::vector<BackendCandidate> result;
  Stats local;
  int const budget = target_.res.max_dynamic_smem_per_cta;
  for (int m = kNativeTileMN; m <= envelope_.max_tile_mn; m *= 2)
    for (int n = kNativeTileMN; n <= envelope_.max_tile_mn; n *= 2)
      for (int k = kNativeTileK; k <= envelope_.max_tile_k; k *= 2)
        for (int s = kNativeStages; s <= envelope_.max_stages; ++s)
          ++local.cartesian;

  // Expansion order is canonical: an axis may only be doubled after the axes
  // before it are done growing, so every shape is reached exactly once and no
  // visited set is needed. Shared storage is monotone in all four axes, which
  // is what makes the wall a subtree cut rather than a per-shape filter.
  struct Node {
    int m, n, k, stages, axis;
  };
  std::vector<Node> work{{kNativeTileMN, kNativeTileMN, kNativeTileK,
                          kNativeStages, 0}};
  while (!work.empty()) {
    Node node = work.back();
    work.pop_back();
    ++local.touched;
    BackendTraits traits = SimtF32Traits(node.m, node.n, node.k, node.stages);
    int const smem = SimtF32SmemBytes(node.m, node.n, node.k, node.stages);
    if (smem > budget) {
      ++local.wall_pruned;
      continue;
    }
    BackendCandidate candidate(traits);
    if (!traits.shape_legal)
      ++local.shape_rejected;
    else if (candidate.isLegal(target_)) {
      ++local.legal;
      result.push_back(candidate);
    } else {
      // Legal shape, rejected by a target budget other than smem.
      ++local.shape_rejected;
    }
    // The shape rules are not monotone (16x16 is illegal at k=8 but 32x32 is
    // legal), so a rejected shape still has to be expanded; only the wall cuts.
    for (int axis = node.axis; axis < 4; ++axis) {
      Node next = node;
      next.axis = axis;
      switch (axis) {
        case 0: next.m *= 2; if (next.m > envelope_.max_tile_mn) continue; break;
        case 1: next.n *= 2; if (next.n > envelope_.max_tile_mn) continue; break;
        case 2: next.k *= 2; if (next.k > envelope_.max_tile_k) continue; break;
        default: ++next.stages; if (next.stages > envelope_.max_stages) continue;
      }
      work.push_back(next);
    }
  }
  std::sort(result.begin(), result.end(),
            [](BackendCandidate const& a, BackendCandidate const& b) {
              auto key = [](BackendCandidate const& c) {
                return std::tuple(c.traits().tile_m, c.traits().tile_n,
                                  c.traits().tile_k, c.traits().stages);
              };
              return key(a) < key(b);
            });
  if (stats) *stats = local;
  return result;
}

AnalyticalCost CandidateGenerator::Analyze(BackendCandidate const& candidate,
                                           GemmProblem const& problem) const {
  AnalyticalCost cost;
  BackendTraits const& traits = candidate.traits();
  if (!candidate.isLegal(target_) || problem.m <= 0 || problem.n <= 0 ||
      problem.k <= 0)
    return cost;
  cost.ctas = CeilDiv(problem.m, traits.tile_m) * CeilDiv(problem.n, traits.tile_n);
  cost.co_resident = candidate.CoResidentPerSM(target_);
  int const slots = std::max(1, target_.res.num_sms * cost.co_resident);
  cost.waves = CeilDiv(cost.ctas, slots);
  cost.k_iters = CeilDiv(problem.k, traits.tile_k);
  // One FFMA per thread per cycle: the UniversalFMA atom issues on the CUDA
  // cores, so the tile's own work maps to cycles without a machine constant.
  cost.cta_cycles = static_cast<double>(traits.tile_m) * traits.tile_n *
                    traits.tile_k / traits.threads;
  cost.cycles = static_cast<double>(cost.waves) * cost.k_iters * cost.cta_cycles;
  return cost;
}

double CandidateGenerator::RankKey(BackendCandidate const& candidate,
                                   std::vector<GemmProblem> const& problems) const {
  double total = 0;
  for (auto const& problem : problems) total += Analyze(candidate, problem).cycles;
  return total;
}

std::vector<BackendCandidate> CandidateGenerator::TopK(
    std::vector<BackendCandidate> const& candidates,
    std::vector<GemmProblem> const& problems, int k) const {
  std::vector<std::pair<double, BackendCandidate>> ranked;
  ranked.reserve(candidates.size());
  for (auto const& candidate : candidates)
    ranked.emplace_back(RankKey(candidate, problems), candidate);
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](auto const& a, auto const& b) {
                     if (a.first != b.first) return a.first < b.first;
                     return a.second.smemBytes() < b.second.smemBytes();
                   });
  std::vector<BackendCandidate> result;
  for (auto const& entry : ranked) {
    if (static_cast<int>(result.size()) >= k) break;
    result.push_back(entry.second);
  }
  return result;
}

std::vector<TileCandidate> CandidateGenerator::Generate() const {
  std::vector<TileCandidate> result;
  for (auto const& candidate : Enumerate())
    result.push_back({candidate.traits().tile_m, candidate.traits().tile_n,
                      candidate.traits().tile_k, candidate.traits().stages});
  return result;
}

}  // namespace tilemega::solver
