// SPDX-License-Identifier: BSD-3-Clause
//
// P4.7 Label, analytic half: how much of a model's coupling traffic can a
// cluster actually hold?
//
// The weight is §4.3's own `w(A, B) = Volume x Frequency`, taken from the
// derived edges rather than from a shape table -- `metrics.volume` is
// |W_p(y) ^ R_c(x)| in elements for one consumer task and `metrics.count` is
// the Frequency, the number of times the edge is traversed in one forward.
// Their product is the traffic the edge moves in one forward.  Nothing here is a GPU measurement: the question is whether the
// partition is worth wiring at all, and that is answered by the fraction of
// total weight a size- and locality-bounded partition can capture.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ReferenceModels.h>
#include <tilemega/Solver/ClusterLabeling.h>
#include <tilemega/Target/TargetSpec.h>

#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace tilemega::analysis;
using namespace tilemega::solver;

namespace {

/// Elements, not bytes: every tensor in the reference models is fp32, and the
/// ratio the report turns on is scale free.
double Weight(CouplingEdge const& edge, ParamBinding const& known) {
  try {
    return static_cast<double>(edge.metrics.volume.Eval(known)) *
           static_cast<double>(edge.metrics.count.Eval(known));
  } catch (std::exception const&) {
    return -1.0;
  }
}

void Report(char const* label, OperatorGraph const& graph,
            ParamBinding const& known, double smem_per_cta,
            double smem_budget) {
  std::map<std::string, int> index;
  for (auto const& node : graph.nodes) {
    int const next = static_cast<int>(index.size());
    index.emplace(node.name, next);
  }
  std::vector<ClusterNode> nodes;
  for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i)
    nodes.push_back({i, smem_per_cta});

  // Parallel edges between the same operator pair are one coupling as far as
  // a cluster is concerned: the cluster either holds both endpoints or holds
  // neither, so their traffic adds.
  std::map<std::pair<int, int>, double> merged;
  int unevaluable = 0;
  for (auto const& edge : CouplingDerivation().Derive(graph, known)) {
    auto src = index.find(edge.src.name), dst = index.find(edge.dst.name);
    if (src == index.end() || dst == index.end()) continue;
    if (src->second == dst->second) continue;
    double const w = Weight(edge, known);
    if (w < 0) { ++unevaluable; continue; }
    int const a = std::min(src->second, dst->second);
    int const b = std::max(src->second, dst->second);
    merged[{a, b}] += w;
  }
  std::vector<ClusterEdge> edges;
  for (auto const& [pair, weight] : merged)
    edges.push_back({pair.first, pair.second, weight});

  double heaviest = 0, total = 0;
  int adjacent = 0;
  for (auto const& e : edges) {
    total += e.weight;
    if (e.weight > heaviest) heaviest = e.weight;
    if (e.b - e.a == 1) adjacent += 1;
  }
  std::printf("GRAPH model=%s nodes=%zu edges=%zu unevaluable=%d "
              "adjacent=%d total_weight=%.0f heaviest=%.0f\n",
              label, graph.nodes.size(), edges.size(), unevaluable, adjacent,
              total, heaviest);

  for (int size : {2, 4, 8, 16}) {
    for (int reach : {1, 2, 4, 1 << 20}) {
      ClusterConstraints limits;
      limits.max_cluster_size = size;
      limits.max_stage_distance = reach;
      limits.smem_budget_bytes = smem_budget;
      ClusterPlan plan = ClusterLabeling().Assign(nodes, edges, limits);
      std::printf("PLAN model=%s size=%d reach=%d clusters=%d largest=%d "
                  "capture=%.4f limited_by=%s\n",
                  label, size, reach, plan.clusters, plan.largest,
                  plan.Capture(), plan.limited_by.c_str());
    }
  }
}

}  // namespace

int main() {
  DecoderShape shape;
  ParamBinding known = DecoderShape::Table27Theta();
  for (auto const& [name, value] : DecoderShape::Table27G().values)
    known.Bind(name, value);
  known.Bind("S", 512).Bind("L_s", 1024).Bind("past", 512);

  // The budget is the distributed shared-memory window a cluster addresses at
  // one CTA per SM: `max_smem_per_sm` per member.  It is read off the probed
  // target rather than written down, and on a target without clusters
  // `max_cluster_size` is 1, so the labelling degenerates by construction.
  auto target = tilemega::TargetSpec::Probe();
  double const smem_per_cta = 10496;  // the harness TaskSmem union, measured
  double const budget =
      static_cast<double>(target.res.max_smem_per_sm) * target.res.max_cluster_size;
  std::printf("TARGET arch=%s cluster=%d max_cluster_size=%d "
              "max_smem_per_sm=%d smem_per_cta=%.0f budget=%.0f\n",
              target.arch_tag.c_str(), target.caps.cluster ? 1 : 0,
              target.res.max_cluster_size, target.res.max_smem_per_sm,
              smem_per_cta, budget);

  Report("gqa2", LlamaStack(shape, /*layers=*/2), known, smem_per_cta, budget);
  Report("mha4", MhaModel(shape, /*layers=*/4), known, smem_per_cta, budget);
  return 0;
}
