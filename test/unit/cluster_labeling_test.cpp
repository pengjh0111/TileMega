// SPDX-License-Identifier: BSD-3-Clause
// P4.7: every assertion here pins one of the three hard constraints, because
// a labelling that quietly violates one produces a cluster the hardware
// cannot launch rather than a slow kernel.
#include <tilemega/Solver/ClusterLabeling.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace tilemega::solver;

#define REQUIRE(condition)                                                 \
  do {                                                                     \
    if (!(condition)) {                                                    \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

namespace {

std::vector<ClusterNode> Chain(int n, double smem) {
  std::vector<ClusterNode> nodes;
  for (int i = 0; i < n; ++i) nodes.push_back({i, smem});
  return nodes;
}

}  // namespace

int main() {
  ClusterLabeling labeling;

  // A target without clusters must produce the identity, whatever the graph
  // says: the size bound is 1 and there is nothing to trade against it.
  {
    ClusterPlan plan = labeling.Assign(Chain(4, 0), {{0, 1, 100.0}},
                                       {/*max_cluster_size=*/1, 8, 0});
    REQUIRE(plan.clusters == 4);
    REQUIRE(plan.largest == 1);
    REQUIRE(plan.internal_weight == 0.0);
    REQUIRE(plan.total_weight == 100.0);
    REQUIRE(plan.limited_by == "target has no clusters");
  }

  // Heaviest edge first: {0,1} is heavier than {1,2}, so with a size bound of
  // 2 the plan must take it and leave 2 alone rather than sweeping in order.
  {
    ClusterPlan plan = labeling.Assign(
        Chain(3, 0), {{0, 1, 10.0}, {1, 2, 9.0}}, {2, 8, 0});
    REQUIRE(plan.labels[0] == plan.labels[1]);
    REQUIRE(plan.labels[2] != plan.labels[0]);
    REQUIRE(plan.internal_weight == 10.0);
    REQUIRE(plan.total_weight == 19.0);
  }

  // Contraction: after {0,1} merge, their separate edges to 2 must add up, so
  // the merged group beats a pair that looked heavier before contraction.
  {
    ClusterPlan plan = labeling.Assign(
        Chain(4, 0), {{0, 1, 10.0}, {0, 2, 4.0}, {1, 2, 4.0}, {2, 3, 7.0}},
        {3, 8, 0});
    REQUIRE(plan.labels[0] == plan.labels[1]);
    REQUIRE(plan.labels[2] == plan.labels[0]);
    REQUIRE(plan.largest == 3);
    REQUIRE(plan.internal_weight == 18.0);
  }

  // Temporal reach binds on the *group*, not the proposing pair: 0 and 1 are
  // adjacent and 1 and 2 are adjacent, but 0 and 2 are two stages apart, so a
  // reach of 1 admits one merge and refuses the second.
  {
    ClusterPlan plan = labeling.Assign(
        Chain(3, 0), {{0, 1, 10.0}, {1, 2, 9.0}},
        {/*max_cluster_size=*/3, /*max_stage_distance=*/1, 0});
    REQUIRE(plan.largest == 2);
    REQUIRE(plan.limited_by == "temporal reach");
  }

  // Shared memory is a hard budget, and the reason has to be reported: a
  // cluster held back by occupancy is a §4.4 bubble result, not a graph one.
  {
    ClusterPlan plan = labeling.Assign(
        Chain(3, 40000.0), {{0, 1, 10.0}, {1, 2, 9.0}},
        {3, 8, /*smem_budget_bytes=*/100000.0});
    REQUIRE(plan.largest == 2);
    REQUIRE(plan.limited_by == "shared memory budget");
  }

  // Capture is measured against every edge, including ones no cluster could
  // ever take -- otherwise the ratio flatters the heuristic.
  {
    ClusterPlan plan = labeling.Assign(
        Chain(4, 0), {{0, 1, 10.0}, {0, 3, 90.0}}, {2, 1, 0});
    REQUIRE(plan.total_weight == 100.0);
    REQUIRE(plan.internal_weight == 10.0);
    REQUIRE(plan.Capture() > 0.099 && plan.Capture() < 0.101);
  }

  // An edge naming no node is a bug in the caller, and swallowing it would
  // report a capture ratio against a graph smaller than the one passed in.
  {
    bool threw = false;
    try {
      labeling.Assign(Chain(2, 0), {{0, 5, 1.0}}, {2, 8, 0});
    } catch (std::invalid_argument const&) {
      threw = true;
    }
    REQUIRE(threw);
  }

  std::printf("cluster labeling ok\n");
  return 0;
}
