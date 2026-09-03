// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §4.3 cluster-label assignment (P4.7 Label).
//
// A cluster is a set of CTAs the hardware co-schedules on one GPC and lets
// talk through distributed shared memory.  Labelling decides which task
// spaces share one, and the decision is a partition problem with three
// constraints that are all hard, not preferences:
//
//   size      a cluster may hold at most `Caps<Arch>::kMaxClusterSize` CTAs
//             (8 on sm_90/sm_120, 1 wherever `kCluster` is false, which makes
//             the labelling on such a target the identity by construction).
//   temporal  the CTAs of a cluster are resident together, so two task spaces
//             can only share one if they are close enough in the stage order
//             to be alive at the same time.
//   shared    they are resident together on one GPC, so their shared memory
//   memory    has to fit there at once.  This is the term that couples the
//             labelling to §4.4's bubble: a cluster that costs occupancy buys
//             DSMEM latency and pays in tail.
//
// The objective is §4.3's `w(A, B) = Volume x Frequency`, the bytes a coupling
// edge moves per iteration times how often it fires -- exactly the traffic a
// cluster-scoped release keeps inside the GPC.  Maximizing the weight kept
// inside clusters under a size bound is NP-hard (it contains maximum-weight
// k-way matching), so this is the standard heavy-edge agglomeration used by
// graph coarseners: repeatedly merge the heaviest admissible pair.  It is a
// heuristic and is labelled as one; `ClusterPlan::internal_weight` against
// `total_weight` says how much of the available traffic it actually captured,
// so the gap to the unattainable optimum is visible rather than assumed.
#pragma once

#include <string>
#include <vector>

namespace tilemega::solver {

/// One task space, in the order the solver numbers them.
struct ClusterNode {
  int stage = 0;          ///< position in the stage order, for temporal reach
  double smem_bytes = 0;  ///< shared memory one CTA of this task space needs
};

/// `weight` is Volume x Frequency, evaluated at the solver's parameter
/// binding.  Direction does not matter: a cluster keeps the traffic inside
/// whichever way it flows.
struct ClusterEdge {
  int a = 0;
  int b = 0;
  double weight = 0;
};

struct ClusterConstraints {
  /// `Caps<Arch>::kMaxClusterSize`.  1 disables clustering entirely, which is
  /// what every target without `kCluster` reports.
  int max_cluster_size = 1;
  /// Stages apart two task spaces may be and still be co-resident.
  int max_stage_distance = 1;
  /// Shared memory available to one cluster.  0 means unconstrained.
  double smem_budget_bytes = 0;
};

struct ClusterPlan {
  std::vector<int> labels;       ///< node -> cluster id, ids dense from 0
  int clusters = 0;
  int largest = 1;
  double internal_weight = 0;    ///< weight on edges inside a cluster
  double total_weight = 0;       ///< weight on all edges
  /// Reason the plan stopped short, empty when the size bound was the only
  /// thing that bound.  Recorded rather than silently absorbed.
  std::string limited_by;

  double Capture() const {
    return total_weight > 0 ? internal_weight / total_weight : 0.0;
  }
};

class ClusterLabeling {
 public:
  /// Heavy-edge agglomeration under the three constraints.  Deterministic:
  /// ties break on (a, b) so the same graph always yields the same plan.
  /// Throws std::invalid_argument on an edge that names no node, rather than
  /// dropping it and reporting a capture ratio against a graph that was
  /// quietly smaller than the one passed in.
  ClusterPlan Assign(std::vector<ClusterNode> const& nodes,
                     std::vector<ClusterEdge> const& edges,
                     ClusterConstraints const& limits) const;

  /// The Phase-3 signature, kept so existing callers still compile: `count`
  /// nodes with no edges and no temporal structure, cut into runs of
  /// `max_cluster_size`.  With no weights there is nothing to optimize, so
  /// this is the blocked labelling and not a degraded version of the above.
  std::vector<int> Assign(int node_count, int max_cluster_size) const;
};

}  // namespace tilemega::solver
