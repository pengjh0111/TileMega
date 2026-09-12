// SPDX-License-Identifier: BSD-3-Clause
// EX-E1: the Plan is the only thing that decides a worker's queue.  Every
// assertion here pins one way of losing that -- sigma quietly re-sorted into
// stage order, a sigma that is not a total order, an over-resident grid, or a
// same-worker producer whose poll would be elided on the strength of the stage
// order rather than sigma (§5.7.2, §5.7.3 L-a/L-b/L-c).
#include <tilemega/Codegen/ResidentSchedule.h>
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Solver/PlanMaterialize.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tilemega;
using solver::MaterializedPlan;

#define REQUIRE(condition)                                                 \
  do {                                                                     \
    if (!(condition)) {                                                    \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

namespace {

/// Two stages of `per_stage` tasks each, with `edges` given as
/// (producer node, consumer node) over the flattened node ids.
codegen::RuntimeTaskGraph MakeGraph(
    int per_stage, std::vector<std::pair<int, int>> const& edges) {
  codegen::RuntimeTaskGraph graph;
  graph.stage_offsets = {0, per_stage, 2 * per_stage};
  graph.successors.assign(2 * per_stage, {});
  for (auto const& edge : edges) graph.successors[edge.first].push_back(edge.second);
  graph.preferred_worker.assign(2 * per_stage, 0);
  return graph;
}

/// pi and sigma straight from the caller, one entry per node in stage order.
MaterializedPlan MakePlan(int per_stage, std::vector<int> const& owner,
                          std::vector<int> const& slot) {
  MaterializedPlan plan;
  plan.owner.assign(2, {});
  plan.slot.assign(2, {});
  for (int stage = 0; stage < 2; ++stage)
    for (int task = 0; task < per_stage; ++task) {
      plan.owner[stage].push_back(owner[stage * per_stage + task]);
      plan.slot[stage].push_back(slot[stage * per_stage + task]);
    }
  return plan;
}

}  // namespace

int main() {
  std::string error;

  {
    // An arbitrary legal sigma that interleaves the two stages on both workers.
    // Worker 0 runs s0t0, s1t0, s0t2, s1t2 and worker 1 runs s1t1, s0t1, in
    // that order; nothing about it is stage major.
    int const n = 3;
    auto plan = MakePlan(n, {0, 1, 0, 0, 1, 0}, {0, 1, 2, 1, 0, 3});
    REQUIRE(solver::BuildPlanQueues({n, n}, 2, &plan, &error));
    std::vector<std::pair<std::uint32_t, int>> const expect_w0 = {
        {0, 0}, {1, 0}, {0, 2}, {1, 2}};
    std::vector<std::pair<std::uint32_t, int>> const expect_w1 = {{1, 1}, {0, 1}};
    REQUIRE(plan.queue[0].size() == expect_w0.size());
    for (std::size_t i = 0; i < expect_w0.size(); ++i)
      REQUIRE(plan.queue[0][i].stage == expect_w0[i].first &&
              plan.queue[0][i].logical == expect_w0[i].second);
    REQUIRE(plan.queue[1].size() == expect_w1.size());
    for (std::size_t i = 0; i < expect_w1.size(); ++i)
      REQUIRE(plan.queue[1][i].stage == expect_w1[i].first &&
              plan.queue[1][i].logical == expect_w1[i].second);

    // That sigma is legal against a DAG whose edges agree with it.
    auto const graph = MakeGraph(n, {{0, 3}, {4, 1}, {2, 5}});
    REQUIRE(solver::CheckPlanLegality(graph, plan, &error));
  }

  {
    // A sigma that is not a dense order per worker has no executable queue: two
    // tasks claiming slot 0 on the same worker is a tie the executor cannot break.
    int const n = 2;
    auto plan = MakePlan(n, {0, 0, 0, 0}, {0, 1, 0, 2});
    REQUIRE(!solver::BuildPlanQueues({n, n}, 1, &plan, &error));
    REQUIRE(error.find("not a dense order") != std::string::npos);
  }

  {
    // L-c negative control.  s0t0 produces for s1t0 and both land on worker 0,
    // but the producer holds the *later* sigma.  The harness elides a poll
    // exactly when the producer shares the worker, so accepting this plan would
    // drop the only synchronization on that edge rather than merely reorder it.
    int const n = 1;
    auto plan = MakePlan(n, {0, 0}, {1, 0});
    REQUIRE(solver::BuildPlanQueues({n, n}, 1, &plan, &error));
    auto const graph = MakeGraph(n, {{0, 1}});
    REQUIRE(!solver::CheckPlanLegality(graph, plan, &error));
    REQUIRE(error.rfind("L-c:", 0) == 0);

    // Equal sigma is rejected too.  BuildPlanQueues would already have refused
    // this plan as non-dense, so the check is reached only by a caller that
    // skips it -- but `strictly smaller` is what L-c says, and a producer that
    // merely shares the consumer's slot is not ordered before it.
    auto tied = MakePlan(n, {0, 0}, {0, 0});
    REQUIRE(!solver::CheckPlanLegality(graph, tied, &error));
    REQUIRE(error.rfind("L-c:", 0) == 0);

    // The same graph with the producer first is accepted, so the rejection is
    // about sigma and not about the edge existing.
    auto legal = MakePlan(n, {0, 0}, {0, 1});
    REQUIRE(solver::BuildPlanQueues({n, n}, 1, &legal, &error));
    REQUIRE(solver::CheckPlanLegality(graph, legal, &error));
  }

  {
    // L-a: neither task edge is same-worker, so L-c has nothing to say, yet the
    // union with the queue edges closes a cycle
    // s0t0 ->(queue) s0t1 ->(task) s1t0 ->(queue) s1t1 ->(task) s0t0.
    int const n = 2;
    auto const graph = MakeGraph(n, {{1, 2}, {3, 0}});
    auto plan = MakePlan(n, {0, 0, 1, 1}, {0, 1, 0, 1});
    REQUIRE(solver::BuildPlanQueues({n, n}, 2, &plan, &error));
    REQUIRE(!solver::CheckPlanLegality(graph, plan, &error));
    REQUIRE(error.rfind("L-a:", 0) == 0);

    // Dropping the back edge leaves the same queues legal.
    auto const acyclic = MakeGraph(n, {{1, 2}});
    REQUIRE(solver::CheckPlanLegality(acyclic, plan, &error));
  }

  {
    // L-b: a plan is only materialized onto a resident grid (§8.7).
    REQUIRE(codegen::ResidentScheduleLegal(true, 64, 64));
    REQUIRE(!codegen::ResidentScheduleLegal(true, 65, 64));
    REQUIRE(!codegen::ResidentScheduleLegal(false, 64, 64));
    REQUIRE(!codegen::ResidentScheduleLegal(true, 64, 0));
  }

  {
    // pi outside the grid is rejected before sigma is even looked at.
    int const n = 1;
    auto plan = MakePlan(n, {0, 2}, {0, 0});
    REQUIRE(!solver::BuildPlanQueues({n, n}, 2, &plan, &error));
    REQUIRE(error.find("outside the grid") != std::string::npos);
  }

  {
    // The stage-major modes stay reachable through the same path: legacy
    // grid-stride on a 2-worker grid gives worker 0 the even tasks of each
    // stage, numbered along the stage order.
    solver::PlanRequest request;
    request.mode = dialect::PlacementMode::kLegacyGridStride;
    request.grid = 2;
    request.counts = {4, 4};
    request.stage_order = {0, 1};
    request.physical_worker = {0, 1};
    MaterializedPlan plan;
    REQUIRE(solver::MaterializePlanPlacement(request, &plan, &error));
    REQUIRE(plan.owner[0] == std::vector<int>({0, 1, 0, 1}));
    REQUIRE(plan.slot[0] == std::vector<int>({0, 0, 1, 1}));
    REQUIRE(plan.slot[1] == std::vector<int>({2, 2, 3, 3}));
    REQUIRE(plan.queue[0].size() == 4 && plan.queue[0][2].stage == 1u &&
            plan.queue[0][2].logical == 0);

    // rotate carries its base as the prefix sum of the counts along the stage
    // order, mod grid: stage 1 starts where stage 0's four tasks stopped.
    request.mode = dialect::PlacementMode::kRotate;
    MaterializedPlan rotated;
    REQUIRE(solver::MaterializePlanPlacement(request, &rotated, &error));
    REQUIRE(rotated.rotate_base == std::vector<int>({0, 0}));
    request.counts = {3, 4};
    REQUIRE(solver::MaterializePlanPlacement(request, &rotated, &error));
    REQUIRE(rotated.rotate_base == std::vector<int>({0, 1}));
    REQUIRE(rotated.owner[1] == std::vector<int>({1, 0, 1, 0}));

    // EX-S2's modes are refused rather than silently materialized as legacy.
    request.mode = dialect::PlacementMode::kEft;
    MaterializedPlan unused;
    REQUIRE(!solver::MaterializePlanPlacement(request, &unused, &error));
    REQUIRE(error.find("not materialized yet") != std::string::npos);
  }

  std::printf("plan_contract_test: ok\n");
  return 0;
}
