// SPDX-License-Identifier: BSD-3-Clause
// EX-S2c: critical-chain placement.  What is pinned here: a chain is ranked by
// the time it will take once chained rather than by how many tasks it holds;
// the spine lands whole on one worker, which is the whole point (its internal
// edges then cost no hop); sigma is earliest-start order, so L-c holds without
// a sort (§5.7.3); and §6.3's cycle -- two chains with cross edges in both
// directions -- closes only in the contracted graph, so both chains still land
// whole and nothing is split.
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Solver/ChainPlacement.h>
#include <tilemega/Solver/PlanMaterialize.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tilemega;
using solver::ChainRequest;
using solver::ChainSchedule;
using solver::MaterializedPlan;
using solver::PlanRequest;

#define REQUIRE(condition)                                                 \
  do {                                                                     \
    if (!(condition)) {                                                    \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

namespace {

codegen::RuntimeTaskGraph MakeGraph(
    std::vector<int> const& counts,
    std::vector<std::pair<int, int>> const& edges) {
  codegen::RuntimeTaskGraph graph;
  graph.stage_offsets.push_back(0);
  for (int count : counts)
    graph.stage_offsets.push_back(graph.stage_offsets.back() + count);
  graph.successors.assign(graph.stage_offsets.back(), {});
  for (auto const& edge : edges) graph.successors[edge.first].push_back(edge.second);
  graph.preferred_worker.assign(graph.stage_offsets.back(), 0);
  return graph;
}

/// Every worker's queue is dense in sigma, no two tasks share a slot, and a
/// same-worker producer holds the smaller slot (L-c).
void CheckQueues(codegen::RuntimeTaskGraph const& graph, ChainSchedule const& schedule,
                 int nodes, int grid) {
  std::vector<std::vector<int>> by_slot(grid);
  for (int node = 0; node < nodes; ++node) {
    int const w = schedule.worker[node];
    REQUIRE(w >= 0 && w < grid);
    REQUIRE(schedule.slot[node] >= 0);
    if (static_cast<int>(by_slot[w].size()) <= schedule.slot[node])
      by_slot[w].resize(schedule.slot[node] + 1, -1);
    REQUIRE(by_slot[w][schedule.slot[node]] == -1);
    by_slot[w][schedule.slot[node]] = node;
  }
  for (int w = 0; w < grid; ++w)
    for (int node : by_slot[w]) REQUIRE(node >= 0);
  for (int producer = 0; producer < nodes; ++producer)
    for (int consumer : graph.successors[producer])
      if (schedule.worker[producer] == schedule.worker[consumer])
        REQUIRE(schedule.slot[producer] < schedule.slot[consumer]);
}

}  // namespace

int main() {
  std::string error;

  {
    // The spine is the whole graph, so it lands whole on one worker and pays
    // no hop at all: 3 x 1000 ns with a 100 ns hop is 3000, not 3200.
    auto const graph = MakeGraph({1, 1, 1}, {{0, 1}, {1, 2}});
    ChainRequest request;
    request.graph = &graph;
    request.task_ns = {1000.0, 1000.0, 1000.0};
    request.grid = 4;
    request.hop.c0 = 100.0;
    ChainSchedule schedule;
    REQUIRE(ScheduleByCriticalChain(request, &schedule, &error));
    REQUIRE(schedule.worker[0] == schedule.worker[1]);
    REQUIRE(schedule.worker[1] == schedule.worker[2]);
    REQUIRE(schedule.slot[0] == 0 && schedule.slot[1] == 1 && schedule.slot[2] == 2);
    REQUIRE(schedule.spine_length_ns == 3000.0);
    REQUIRE(schedule.makespan_ns == 3000.0);
    REQUIRE(schedule.split_count == 0);
    REQUIRE(schedule.chain_interleaves == 0);
    CheckQueues(graph, schedule, 3, 4);
  }

  {
    // A chain is ranked in nanoseconds, not in tasks: two 5000 ns tasks are a
    // longer chain than five 1000 ns ones, and the spine is the heavy pair.
    // Round one's edge-count height would have picked the other path, which is
    // the reason the weight comes from the cost model at a bound theta.
    auto const graph = MakeGraph({7}, {{0, 1}, {2, 3}, {3, 4}, {4, 5}, {5, 6}});
    ChainRequest request;
    request.graph = &graph;
    request.task_ns = {5000.0, 5000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0};
    request.grid = 4;
    request.hop.c0 = 50.0;
    ChainSchedule schedule;
    REQUIRE(ScheduleByCriticalChain(request, &schedule, &error));
    REQUIRE(schedule.spine_length_ns == 10000.0);
    REQUIRE(schedule.worker[0] == schedule.worker[1]);
    CheckQueues(graph, schedule, 7, 4);
  }

  {
    // §6.3.  Two chains, 0->1 and 2->3, with cross edges in both directions:
    // 0->3 and 2->1.  The task DAG is acyclic -- 0, 2, 1, 3 is a topological
    // order -- but contracting each chain to a node closes an A->B->A cycle,
    // so both cannot be laid down as whole blocks.  The weights make the
    // extractor pick exactly those two chains: 0->1 is 200 ns, every mixed
    // path is 190 or less.  Both land whole, and the Plan is legal: the cycle
    // §6.3 describes exists only in the contracted graph, while L-a tests the
    // union of task and queue edges, where 0->1, 0->3, 2->3, 2->1 is acyclic.
    // A cycle there would need a2->b1 and b2->a1, which is already a cycle in
    // the task DAG and so cannot be handed to a scheduler at all.  Nothing is
    // split, and `CheckPlanLegality` below is what says so.
    auto const graph = MakeGraph({4}, {{0, 1}, {2, 3}, {0, 3}, {2, 1}});
    ChainRequest request;
    request.graph = &graph;
    request.task_ns = {100.0, 100.0, 90.0, 90.0};
    request.grid = 2;
    request.hop.c0 = 10.0;
    ChainSchedule schedule;
    REQUIRE(ScheduleByCriticalChain(request, &schedule, &error));
    REQUIRE(schedule.split_count == 0);
    REQUIRE(schedule.worker[0] == schedule.worker[1]);
    REQUIRE(schedule.worker[2] == schedule.worker[3]);
    REQUIRE(schedule.worker[0] != schedule.worker[2]);
    CheckQueues(graph, schedule, 4, 2);

    // And the result is legal under the check that actually gates a
    // Plan, not merely under this test's own idea of legal (L-a, L-c).
    std::vector<int> const counts = {4};
    PlanRequest host_form;
    host_form.mode = dialect::PlacementMode::kEft;
    host_form.grid = 2;
    host_form.counts = counts;
    host_form.graph = &graph;
    host_form.stage_order = {0};
    host_form.physical_worker = {0, 1};
    host_form.eft_worker = schedule.worker;
    host_form.eft_slot = schedule.slot;
    MaterializedPlan plan;
    REQUIRE(MaterializePlanPlacement(host_form, &plan, &error));
    REQUIRE(solver::CheckPlanLegality(graph, plan, &error));
  }

  {
    // The fill step caps a queue at the spine, not at a balanced plan's
    // longest queue (F-129): the spine is 4000 ns and the eight loose 500 ns
    // tasks spread rather than pile onto the spine's worker.
    auto const graph = MakeGraph({2, 8}, {{0, 1}});
    ChainRequest request;
    request.graph = &graph;
    request.task_ns.assign(10, 500.0);
    request.task_ns[0] = 2000.0;
    request.task_ns[1] = 2000.0;
    request.grid = 4;
    request.hop.c0 = 20.0;
    ChainSchedule schedule;
    REQUIRE(ScheduleByCriticalChain(request, &schedule, &error));
    REQUIRE(schedule.spine_length_ns == 4000.0);
    REQUIRE(schedule.max_queue_ns <= 4000.0);
    CheckQueues(graph, schedule, 10, 4);
  }

  {
    // A cycle in the task DAG is reported, not walked into.
    auto const graph = MakeGraph({2}, {{0, 1}, {1, 0}});
    ChainRequest request;
    request.graph = &graph;
    request.task_ns = {10.0, 10.0};
    request.grid = 2;
    ChainSchedule schedule;
    REQUIRE(!ScheduleByCriticalChain(request, &schedule, &error));
    REQUIRE(error.find("cycle") != std::string::npos);
  }

  {
    // The stop ratio is a calibrated input, so a meaningless one is a missing
    // input rather than a silent default.
    auto const graph = MakeGraph({2}, {{0, 1}});
    ChainRequest request;
    request.graph = &graph;
    request.task_ns = {10.0, 10.0};
    request.grid = 2;
    request.chain_stop_ratio = 0.0;
    ChainSchedule schedule;
    REQUIRE(!ScheduleByCriticalChain(request, &schedule, &error));
    REQUIRE(error.find("chain_stop_ratio") != std::string::npos);
  }

  std::printf("chain_placement_test: ok\n");
  return 0;
}
