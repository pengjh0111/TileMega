// SPDX-License-Identifier: BSD-3-Clause
// EX-S1: what the execution simulator must get right for its makespan to mean
// anything.  The load-bearing assertion is the head-of-line one: the same pi
// with two different sigmas must give two different makespans, because that
// difference is the entire reason round one's max(work_lb, queue_lb,
// critical_path) underestimated mode 5 (F-139).  The rest pin the hop, the
// co-residency stretch, and the two ways the loop could lie instead of fail --
// a silently stalled worker and a silently accepted W > 1.
#include <tilemega/Solver/ExecutionSimulator.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tilemega;
using solver::HopCurve;
using solver::MaterializedPlan;
using solver::SimulatorInput;
using solver::SimulatorOptions;
using solver::SimulatorResult;

#define REQUIRE(condition)                                                 \
  do {                                                                     \
    if (!(condition)) {                                                    \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

#define REQUIRE_NEAR(a, b)                                                  \
  do {                                                                      \
    double const lhs = (a), rhs = (b);                                      \
    if (std::fabs(lhs - rhs) > 1e-6) {                                      \
      std::fprintf(stderr, "%s:%d: %s = %.6f, expected %s = %.6f\n",        \
                   __FILE__, __LINE__, #a, lhs, #b, rhs);                   \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

namespace {

codegen::RuntimeTaskGraph MakeGraph(std::vector<int> const& per_stage,
                                    std::vector<std::pair<int, int>> const& edges) {
  codegen::RuntimeTaskGraph graph;
  graph.stage_offsets.push_back(0);
  for (int count : per_stage)
    graph.stage_offsets.push_back(graph.stage_offsets.back() + count);
  graph.successors.assign(graph.stage_offsets.back(), {});
  for (auto const& edge : edges) graph.successors[edge.first].push_back(edge.second);
  graph.preferred_worker.assign(graph.stage_offsets.back(), 0);
  return graph;
}

/// `owner` and `slot` are given per node in flat order, so the test reads as
/// the plan it is describing.
MaterializedPlan MakePlan(std::vector<int> const& per_stage,
                          std::vector<int> const& owner,
                          std::vector<int> const& slot, int grid) {
  MaterializedPlan plan;
  plan.owner.resize(per_stage.size());
  plan.slot.resize(per_stage.size());
  int flat = 0;
  for (std::size_t stage = 0; stage < per_stage.size(); ++stage)
    for (int task = 0; task < per_stage[stage]; ++task, ++flat) {
      plan.owner[stage].push_back(owner[flat]);
      plan.slot[stage].push_back(slot[flat]);
    }
  std::string error;
  std::vector<int> counts(per_stage.begin(), per_stage.end());
  REQUIRE(solver::BuildPlanQueues(counts, grid, &plan, &error));
  return plan;
}

SimulatorInput MakeInput(codegen::RuntimeTaskGraph const& graph,
                         std::vector<double> const& task_ns) {
  SimulatorInput input;
  input.graph = &graph;
  input.task_ns = task_ns;
  return input;
}

}  // namespace

int main() {
  HopCurve const flat{1000.0, 0.0, 0.0};
  std::string error;

  {  // A chain on one worker pays no hop, and the whole queue serializes.
    auto graph = MakeGraph({1, 1, 1}, {{0, 1}, {1, 2}});
    auto plan = MakePlan({1, 1, 1}, {0, 0, 0}, {0, 1, 2}, 1);
    auto input = MakeInput(graph, {100.0, 200.0, 300.0});
    SimulatorResult result;
    SimulatorOptions options;
    options.sms = 1;
    options.ctas_per_sm = 1;
    REQUIRE(solver::SimulateExecution(input, plan, options, flat, &result, &error));
    REQUIRE_NEAR(result.makespan_ns, 600.0);
    REQUIRE_NEAR(result.total_block_ns, 0.0);
    REQUIRE_NEAR(result.solo_work_ns, 600.0);
    REQUIRE(result.same_worker_edges == 2 && result.cross_worker_edges == 0);
  }

  {  // The same chain split across two workers pays the hop twice, and each
     // worker's idle time shows up as block_ns.
    auto graph = MakeGraph({1, 1, 1}, {{0, 1}, {1, 2}});
    auto plan = MakePlan({1, 1, 1}, {0, 1, 0}, {0, 0, 1}, 2);
    auto input = MakeInput(graph, {100.0, 200.0, 300.0});
    SimulatorResult result;
    SimulatorOptions options;
    options.sms = 2;
    REQUIRE(solver::SimulateExecution(input, plan, options, flat, &result, &error));
    REQUIRE_NEAR(result.makespan_ns, 100.0 + 1000.0 + 200.0 + 1000.0 + 300.0);
    REQUIRE(result.cross_worker_edges == 2 && result.same_worker_edges == 0);
    // Worker 0 finished node 0 at 100 and node 2 cannot start until node 1's
    // publish lands at 1300 + 1000 = 2300, so the idle stretch is 2200 -- two
    // hops and node 1's own run, not one hop.
    REQUIRE_NEAR(result.tasks[2].block_ns, 2200.0);
  }

  {  // A producer with one successor on its own worker and one elsewhere pays
     // the hop on the second only.  Without this case a simulator that charged
     // the hop on every edge would still pass the two above, because a node with
     // no cross-worker successor is priced at zero either way.
    auto graph = MakeGraph({1, 2}, {{0, 1}, {0, 2}});
    auto plan = MakePlan({1, 2}, {0, 0, 1}, {0, 1, 0}, 2);
    auto input = MakeInput(graph, {100.0, 50.0, 50.0});
    SimulatorOptions options;
    options.sms = 2;
    SimulatorResult result;
    REQUIRE(solver::SimulateExecution(input, plan, options, flat, &result, &error));
    REQUIRE_NEAR(result.tasks[1].start_ns, 100.0);
    REQUIRE_NEAR(result.tasks[2].start_ns, 1100.0);
    REQUIRE_NEAR(result.makespan_ns, 1150.0);
    REQUIRE(result.same_worker_edges == 1 && result.cross_worker_edges == 1);
  }

  {  // Head of line.  Worker 0 owns a task that waits on worker 1, and an
     // independent task.  Only sigma decides whether the independent task runs
     // first, and the makespan differs by exactly the stall -- this is the
     // behaviour no closed-form bound can express.
     auto graph = MakeGraph({1, 2}, {{0, 1}});
     auto input = MakeInput(graph, {500.0, 100.0, 100.0});
     SimulatorOptions options;
     options.sms = 2;
     // sigma puts the blocked consumer first: worker 0 idles through the hop.
     auto blocked = MakePlan({1, 2}, {1, 0, 0}, {0, 0, 1}, 2);
     SimulatorResult slow;
     REQUIRE(solver::SimulateExecution(input, blocked, options, flat, &slow, &error));
     REQUIRE_NEAR(slow.makespan_ns, 500.0 + 1000.0 + 100.0 + 100.0);
     // sigma puts the independent task first: the same pi, 100 ns shorter.
     auto filled = MakePlan({1, 2}, {1, 0, 0}, {0, 1, 0}, 2);
     SimulatorResult fast;
     REQUIRE(solver::SimulateExecution(input, filled, options, flat, &fast, &error));
     REQUIRE_NEAR(fast.makespan_ns, 500.0 + 1000.0 + 100.0);
     REQUIRE(fast.makespan_ns < slow.makespan_ns);
     // Both place every task on the same worker as the other plan does.
     REQUIRE(fast.busiest_worker_ns == slow.busiest_worker_ns);
  }

  {  // W = 1 across a foreign wake-up.  Worker 0 is 3000 ns into a task when
     // the hop for its *next* queued task lands at 1500.  A loop that decides
     // readiness from "the previous task has ended" rather than "this worker is
     // idle" starts the second task alongside the first, and then the busiest
     // worker's own work (3100) exceeds the makespan (3000) -- an impossible
     // schedule that still looks like a plausible number.
    auto graph = MakeGraph({1, 2}, {{0, 2}});
    auto plan = MakePlan({1, 2}, {1, 0, 0}, {0, 0, 1}, 2);
    auto input = MakeInput(graph, {500.0, 3000.0, 100.0});
    SimulatorOptions options;
    options.sms = 2;
    SimulatorResult result;
    REQUIRE(solver::SimulateExecution(input, plan, options, flat, &result, &error));
    // Node 2 is ready at 500 + 1000, but its worker is not free until 3000.
    REQUIRE_NEAR(result.tasks[2].start_ns, 3000.0);
    REQUIRE_NEAR(result.tasks[2].block_ns, 0.0);
    REQUIRE_NEAR(result.makespan_ns, 3100.0);
    REQUIRE_NEAR(result.busiest_worker_ns, 3100.0);
    REQUIRE(result.busiest_worker_ns <= result.makespan_ns);
  }

  {  // Co-residency: two workers on one SM, identical single-lane demand, so
     // the set's aggregate demand on that lane is twice either member's and
     // both tasks take twice as long.
    auto graph = MakeGraph({2}, {});
    auto plan = MakePlan({2}, {0, 1}, {0, 0}, 2);
    auto input = MakeInput(graph, {100.0, 100.0});
    solver::ResourceVector lane;
    lane.cuda_core = 100.0;
    input.task_lanes = {lane, lane};
    SimulatorOptions options;
    options.sms = 1;
    options.ctas_per_sm = 2;
    SimulatorResult shared;
    REQUIRE(solver::SimulateExecution(input, plan, options, flat, &shared, &error));
    REQUIRE_NEAR(shared.makespan_ns, 200.0);
    REQUIRE_NEAR(shared.tasks[0].stretch, 2.0);
    REQUIRE_NEAR(shared.solo_work_ns, 200.0);
    REQUIRE_NEAR(shared.total_work_ns, 400.0);
    // Disjoint lanes contend for nothing, so the same pair runs concurrently.
    solver::ResourceVector other;
    other.dram = 100.0;
    input.task_lanes = {lane, other};
    SimulatorResult disjoint;
    REQUIRE(solver::SimulateExecution(input, plan, options, flat, &disjoint, &error));
    REQUIRE_NEAR(disjoint.makespan_ns, 100.0);
    // Two SMs never share anything, whatever the lanes say.
    options.sms = 2;
    options.ctas_per_sm = 1;
    input.task_lanes = {lane, lane};
    SimulatorResult apart;
    REQUIRE(solver::SimulateExecution(input, plan, options, flat, &apart, &error));
    REQUIRE_NEAR(apart.makespan_ns, 100.0);
    // The control arm ignores the lanes and charges the set's size.
    options.sms = 1;
    options.ctas_per_sm = 2;
    options.proportional_sharing = true;
    input.task_lanes = {lane, other};
    SimulatorResult proportional;
    REQUIRE(solver::SimulateExecution(input, plan, options, flat, &proportional,
                                      &error));
    REQUIRE_NEAR(proportional.makespan_ns, 200.0);
  }

  {  // A queue order that closes a cycle with the task edges must be reported
     // as an L-a violation, not run to a step budget.
    auto graph = MakeGraph({2, 2}, {{1, 2}, {3, 0}});
    auto plan = MakePlan({2, 2}, {0, 0, 1, 1}, {0, 1, 0, 1}, 2);
    auto input = MakeInput(graph, {10.0, 10.0, 10.0, 10.0});
    SimulatorOptions options;
    options.sms = 2;
    SimulatorResult result;
    REQUIRE(!solver::SimulateExecution(input, plan, options, flat, &result, &error));
    REQUIRE(error.find("deadlock") != std::string::npos);
    REQUIRE(error.find("L-a") != std::string::npos);
  }

  {  // W > 1 is out of scope this round and must be refused, not run as FIFO.
    auto graph = MakeGraph({1}, {});
    auto plan = MakePlan({1}, {0}, {0}, 1);
    auto input = MakeInput(graph, {10.0});
    SimulatorOptions options;
    options.sms = 1;
    options.window = 2;
    SimulatorResult result;
    REQUIRE(!solver::SimulateExecution(input, plan, options, flat, &result, &error));
    REQUIRE(error.find("EX-E2") != std::string::npos);
  }

  {  // A node no worker owns is a malformed plan, not a zero-cost task.
    auto graph = MakeGraph({2}, {});
    MaterializedPlan plan;
    plan.owner = {{0}};
    plan.slot = {{0}};
    plan.queue.assign(1, {});
    plan.queue[0].push_back({0u, 0});
    auto input = MakeInput(graph, {10.0, 10.0});
    SimulatorOptions options;
    options.sms = 1;
    SimulatorResult result;
    REQUIRE(!solver::SimulateExecution(input, plan, options, flat, &result, &error));
    REQUIRE(error.find("in no worker's queue") != std::string::npos);
  }

  {  // The curve itself: c1 and c2 are in ns per doubling, and N = R = 1 is c0.
    HopCurve curve{1200.0, 8.0, -2.0};
    REQUIRE_NEAR(curve.Ns(1, 1), 1200.0 + 8.0);
    REQUIRE_NEAR(curve.Ns(3, 1), 1200.0 + 16.0);
    REQUIRE_NEAR(curve.Ns(1, 2), 1200.0 + 8.0 * std::log2(1.5) - 2.0);
    // flat_hop must price c0 exactly, so the sensitivity arm is a real control.
    auto graph = MakeGraph({1, 1}, {{0, 1}});
    auto plan = MakePlan({1, 1}, {0, 1}, {0, 0}, 2);
    auto input = MakeInput(graph, {10.0, 10.0});
    SimulatorOptions options;
    options.sms = 2;
    options.flat_hop = true;
    SimulatorResult result;
    REQUIRE(solver::SimulateExecution(input, plan, options, curve, &result, &error));
    REQUIRE_NEAR(result.makespan_ns, 10.0 + 1200.0 + 10.0);
  }

  std::printf("execution_simulator_test: PASS\n");
  return 0;
}
