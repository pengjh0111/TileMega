// SPDX-License-Identifier: BSD-3-Clause
// EX-S2: earliest-finish-time placement, and the host's replay of it.  What is
// pinned here: the greedy actually minimises finish time rather than queue
// length; sigma is start-time order, so L-c holds without a sort (§5.7.3); the
// queues the host builds from the shipped (worker, slot) table are the
// scheduler's own queues, which is what §5.7.4 means by the solver deciding and
// the host materializing; and the two closed-form templates are legal and are
// not each other.
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Solver/EftPlacement.h>
#include <tilemega/Solver/PlanMaterialize.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tilemega;
using solver::EftRequest;
using solver::EftSchedule;
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

PlanRequest MakeRequest(codegen::RuntimeTaskGraph const& graph,
                        std::vector<int> const& counts, int grid) {
  PlanRequest request;
  request.mode = dialect::PlacementMode::kEft;
  request.grid = grid;
  request.counts = counts;
  request.graph = &graph;
  for (std::size_t stage = 0; stage < counts.size(); ++stage)
    request.stage_order.push_back(static_cast<std::uint32_t>(stage));
  for (int w = 0; w < grid; ++w) request.physical_worker.push_back(w);
  return request;
}

}  // namespace

int main() {
  std::string error;

  {
    // A chain gets no parallelism, so every task lands where its predecessor
    // did: a hop costs 100 ns and moving cannot buy anything back.
    auto const graph = MakeGraph({1, 1, 1}, {{0, 1}, {1, 2}});
    EftRequest request;
    request.graph = &graph;
    request.task_ns = {1000.0, 1000.0, 1000.0};
    request.grid = 4;
    request.sms = 4;
    request.ctas_per_sm = 1;
    request.hop.c0 = 100.0;
    EftSchedule schedule;
    REQUIRE(ScheduleByEarliestFinish(request, &schedule, &error));
    REQUIRE(schedule.worker[0] == schedule.worker[1]);
    REQUIRE(schedule.worker[1] == schedule.worker[2]);
    REQUIRE(schedule.slot[0] == 0 && schedule.slot[1] == 1 && schedule.slot[2] == 2);
    REQUIRE(schedule.makespan_ns == 3000.0);
  }

  {
    // The same chain with a hop cheaper than the wait it avoids: four
    // independent tasks feeding one consumer spread out, because a second task
    // on worker 0 would wait 1000 ns and a hop costs 10.
    auto const graph = MakeGraph({4, 1}, {{0, 4}, {1, 4}, {2, 4}, {3, 4}});
    EftRequest request;
    request.graph = &graph;
    request.task_ns = {1000.0, 1000.0, 1000.0, 1000.0, 500.0};
    request.grid = 4;
    request.sms = 4;
    request.ctas_per_sm = 1;
    request.hop.c0 = 10.0;
    EftSchedule schedule;
    REQUIRE(ScheduleByEarliestFinish(request, &schedule, &error));
    std::vector<int> seen(4, 0);
    for (int node = 0; node < 4; ++node) ++seen[schedule.worker[node]];
    for (int w = 0; w < 4; ++w) REQUIRE(seen[w] == 1);
    // The consumer waits for the last producer plus one hop wherever it goes,
    // so it stays where it is free earliest and the makespan is 1510, not 2010.
    REQUIRE(schedule.makespan_ns == 1510.0);
  }

  {
    // Co-residency is charged, and it changes the choice: one SM with two
    // resident CTAs, a 1000 ns task already running on the sibling, and a
    // 400 ns task that can either wait for the sibling's worker to free up
    // (finishing at 1400) or run alongside it at half rate (finishing at 800).
    auto const graph = MakeGraph({2}, {});
    EftRequest request;
    request.graph = &graph;
    request.task_ns = {1000.0, 400.0};
    request.grid = 2;
    request.sms = 1;
    request.ctas_per_sm = 2;
    EftSchedule schedule;
    REQUIRE(ScheduleByEarliestFinish(request, &schedule, &error));
    REQUIRE(schedule.worker[0] != schedule.worker[1]);
    REQUIRE(schedule.start_ns[1] == 0.0);
    REQUIRE(schedule.end_ns[1] == 800.0);
  }

  {
    // sigma is start-time order per worker, which is what makes L-c hold
    // without sorting: check it directly rather than trusting the comment.
    auto const graph = MakeGraph({6, 6}, {{0, 6}, {1, 7}, {2, 8}, {3, 9},
                                          {4, 10}, {5, 11}, {0, 11}});
    EftRequest request;
    request.graph = &graph;
    request.task_ns.assign(12, 700.0);
    request.task_ns[3] = 4000.0;  // one long task, so the greedy must reorder
    request.grid = 3;
    request.sms = 3;
    request.ctas_per_sm = 1;
    request.hop.c0 = 50.0;
    EftSchedule schedule;
    REQUIRE(ScheduleByEarliestFinish(request, &schedule, &error));
    std::vector<double> last_start(3, -1.0);
    std::vector<int> last_slot(3, -1);
    std::vector<std::vector<int>> by_slot(3);
    for (int node = 0; node < 12; ++node) {
      int const w = schedule.worker[node];
      if (static_cast<int>(by_slot[w].size()) <= schedule.slot[node])
        by_slot[w].resize(schedule.slot[node] + 1, -1);
      REQUIRE(by_slot[w][schedule.slot[node]] == -1);  // sigma is dense
      by_slot[w][schedule.slot[node]] = node;
    }
    for (int w = 0; w < 3; ++w)
      for (int node : by_slot[w]) {
        REQUIRE(node >= 0);
        REQUIRE(schedule.start_ns[node] >= last_start[w]);
        REQUIRE(schedule.slot[node] > last_slot[w]);
        last_start[w] = schedule.start_ns[node];
        last_slot[w] = schedule.slot[node];
      }
    for (int producer = 0; producer < 12; ++producer)
      for (int consumer : graph.successors[producer])
        if (schedule.worker[producer] == schedule.worker[consumer])
          REQUIRE(schedule.slot[producer] < schedule.slot[consumer]);
  }

  {
    // The two spellings of one Plan: the solver computes (pi, sigma) offline,
    // the host replays the table it shipped, and the queues must be identical.
    std::vector<int> const counts = {8, 8, 4};
    auto const graph = MakeGraph(counts, {{0, 8}, {1, 9}, {2, 10}, {3, 11},
                                          {4, 12}, {5, 13}, {6, 14}, {7, 15},
                                          {8, 16}, {10, 16}, {12, 17}, {14, 17},
                                          {9, 18}, {11, 18}, {13, 19}, {15, 19}});
    EftRequest eft;
    eft.graph = &graph;
    eft.task_ns.assign(20, 900.0);
    for (int node = 8; node < 16; ++node) eft.task_ns[node] = 1500.0;
    eft.grid = 4;
    eft.sms = 2;
    eft.ctas_per_sm = 2;
    eft.hop.c0 = 120.0;
    eft.hop.c1 = 8.0;

    EftSchedule schedule;
    REQUIRE(ScheduleByEarliestFinish(eft, &schedule, &error));
    PlanRequest host_form = MakeRequest(graph, counts, 4);
    host_form.eft_worker = schedule.worker;
    host_form.eft_slot = schedule.slot;
    MaterializedPlan from_table;
    REQUIRE(MaterializePlanPlacement(host_form, &from_table, &error));
    REQUIRE(solver::CheckPlanLegality(graph, from_table, &error));
    // The host replays rather than re-decides: every queue is the scheduler's
    // own answer read back, node for node, in sigma order.
    for (int node = 0; node < 20; ++node) {
      int const stage = node < 8 ? 0 : (node < 16 ? 1 : 2);
      int const task = node - graph.stage_offsets[stage];
      REQUIRE(from_table.owner[stage][task] == schedule.worker[node]);
      REQUIRE(from_table.slot[stage][task] == schedule.slot[node]);
      auto const& item = from_table.queue[schedule.worker[node]][schedule.slot[node]];
      REQUIRE(static_cast<int>(item.stage) == stage);
      REQUIRE(item.logical == task);
    }

    // The mode has no closed form, so a request with no table is a missing
    // input rather than a default (H5).
    MaterializedPlan scratch;
    PlanRequest none = MakeRequest(graph, counts, 4);
    REQUIRE(!MaterializePlanPlacement(none, &scratch, &error));
    REQUIRE(error.find("materialized (worker, slot) table") != std::string::npos);

    // A table from a different theta must hard-fail, not be padded (H5).
    PlanRequest stale = host_form;
    stale.eft_slot.pop_back();
    REQUIRE(!MaterializePlanPlacement(stale, &scratch, &error));
    REQUIRE(error.find("bound theta") != std::string::npos);

    // The closed forms: both legal, and band is not wavefront.  Band blocks
    // within a stage, so with 8 tasks over 4 workers tasks 0 and 1 share a
    // worker; wavefront numbers across a level, so they do not.
    PlanRequest band = MakeRequest(graph, counts, 4);
    band.mode = dialect::PlacementMode::kTemplate;
    band.params = {static_cast<std::int64_t>(dialect::PlacementTemplate::kBand)};
    MaterializedPlan band_plan;
    REQUIRE(MaterializePlanPlacement(band, &band_plan, &error));
    REQUIRE(solver::CheckPlanLegality(graph, band_plan, &error));
    REQUIRE(band_plan.owner[0][0] == band_plan.owner[0][1]);
    REQUIRE(band_plan.owner[0][0] != band_plan.owner[0][2]);

    PlanRequest wavefront = MakeRequest(graph, counts, 4);
    wavefront.mode = dialect::PlacementMode::kTemplate;
    wavefront.params = {
        static_cast<std::int64_t>(dialect::PlacementTemplate::kWavefront)};
    MaterializedPlan wavefront_plan;
    REQUIRE(MaterializePlanPlacement(wavefront, &wavefront_plan, &error));
    REQUIRE(solver::CheckPlanLegality(graph, wavefront_plan, &error));
    REQUIRE(wavefront_plan.owner[0][0] != wavefront_plan.owner[0][1]);
    REQUIRE(wavefront_plan.owner != band_plan.owner);

    PlanRequest bad = MakeRequest(graph, counts, 4);
    bad.mode = dialect::PlacementMode::kTemplate;
    bad.params = {7};
    REQUIRE(!MaterializePlanPlacement(bad, &scratch, &error));
    REQUIRE(error.find("kBand or kWavefront") != std::string::npos);
  }

  {
    // Wavefront is not stage-major, which only shows on a graph whose levels
    // cut across the stages: stage 2 here has a task with no predecessor, so it
    // sits in level 0 alongside stage 0 and lands on a worker before that
    // worker's stage-1 task.  The stage-major modes cannot express this order,
    // which is the reason sigma is a first-class part of the contract (EX-E1).
    std::vector<int> const counts = {2, 2, 2};
    auto const graph = MakeGraph(counts, {{0, 2}, {1, 3}});
    PlanRequest request = MakeRequest(graph, counts, 2);
    request.mode = dialect::PlacementMode::kTemplate;
    request.params = {
        static_cast<std::int64_t>(dialect::PlacementTemplate::kWavefront)};
    MaterializedPlan plan;
    REQUIRE(MaterializePlanPlacement(request, &plan, &error));
    REQUIRE(solver::CheckPlanLegality(graph, plan, &error));
    bool interleaved = false;
    for (int w = 0; w < 2; ++w)
      for (std::size_t i = 1; i < plan.queue[w].size(); ++i)
        if (plan.queue[w][i].stage < plan.queue[w][i - 1].stage) interleaved = true;
    REQUIRE(interleaved);
  }

  {
    // A cycle is reported, not walked into.
    auto const graph = MakeGraph({2}, {{0, 1}, {1, 0}});
    EftRequest request;
    request.graph = &graph;
    request.task_ns = {10.0, 10.0};
    request.grid = 2;
    request.sms = 2;
    EftSchedule schedule;
    REQUIRE(!ScheduleByEarliestFinish(request, &schedule, &error));
    REQUIRE(error.find("cycle") != std::string::npos);
  }

  std::printf("eft_placement_test: ok\n");
  return 0;
}
