// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
// P4.8: the schedule is only worth anything if `levels` is really the DAG's
// depth, so every assertion here pins one way of getting that wrong.
#include <tilemega/Solver/ListScheduler.h>
#include <tilemega/Solver/BalancedPlacement.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
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

int main() {
  tilemega::analysis::IslContext isl_context;
  ListScheduler scheduler;
  {
    std::vector<std::vector<int>> dag={{2,2},{3},{},{}};
    auto mapped=BalanceTaskPlacement(dag,{0,1,2,3},{0,1,1,0},2,2);
    REQUIRE(mapped.max_queue==2);
    REQUIRE(mapped.same_worker_edges==2 && mapped.fence_free_producers==2);
    REQUIRE(mapped.worker[0]==mapped.worker[2] && mapped.worker[1]==mapped.worker[3]);
    int errors=0;
    auto reject=[&](auto action) {
      bool failed=false;
      try { action(); } catch (std::invalid_argument const&) { failed=true; }
      REQUIRE(failed); ++errors;
    };
    reject([&]{BalanceTaskPlacement(dag,{0,1,2,3},{0,1,1,0},2,1);});
    reject([&]{BalanceTaskPlacement(dag,{2,1,0,3},{0,1,1,0},2,2);});
    reject([&]{BalanceTaskPlacement(dag,{0,1,2,3},{0,2,1,0},2,2);});
    REQUIRE(errors==3);
  }

  // A chain: nothing to pack, one level per node.
  {
    std::vector<std::vector<int>> chain = {{1}, {2}, {3}, {}};
    ScheduleStats stats;
    std::vector<int> order = scheduler.Schedule(chain, &stats);
    REQUIRE(stats.levels == 4);
    REQUIRE(stats.widest_level == 1);
    REQUIRE(stats.barriers_saved == 0);
    REQUIRE((order == std::vector<int>{0, 1, 2, 3}));
  }

  // A diamond with an extra-long left arm.  The level of the join is the
  // *longest* path to it, not the shortest -- taking the shortest would report
  // a depth the barrier schedule could not actually run.
  {
    std::vector<std::vector<int>> diamond = {{1, 3}, {2}, {4}, {4}, {}};
    ScheduleStats stats;
    std::vector<int> order = scheduler.Schedule(diamond, &stats);
    ScheduleSafety safety = scheduler.Validate(diamond, order);
    REQUIRE(stats.levels == 4);        // 0 -> 1 -> 2 -> 4
    REQUIRE(stats.widest_level == 2);  // {1,3} share a level, {2} does not
    REQUIRE(stats.barriers_saved == 1);
    // Level 1 holds 1 and 3; height breaks the tie and 1 has the longer tail.
    REQUIRE(order[1] == 1 && order[2] == 3);
    REQUIRE(safety.max_dependency_span == 2);

    // Small-instance oracle: enumerate every permutation, retain only complete
    // topological schedules, and prove the critical-path priority reaches the
    // minimum possible maximum producer-to-consumer span.
    std::vector<int> candidate(diamond.size());
    std::iota(candidate.begin(), candidate.end(), 0);
    int oracle_span = std::numeric_limits<int>::max();
    int feasible = 0;
    do {
      try {
        ScheduleSafety const trial = scheduler.Validate(diamond, candidate);
        oracle_span = std::min(oracle_span, trial.max_dependency_span);
        ++feasible;
      } catch (std::invalid_argument const&) {
      }
    } while (std::next_permutation(candidate.begin(), candidate.end()));
    REQUIRE(feasible == 3);
    REQUIRE(safety.max_dependency_span == oracle_span);
    std::printf("PLACE_ORACLE nodes=5 feasible=%d optimum_span=%d cp_span=%d\n",
                feasible, oracle_span, safety.max_dependency_span);
  }

  // A dependency reversed by a purported placement is a generation error,
  // even when the dependency graph itself is acyclic.
  {
    bool threw = false;
    try {
      scheduler.Validate({{1}, {}}, {1, 0});
    } catch (std::invalid_argument const&) {
      threw = true;
    }
    REQUIRE(threw);
  }

  // Duplicate rows imply a missing worker slot and are rejected as one error.
  {
    bool threw = false;
    try {
      scheduler.Validate({{1}, {}}, {0, 0});
    } catch (std::invalid_argument const&) {
      threw = true;
    }
    REQUIRE(threw);
  }

  // Height is the longest path to a sink, so a node feeding both arms of the
  // diamond outranks one feeding only the short arm.
  {
    std::vector<std::vector<int>> diamond = {{1, 3}, {2}, {4}, {4}, {}};
    std::vector<int> height = scheduler.Heights(diamond);
    REQUIRE((height == std::vector<int>{3, 2, 1, 1, 0}));
  }

  // A cycle is not a schedule.  Returning a partial order here would look
  // exactly like a valid answer to every caller.
  {
    std::vector<std::vector<int>> cycle = {{1}, {2}, {0}};
    bool threw = false;
    try {
      scheduler.Levels(cycle);
    } catch (std::invalid_argument const&) {
      threw = true;
    }
    REQUIRE(threw);
  }

  // An edge pointing outside the graph is a malformed table, not an isolated
  // node to be scheduled around.
  {
    std::vector<std::vector<int>> broken = {{5}, {}};
    bool threw = false;
    try {
      scheduler.Levels(broken);
    } catch (std::invalid_argument const&) {
      threw = true;
    }
    REQUIRE(threw);
  }

  std::printf("list_scheduler_test PASS\n");
  return 0;
}
