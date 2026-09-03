// SPDX-License-Identifier: BSD-3-Clause
// P4.8: the schedule is only worth anything if `levels` is really the DAG's
// depth, so every assertion here pins one way of getting that wrong.
#include <tilemega/Solver/ListScheduler.h>

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

int main() {
  ListScheduler scheduler;

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
    REQUIRE(stats.levels == 4);        // 0 -> 1 -> 2 -> 4
    REQUIRE(stats.widest_level == 2);  // {1,3} share a level, {2} does not
    REQUIRE(stats.barriers_saved == 1);
    // Level 1 holds 1 and 3; height breaks the tie and 1 has the longer tail.
    REQUIRE(order[1] == 1 && order[2] == 3);
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
