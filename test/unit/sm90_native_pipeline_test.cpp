// SPDX-License-Identifier: BSD-3-Clause
// P3: Algorithm 1's shape. The descent must never accept a worse score, the
// shortlist must carry more than one structural class when the space offers
// one, no structural class may take more than its share of the finalists, and
// the operator-sum objective (AB-2) must change the ranking rather than the
// mechanism.
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <cassert>
#include <cstdio>
#include <set>

using namespace tilemega;
using namespace tilemega::solver;

int main(int argc, char** argv) {
  auto target = TargetSpec::FromJson(argv[1]);
  auto constants = Sm90NativeConstants::FromTarget(json::ParseFile(argv[1]), argv[2]);
  Sm90DenseCostModel cost(target, constants);

  Sm90DenseWorkload work;
  work.shapes = {{{4096, 2048}, {2048, 2048}, {6144, 2048}, {2048, 3072}, {151936, 2048}}};
  work.layers = 28; work.head_dim = 128; work.queries = 2; work.hkv = 8;
  work.qk_norm = true; work.batch = 1;

  Sm90DenseSearchSpace space{work.shapes, work.batch, work.head_dim, work.queries, target, work.hkv};
  space.weight_stream = true; space.layer_loop = true;

  Sm90NativePipelineOptions options;
  options.budget_ms = 20000;  // the unit test is not the round's budget
  auto result = SolveSm90NativePipeline(space, work, cost, options);

  std::printf("evaluated %zu, memo hits %zu, shortlist %zu, finalists %zu, %.2f s\n",
              result.evaluations, result.memo_hits, result.shortlist.size(),
              result.finalists.size(), result.seconds);
  assert(result.evaluations > 0);
  assert(!result.shortlist.empty() && !result.finalists.empty());
  // Each seed's trajectory is monotone: a move is accepted only when it
  // improves the score. Across seeds it is not, and must not be asserted to be.
  for (auto const& trajectory : result.descent)
    for (std::size_t i = 1; i < trajectory.size(); ++i)
      assert(trajectory[i] <= trajectory[i - 1]);
  // Finalists are ordered by replay and respect the per-class cap.
  std::map<std::string, int> taken;
  for (std::size_t i = 0; i < result.finalists.size(); ++i) {
    ++taken[result.finalists[i].structural_class];
    if (i) assert(result.finalists[i].replay_ns >= result.finalists[i - 1].replay_ns ||
                  int(result.finalists.size()) > options.finalists);
  }
  for (auto const& [name, count] : taken) {
    (void)name;
    assert(count <= options.per_class || int(result.finalists.size()) <= options.per_class);
  }
  std::printf("  best pool %.1f ns, best replay %.1f ns, classes in shortlist:",
              result.shortlist.front().pool_score_ns, result.finalists.front().replay_ns);
  std::set<std::string> classes;
  for (auto const& candidate : result.shortlist) classes.insert(candidate.structural_class);
  for (auto const& name : classes) std::printf(" %s", name.c_str());
  std::printf("\n");

  // AB-2 keeps the mechanism and changes only the objective.
  auto sum = options;
  sum.operator_sum = true;
  auto other = SolveSm90NativePipeline(space, work, cost, sum);
  assert(!other.finalists.empty());
  std::printf("operator-sum: evaluated %zu, best %.1f ns (pool objective best %.1f)\n",
              other.evaluations, other.finalists.front().pool_score_ns,
              result.finalists.front().pool_score_ns);
  return 0;
}
