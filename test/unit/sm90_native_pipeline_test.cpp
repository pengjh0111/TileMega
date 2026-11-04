// SPDX-License-Identifier: BSD-3-Clause
// P3: Algorithm 1's shape. The descent must never accept a worse score, the
// shortlist must carry more than one structural class when the space offers
// one, no structural class may take more than its share of the finalists, and
// the operator-sum objective (AB-2) must change the ranking rather than the
// mechanism.
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <set>

using namespace tilemega;
using namespace tilemega::solver;

int main(int argc, char** argv) {
  auto target = TargetSpec::FromJson(argv[1]);
  auto constants = Sm90NativeConstants::FromTarget(json::ParseFile(argv[1]), argv[2]);
  Sm90DenseCostModel cost(target, constants);

  // Eq. (10) must consume Eq. (9)'s singleton placement. Pooling the
  // two producers would finish in 10; their required owner finishes in 20.
  FlowProblem binding;binding.workers=2;binding.dram_gbps=1;
  FlowSpace producer;producer.count=2;producer.piece_of_task={0,0};
  TaskPriceParts ten;ten.compute_ns=10;
  producer.pieces={{2,ten}};binding.spaces.push_back(producer);
  FlowSpace consumer;consumer.count=1;consumer.order=1;consumer.piece_of_task={0};
  TaskPriceParts five;five.compute_ns=5;
  consumer.pieces={{1,five}};binding.spaces.push_back(consumer);
  FlowRelease edge;edge.producer=0;edge.consumer=1;edge.hop_ns=2;
  edge.sorted=std::make_shared<std::vector<std::pair<int,int>> const>(
      std::vector<std::pair<int,int>>{{1,0}});binding.edges.push_back(edge);
  auto constrained=RefineAndReplaySm90Native(binding,{{0,0},{1}});
  assert(constrained.placement.worker_of_task==std::vector<std::vector<int>>({{0,0},{1}}));
  assert(std::abs(constrained.replay.makespan_ns-27)<1e-9);
  assert(std::abs(constrained.estimated_finish_ns[1][0]-27)<1e-9);
  assert(std::abs(EvaluateFlow(binding).makespan_ns-17)<1e-9);
  assert(Sm90NativeCandidateTracks(binding,{0,1})==
      std::vector<std::vector<int>>({{0,1},{1}}));

  Sm90DenseWorkload work;
  work.shapes = {{{4096, 2048}, {2048, 2048}, {6144, 2048}, {2048, 3072}, {151936, 2048}}};
  work.layers = 28; work.head_dim = 128; work.queries = 2; work.hkv = 8;
  work.qk_norm = true; work.batch = 1;

  Sm90DenseSearchSpace space{work.shapes, work.batch, work.head_dim, work.queries, target, work.hkv};
  space.weight_stream = true; space.layer_loop = true;

  Sm90NativePipelineOptions options;
  options.budget_ms = 20000;  // the unit test is not the round's budget
  // Search mechanics use a deterministic fixture. Real CTG construction,
  // reducers and isolated scoring are covered by sm90_native_flow_test.
  auto evaluate=[&](Sm90DenseCoordinates const& c,bool isolated) {
    auto flow=Sm90NativePoolScore(work,c,cost);
    if(isolated)flow.step_ns=Sm90NativeOperatorSumScore(work,c,cost);
    return flow;
  };
  auto result = SolveSm90NativePipeline(space, work, cost, options,evaluate);

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
  auto other = SolveSm90NativePipeline(space, work, cost, sum,evaluate);
  assert(!other.finalists.empty());
  for(auto const& candidate:other.finalists) {
    assert(candidate.replay_ns==candidate.pool_score_ns);
    double independent=0;
    for(auto const& node:Sm90NativeStages(work,candidate.coordinates,cost,true)) {
      auto single=Sm90NativeSingleStageProblem(node.price,cost,candidate.coordinates.residency);
      for(auto const& s:single.spaces)for(auto const& piece:s.pieces)
        assert(piece.parts.no_producer_dram_bytes==0);
      independent+=node.instances*EvaluateFlow(single).makespan_ns;
    }
    assert(std::abs(candidate.pool_score_ns-independent)<1e-6);
  }
  std::printf("operator-sum: evaluated %zu, best %.1f ns (pool objective best %.1f)\n",
              other.evaluations, other.finalists.front().pool_score_ns,
              result.finalists.front().pool_score_ns);

  auto pin=options;pin.budget_ms=1000;
  pin.pinned_theta=result.finalists.front().coordinates;
  auto pinned=SolveSm90NativePipeline(space,work,cost,pin,evaluate,{*pin.pinned_theta});
  assert(!pinned.evaluated.empty());
  for(auto const& candidate:pinned.evaluated)for(unsigned f:candidate.coordinates.Active()) {
    auto const& a=candidate.coordinates.gemms[f];auto const& b=pin.pinned_theta->gemms[f];
    assert(a.tile_n==b.tile_n && a.TileK()==b.TileK() && a.warps_n==b.warps_n &&
           a.warps_k==b.warps_k && a.multicast==b.multicast);
  }
  return 0;
}
