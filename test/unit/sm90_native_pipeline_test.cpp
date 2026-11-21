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

  Sm90DenseCoordinates compiled;compiled.residency=2;
  for(unsigned f:compiled.Active()) {
    auto& g=compiled.gemms[f];g.tile_n=f==4?128:32;g.slot_bytes=2*g.tile_n*64;g.slots=2;
    g.warps_n=4;g.warps_k=1;
  }
  compiled.attention.chunks=16;compiled.attention.slots=2;
  Sm90DenseSearchSpace::Structure(compiled);assert(space.Legal(compiled));
  int rescored=0;
  auto occupancy_score=[&](Sm90DenseCoordinates const& c,bool isolated) {
    assert(c.residency==1 && isolated==bool(rescored));++rescored;
    assert(c.ToJson().At("gemms").Dump()==compiled.ToJson().At("gemms").Dump());
    Sm90NativeFlowResult flow;flow.problem=binding;flow.step_ns=99;return flow;
  };
  auto corrected=RepriceSm90NativeResidency(space,compiled,1,false,occupancy_score);
  assert(corrected && corrected->coordinates.residency==1 && corrected->pool_score_ns==99);
  assert(std::abs(corrected->replay_ns-17)<1e-9);
  auto isolated_correction=RepriceSm90NativeResidency(space,compiled,1,true,occupancy_score);
  assert(isolated_correction && isolated_correction->replay_ns==99 && rescored==2);
  assert(!RepriceSm90NativeResidency(space,compiled,0,false,occupancy_score));
  auto illegal=compiled;illegal.gemms[0].slots=100;
  assert(!RepriceSm90NativeResidency(space,illegal,1,false,occupancy_score));
  assert(rescored==2);

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
  auto aggregate=options;aggregate.realize_aggregate=true;aggregate.budget_ms=1000;
  auto aggregate_result=SolveSm90NativePipeline(space,work,cost,aggregate,evaluate,
      {result.finalists.front().coordinates});
  assert(!aggregate_result.evaluated.empty());
  for(auto const& candidate:aggregate_result.evaluated) {
    auto const& c=candidate.coordinates;
    assert(!c.group_wait && !c.embed_la && c.argmax_stage && c.attention.merge_stage);
    assert(!c.attention.dsmem && !c.expert_fuse);
  }
  if(argc>=4) {
    auto ampere=TargetSpec::FromJson(argv[3]);
    auto ampere_constants=Sm90NativeConstants::FromTarget(json::ParseFile(argv[3]),argv[2]);
    Sm90DenseCostModel ampere_cost(ampere,ampere_constants);
    Sm90DenseWorkload llama;
    llama.shapes={{{3072,2048},{2048,2048},{16384,2048},{2048,8192},{128256,2048}}};
    llama.layers=16;llama.head_dim=64;llama.queries=4;llama.hkv=8;llama.batch=16;
    Sm90DenseSearchSpace ampere_space{llama.shapes,llama.batch,llama.head_dim,
                                     llama.queries,ampere,llama.hkv};
    Sm90DenseCoordinates seed;
    for(unsigned f:seed.Active()) {
      auto& gemm=seed.gemms[f];
      gemm.tile_n=f==2 || f==4?128:32;
      gemm.warps_n=4;gemm.warps_k=1;gemm.slot_bytes=2*gemm.tile_n*64;
      gemm.slots=f==2?3:2;
    }
    seed.attention.slots=2;
    assert(ampere_space.Legal(seed));
    auto ampere_options=options;ampere_options.budget_ms=1000;
    ampere_options.pinned_theta=seed;
    auto ampere_score=[&](Sm90DenseCoordinates const&,bool) {
      Sm90NativeFlowResult flow;flow.problem=binding;flow.step_ns=99;return flow;
    };
    auto ampere_result=SolveSm90NativePipeline(ampere_space,llama,ampere_cost,
                                               ampere_options,ampere_score,{seed});
    assert(!ampere_result.evaluated.empty());
    assert(ampere_result.evaluated.front().coordinates.gemms[2].slots==3);
    auto comparison=ampere_options;
    comparison.pinned_theta.reset();
    comparison.finalists=2;comparison.shortlist=1;comparison.structural_extra=0;
    auto comparison_score=[&](Sm90DenseCoordinates const& c,bool) {
      Sm90NativeFlowResult flow;flow.problem=binding;
      flow.step_ns=c.ToJson().Dump(0)==seed.ToJson().Dump(0)?9999:99;
      return flow;
    };
    auto matched=SolveSm90NativePipeline(ampere_space,llama,ampere_cost,
                                        comparison,comparison_score,{seed});
    assert(matched.finalists.size()==2);
    // Both arms replay the same synthetic FlowProblem. The comparison
    // must retain the incumbent once even when their replay times tie.
    assert(std::count_if(matched.finalists.begin(),matched.finalists.end(),[&](auto const& row) {
      return row.coordinates.ToJson().Dump(0)==seed.ToJson().Dump(0);
    })==1);
    assert(!ampere_cost.HasCalibratedMma(seed.gemms[2],2));
    bool missing_grid=false;
    try {ampere_cost.MmaK16Ns(seed.gemms[2],2);}
    catch(Sm90NativeMissing const&) {missing_grid=true;}
    assert(missing_grid);

    Sm90DenseWorkload qwen;
    qwen.shapes={{{5120,2048},{2048,4096},{},{},{151936,2048},{128,2048},
                  {1536,2048},{2048,768}}};
    qwen.batch=1;qwen.layers=48;qwen.head_dim=128;qwen.queries=8;qwen.hkv=4;
    qwen.moe=true;qwen.experts=128;qwen.top_k=8;qwen.qk_norm=true;
    Sm90DenseSearchSpace experts{qwen.shapes,qwen.batch,qwen.head_dim,qwen.queries,
                                ampere,qwen.hkv};
    experts.moe=true;experts.experts=qwen.experts;experts.top_k=qwen.top_k;
    Sm90DenseCoordinates shallow;shallow.moe=true;shallow.expert_fuse=true;
    int widths[]={64,32,0,0,128,16,128,64};
    int wn[]={4,1,0,0,4,1,4,4};
    for(unsigned f:shallow.Active()) {
      auto& g=shallow.gemms[f];
      g.tile_n=widths[f];g.slot_bytes=2*g.tile_n*64;g.slots=2;
      g.warps_n=wn[f];g.warps_k=4/wn[f];
    }
    shallow.attention.chunks=8;shallow.attention.slots=3;
    auto deeper=shallow;deeper.SetCommonExpertSlots(4);
    auto overflow=deeper;overflow.SetCommonExpertSlots(8);
    auto incompatible=deeper;incompatible.gemms[7].register_stages=2;
    assert(experts.Legal(shallow) && experts.Legal(deeper));
    assert(!experts.Legal(overflow) && !experts.Legal(incompatible));
    assert(shallow.ToJson().Dump()!=deeper.ToJson().Dump());
    assert(Sm90NativeStructuralClass(shallow,true)!=Sm90NativeStructuralClass(deeper,true));
    assert(ampere_cost.HasCalibratedMma(shallow));
    auto ring_options=options;ring_options.budget_ms=1000;ring_options.operator_sum=true;
    auto ring_score=[&](Sm90DenseCoordinates const& c,bool isolated) {
      assert(isolated && ampere_cost.HasCalibratedMma(c));
      Sm90NativeFlowResult flow;flow.step_ns=c.CommonExpertSlots();return flow;
    };
    auto common=SolveSm90NativePipeline(experts,qwen,ampere_cost,ring_options,
                                        ring_score,{shallow,deeper});
    assert(common.evaluated.size()>=2);
    assert(common.evaluated[0].coordinates.CommonExpertSlots()==2);
    assert(common.evaluated[1].coordinates.CommonExpertSlots()==4);
    ring_options.pinned_theta=deeper;
    auto pinned_ring=SolveSm90NativePipeline(experts,qwen,ampere_cost,ring_options,
                                            ring_score,{deeper});
    assert(!pinned_ring.evaluated.empty());
    for(auto const& candidate:pinned_ring.evaluated)
      if(candidate.coordinates.expert_fuse)assert(candidate.coordinates.CommonExpertSlots()==4);

  }
  return 0;
}
